import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

import capytaine as cpt
cpt.set_logging('WARNING')

from SharedCapytaineFunctions import (
    generate_frequencies,
    jonswap_frequency_amplitudes,
    generate_buoy,
    solve_with_capytaine,
    get_cummins_components,
    solve_cummins_stepwise_no_control_limited,
    calc_power_absorbed,
)

from Latching.LatchingFunctions import (
    solve_cummins_stepwise_latch_limited,
)

from Pontryagin.PontryaginFunctions import (
    build_excitation_time_series,
    calc_power,
    solve_pontryagin_latching_limited,
)


# ---------------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------------

def run_sea_state(args, peakperiod, hs, cpto, include_pontryagin=False):
    """
    Run one sea state at a fixed cpto.  Returns a dict with histories and
    derived power quantities.
    """
    print(f"\n=== Running sea state: Tp={peakperiod} s, Hs={hs} m, C_PTO={cpto:.0f} Ns/m ===")

    buoy = generate_buoy(radius=args.buoyradius, mass=args.buoymass)
    omegas, delta_omega = generate_frequencies(N=args.nfreqcomponents, Tp=peakperiod)
    wave_amplitudes = jonswap_frequency_amplitudes(omegas, delta_omega, Hs=hs, Tp=peakperiod)

    cap_ds = solve_with_capytaine(
        body=buoy,
        omegas=omegas,
        wave_direction=args.wavedirection,
        water_depth=args.waterdepth,
        water_density=args.waterdensity,
    )

    A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, B_heave = \
        get_cummins_components(
            body=buoy,
            capytaine_dataset=cap_ds,
            wave_direction=args.wavedirection,
            wave_amplitudes=wave_amplitudes,
            omegas=omegas,
            seed=args.seed,
        )

    t_span = [0, args.tspan]

    # Fixed-control (no latching)
    h_fixed = solve_cummins_stepwise_no_control_limited(
        buoy, A_heave_inf, t_kernel, kernel, K_heave,
        F_ex_time, F_ex_time_dot,
        C_pto=cpto, K_pto=args.kpto,
        t_span=t_span, pto_force_max=args.ptoforcemax,
    )

    # Latching
    h_latch = solve_cummins_stepwise_latch_limited(
        buoy, A_heave_inf, t_kernel, kernel, K_heave,
        F_ex_time, F_ex_time_dot,
        C_pto=cpto, K_pto=args.kpto,
        t_span=t_span, pto_force_max=args.ptoforcemax,
    )

    p_inst_fixed, p_mean_fixed = calc_power_absorbed(h_fixed)
    p_inst_latch, p_mean_latch = calc_power_absorbed(h_latch)

    result = {
        'h_fixed': h_fixed,
        'h_latch': h_latch,
        'p_inst_fixed': p_inst_fixed,
        'p_inst_latch': p_inst_latch,
        'p_mean_fixed': p_mean_fixed,
        'p_mean_latch': p_mean_latch,
        'cpto': cpto,
        'K_heave': K_heave,
    }

    if include_pontryagin:
        t_grid = np.arange(0.0, args.tspan + args.dt, args.dt)
        _, forcing_interp = build_excitation_time_series(F_ex_time, t_grid)

        latch_gain = args.latchgain * 1e3 * (args.buoymass + A_heave_inf)
        params = (args.buoymass, A_heave_inf, cpto, latch_gain, K_heave, args.buoyradius)

        h_pont_opt, h_pont_no, _ = solve_pontryagin_latching_limited(
            omega=omegas,
            radiation_damping=B_heave,
            t_grid=t_grid,
            params=params,
            forcing=forcing_interp,
            max_iter=args.maxiter,
            n_terms=args.nprony,
        )

        _, p_mean_pont = calc_power(h_pont_opt['v'], cpto)
        result['h_pont'] = h_pont_opt
        result['p_mean_pont'] = p_mean_pont

    return result


# ---------------------------------------------------------------------------
# plotting functions
# --------------------------------------------------------------------------

def plot_instantaneous_power(result, peakperiod, hs, save=False):
    """Instantaneous absorbed power for latch vs fixed control."""
    h_fixed = result['h_fixed']
    h_latch = result['h_latch']

    # histories already trimmed inside calc_power_absorbed (first 50 pts removed)
    t_fixed = np.array(h_fixed['t'][50:])
    t_latch = np.array(h_latch['t'][50:])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(
        f'Instantaneous Absorbed Power — Tp={peakperiod} s, Hs={hs} m'
        f'  (C_PTO={result["cpto"]/1e3:.1f} kNs/m)',
        fontsize=14,
    )
    ax.plot(t_latch, result['p_inst_latch'] / 1e3, label='Stepwise Latch', alpha=0.85, lw=1.2)
    ax.plot(t_fixed, result['p_inst_fixed'] / 1e3, label='Fixed Control', alpha=0.7, lw=1.0)

    ax.axhline(result['p_mean_latch'] / 1e3, color='C0', ls='--', lw=1.0,
               label=f'Mean (latch) = {result["p_mean_latch"]/1e3:.2f} kW')
    ax.axhline(result['p_mean_fixed'] / 1e3, color='C1', ls='--', lw=1.0,
               label=f'Mean (fixed) = {result["p_mean_fixed"]/1e3:.2f} kW')

    ax.set_xlabel('Time [s]', fontsize=13)
    ax.set_ylabel('Instantaneous Power [kW]', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        fname = f'inst_power_Tp_{peakperiod}_Hs_{hs}.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f'Saved: {fname}')
    return fig


def plot_pto_force(result, peakperiod, hs, save=False):
    """PTO force time-history for latch vs fixed control."""
    h_fixed = result['h_fixed']
    h_latch = result['h_latch']
    cpto = result['cpto']

    t_fixed = np.array(h_fixed['t'])
    v_fixed = np.array(h_fixed['v'])
    c_fixed = np.array(h_fixed['c_pto'])
    f_pto_fixed = c_fixed * v_fixed   # linear PTO: F = C * v

    t_latch = np.array(h_latch['t'])
    v_latch = np.array(h_latch['v'])
    c_latch = np.array(h_latch['c_pto'])
    f_pto_latch = c_latch * v_latch

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(
        f'PTO Force — Tp={peakperiod} s, Hs={hs} m'
        f'  (C_PTO={cpto/1e3:.1f} kNs/m)',
        fontsize=14,
    )
    ax.plot(t_latch, f_pto_latch / 1e3, label='Stepwise Latch', alpha=0.85, lw=1.2)
    ax.plot(t_fixed, f_pto_fixed / 1e3, label='Fixed Control', alpha=0.7, lw=1.0)

    ax.set_xlabel('Time [s]', fontsize=13)
    ax.set_ylabel('PTO Force [kN]', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        fname = f'pto_force_Tp_{peakperiod}_Hs_{hs}.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f'Saved: {fname}')
    return fig


def plot_displacement(result, peakperiod, hs, save=False):
    """Displacement time-history: fixed / latch / Pontryagin (if present)."""
    h_fixed = result['h_fixed']
    h_latch = result['h_latch']

    t_fixed = np.array(h_fixed['t'])
    f_ex_scaled = np.array(h_fixed['F_ex']) / result['K_heave']

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(
        f'Displacement — Tp={peakperiod} s, Hs={hs} m'
        f'  (C_PTO={result["cpto"]/1e3:.1f} kNs/m)',
        fontsize=14,
    )

    ax.plot(t_fixed, f_ex_scaled, color='black', ls=':', alpha=0.35,
            label='Scaled Excitation Force (F_ex / K)')
    ax.plot(t_fixed, h_fixed['x'], label='Fixed Control', alpha=0.7, lw=1.2)
    ax.plot(h_latch['t'], h_latch['x'], label='Stepwise Latch', alpha=0.9, lw=1.2)

    if 'h_pont' in result:
        h_pont = result['h_pont']
        ax.plot(h_pont['t'], h_pont['x'], ls='--', label='Pontryagin Latch', alpha=0.9, lw=1.1)

    ax.set_xlabel('Time [s]', fontsize=13)
    ax.set_ylabel('Displacement z [m]', fontsize=13)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        fname = f'displacement_Tp_{peakperiod}_Hs_{hs}.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f'Saved: {fname}')
    return fig


def plot_mean_power_bar(result, peakperiod, hs, save=False):
    """Mean power bar chart comparing fixed / latch / Pontryagin controllers."""
    labels = ['Fixed Control', 'Stepwise Latch']
    values = [result['p_mean_fixed'] / 1e3, result['p_mean_latch'] / 1e3]
    colors = ['C0', 'C1']

    if 'p_mean_pont' in result:
        labels.append('Pontryagin Latch')
        values.append(result['p_mean_pont'] / 1e3)
        colors.append('C2')

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(
        f'Mean Absorbed Power — Tp={peakperiod} s, Hs={hs} m'
        f'  (C_PTO={result["cpto"]/1e3:.1f} kNs/m)',
        fontsize=13,
    )
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01,
                f'{v:.2f} kW', ha='center', fontsize=10)

    ax.set_ylabel('Mean Absorbed Power [kW]', fontsize=12)
    ax.tick_params(axis='x', labelsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    if save:
        fname = f'mean_power_Tp_{peakperiod}_Hs_{hs}.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f'Saved: {fname}')
    return fig


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='EtraPlots: fast single-run extra plots (no grid search).'
    )

    # run controls
    parser.add_argument('--save', action='store_true', help='Save plots to disk.')
    parser.add_argument('--tspan', type=int, required=True, help='Simulation duration [s].')
    parser.add_argument('--dt', type=float, default=0.05, help='Time step [s].')
    parser.add_argument('--seed', type=int, required=True, help='RNG seed.')

    # Sea state 1: instantaneous power + PTO force
    parser.add_argument('--tp1', type=float, required=True,
                        help='Peak period for sea state 1 [s].')
    parser.add_argument('--hs1', type=float, required=True,
                        help='Significant wave height for sea state 1 [m].')
    parser.add_argument('--cpto1', type=float, required=True,
                        help='PTO damping for sea state 1 [Ns/m].')

    # Sea state 2: displacement + mean power
    parser.add_argument('--tp2', type=float, required=True,
                        help='Peak period for sea state 2 [s].')
    parser.add_argument('--hs2', type=float, required=True,
                        help='Significant wave height for sea state 2 [m].')
    parser.add_argument('--cpto2', type=float, required=True,
                        help='PTO damping for sea state 2 [Ns/m].')

    # Buoy / water
    parser.add_argument('--buoyradius', type=float, default=2.0)
    parser.add_argument('--buoymass', type=float, required=True)
    parser.add_argument('--waterdensity', type=float, default=1025.0)
    parser.add_argument('--waterdepth', type=float, default=np.inf)
    parser.add_argument('--wavedirection', type=float, default=np.pi)

    # PTO / control
    parser.add_argument('--kpto', type=float, default=0.0)
    parser.add_argument('--ptoforcemax', type=float, default=np.inf)
    parser.add_argument('--nfreqcomponents', type=int, default=60)

    # Pontryagin options (used for sea state 2 displacement & mean power)
    parser.add_argument('--latchgain', type=float, default=0.5)
    parser.add_argument('--maxiter', type=int, default=30)
    parser.add_argument('--nprony', type=int, default=6)

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Sea state 1 - instantaneous power + PTO force
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("SEA STATE 1 — instantaneous power & PTO force")
    print("="*60)

    result1 = run_sea_state(
        args,
        peakperiod=args.tp1,
        hs=args.hs1,
        cpto=args.cpto1,
        include_pontryagin=False,
    )

    fig_inst = plot_instantaneous_power(result1, args.tp1, args.hs1, save=args.save)
    fig_pto  = plot_pto_force(result1, args.tp1, args.hs1, save=args.save)

    print(f"\nSea State 1 summary (Tp={args.tp1} s, Hs={args.hs1} m):")
    print(f"  Mean power — Fixed:  {result1['p_mean_fixed']/1e3:.2f} kW")
    print(f"  Mean power — Latch:  {result1['p_mean_latch']/1e3:.2f} kW")

    # -----------------------------------------------------------------------
    # Sea state 2 - displacement + mean power bar chart (+ Pontryagin)
    # -----------------------------------------------------------------------
    
    print("\n" + "="*60)
    print("SEA STATE 2 — displacement & mean power")
    print("="*60)

    result2 = run_sea_state(
        args,
        peakperiod=args.tp2,
        hs=args.hs2,
        cpto=args.cpto2,
        include_pontryagin=True,
    )

    fig_disp = plot_displacement(result2, args.tp2, args.hs2, save=args.save)
    fig_bar  = plot_mean_power_bar(result2, args.tp2, args.hs2, save=args.save)

    print(f"\nSea State 2 summary (Tp={args.tp2} s, Hs={args.hs2} m):")
    print(f"  Mean power — Fixed:      {result2['p_mean_fixed']/1e3:.2f} kW")
    print(f"  Mean power — Latch:      {result2['p_mean_latch']/1e3:.2f} kW")
    if 'p_mean_pont' in result2:
        print(f"  Mean power — Pontryagin: {result2['p_mean_pont']/1e3:.2f} kW")

    if not args.save:
        plt.show()

    plt.close('all')
    print("\nDone.")
