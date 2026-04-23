"""
CaptureWidthPlot.py
===================
Computes and plots the **capture width** and **capture width ratio** across
multiple sea states for fixed-control vs stepwise-latching vs Pontryagin
latching.  No cpto sweep / grid search — runs once per sea state at the
user-supplied C_PTO for fast execution.

Capture width
-------------
    CW  = P_absorbed / J        [m]

where J is the incident wave power per unit crest width in deep water:

    J   = (rho * g^2) / (64 * pi) * Hs^2 * Te      [W/m]

Te is the energy period.  For a JONSWAP spectrum with gamma = 3.3 the
standard approximation is  Te ~= 0.9 * Tp.

Capture width ratio (non-dimensional efficiency):

    CWR = CW / D                [-]

where D = 2 * buoy_radius is the buoy diameter.

Example
-------
python CaptureWidthPlot.py ^
    --tspan 300 --seed 42 ^
    --peakperiods 5.2 6.01 7.05 8.0 ^
    --significantwaveheights 0.5 1.5 2.5 3.0 ^
    --cptos 80000 120000 200000 250000 ^
    --buoymasses 427561 427561 427561 427561 ^
    --buoyradius 5.0 ^
    --save
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse

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
# Physics helpers
# ---------------------------------------------------------------------------

def incident_wave_power(Hs, Tp, rho=1025.0, g=9.81, gamma_jonswap=3.3):
    """
    function calculates Deep-water wave power per unit crest width  J  [W/m]

    Uses the energy period approximation  Te = alpha * Tp  where alpha
    depends on the JONSWAP peak-enhancement factor gamma.

    For gamma = 3.3 the commonly used value is alpha ~ 0.9.
    """
    # Te / Tp ratio for JONSWAP (Holthuijsen 2007, DNV-RP-C205)
    alpha_Te = 0.9066  # gamma = 3.3
    Te = alpha_Te * Tp

    J = (rho * g**2) / (64.0 * np.pi) * Hs**2 * Te  # [W/m]
    return J


def capture_width(P_absorbed, J):
    """Capture width  CW = P / J  [m]."""
    if J <= 0.0:
        return 0.0
    return P_absorbed / J


def capture_width_ratio(CW, diameter):
    """Capture width ratio  CWR = CW / D  [-]."""
    if diameter <= 0.0:
        return 0.0
    return CW / diameter


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='CaptureWidthPlot: capture width across sea states (no grid search).'
    )

    # run controls
    parser.add_argument('--save', action='store_true', help='Save plots to disk.')
    parser.add_argument('--tspan', type=int, required=True, help='Simulation duration [s].')
    parser.add_argument('--dt', type=float, default=0.05, help='Time step [s].')
    parser.add_argument('--seed', type=int, required=True, help='RNG seed.')

    # sea states (parallel lists — one entry per sea state)
    parser.add_argument('--peakperiods', type=float, nargs='+', required=True,
                        help='Peak periods [s], one per sea state.')
    parser.add_argument('--significantwaveheights', type=float, nargs='+', required=True,
                        help='Significant wave heights [m], one per sea state.')
    parser.add_argument('--cptos', type=float, nargs='+', required=True,
                        help='PTO damping coefficients [Ns/m], one per sea state.')
    parser.add_argument('--buoymasses', type=float, nargs='+', required=True,
                        help='Buoy masses [kg], one per sea state.')

    # buoy / water
    parser.add_argument('--buoyradius', type=float, default=2.0)
    parser.add_argument('--buoyheight', type=float, required=False)
    parser.add_argument('--buoydraft', type=float, required=False)
    parser.add_argument('--waterdensity', type=float, default=1025.0)
    parser.add_argument('--waterdepth', type=float, default=np.inf)
    parser.add_argument('--wavedirection', type=float, default=np.pi)

    # PTO / control
    parser.add_argument('--kpto', type=float, default=0.0)
    parser.add_argument('--ptoforcemax', type=float, default=np.inf)
    parser.add_argument('--nfreqcomponents', type=int, default=60)

    # Pontryagin
    parser.add_argument('--latchgain', type=float, default=0.5)
    parser.add_argument('--maxiter', type=int, default=30)
    parser.add_argument('--nprony', type=int, default=6)

    # Whether to include pontryagin (slow)
    parser.add_argument('--no-pontryagin', action='store_true',
                        help='Skip Pontryagin solver to save time.')

    args = parser.parse_args()

    # Validate and broadcast parallel lists (single value → repeated for all sea states)
    n_ss = len(args.peakperiods)

    def _broadcast(lst, name):
        if len(lst) == 1:
            return lst * n_ss
        if len(lst) != n_ss:
            raise ValueError(
                f'--{name} must have either 1 value (broadcast) or {n_ss} values '
                f'(one per peak period), got {len(lst)}.'
            )
        return lst

    args.significantwaveheights = _broadcast(args.significantwaveheights, 'significantwaveheights')
    args.cptos                  = _broadcast(args.cptos,                  'cptos')
    args.buoymasses             = _broadcast(args.buoymasses,             'buoymasses')

    diameter = 2.0 * args.buoyradius
    include_pont = not args.no_pontryagin

    # Storage
    Tps = []
    J_vals = []

    cw_fixed = []
    cw_latch = []
    cw_pont  = []

    cwr_fixed = []
    cwr_latch = []
    cwr_pont  = []

    p_mean_fixed_all = []
    p_mean_latch_all = []
    p_mean_pont_all  = []

    for idx, (Tp, Hs, cpto, buoymass) in enumerate(
            zip(args.peakperiods, args.significantwaveheights,
                args.cptos, args.buoymasses)):

        print(f'\n{"="*60}')
        print(f'Sea state {idx+1}/{n_ss}:  Tp = {Tp} s,  Hs = {Hs} m,  C_PTO = {cpto:.0f} Ns/m')
        print(f'{"="*60}')

        Tps.append(Tp)

        # Incident wave power
        J = incident_wave_power(Hs, Tp, rho=args.waterdensity)
        J_vals.append(J)
        print(f'  Incident wave power density  J = {J:.1f} W/m')

        buoy = generate_buoy(radius=args.buoyradius, mass=buoymass, height=args.buoyheight, draft=args.buoydraft)
        omegas, dw = generate_frequencies(N=args.nfreqcomponents, Tp=Tp)
        wave_amps = jonswap_frequency_amplitudes(omegas, dw, Hs=Hs, Tp=Tp)

        cap_ds = solve_with_capytaine(
            body=buoy, omegas=omegas,
            wave_direction=args.wavedirection,
            water_depth=args.waterdepth,
            water_density=args.waterdensity,
        )

        A_inf, t_kern, kern, K_heave, F_ex, F_ex_dot, B_heave = \
            get_cummins_components(
                body=buoy, capytaine_dataset=cap_ds,
                wave_direction=args.wavedirection,
                wave_amplitudes=wave_amps, omegas=omegas,
                seed=args.seed,
            )

        t_span = [0, args.tspan]

        # --- fixed control ---
        h_fixed = solve_cummins_stepwise_no_control_limited(
            buoy, A_inf, t_kern, kern, K_heave,
            F_ex, F_ex_dot,
            b_pto=cpto, K_pto=args.kpto,
            t_span=t_span, pto_force_max=args.ptoforcemax,
        )
        _, pmean_f = calc_power_absorbed(h_fixed)

        # --- stepwise latching ---
        h_latch = solve_cummins_stepwise_latch_limited(
            buoy, A_inf, t_kern, kern, K_heave,
            F_ex, F_ex_dot,
            B_pto=cpto, K_pto=args.kpto,
            t_span=t_span, pto_force_max=args.ptoforcemax,
        )
        _, pmean_l = calc_power_absorbed(h_latch)

        # --- pontryagin -----
        pmean_p = 0.0
        if include_pont:
            t_grid = np.arange(0.0, args.tspan + args.dt, args.dt)
            _, forcing_interp = build_excitation_time_series(F_ex, t_grid)
            latch_gain = args.latchgain * 1e3 * (buoymass + A_inf)
            params = (buoymass, A_inf, cpto, latch_gain, K_heave, args.buoyradius)

            h_pont_opt, _, _ = solve_pontryagin_latching_limited(
                omega=omegas, radiation_damping=B_heave,
                t_grid=t_grid, params=params,
                forcing=forcing_interp,
                max_iter=args.maxiter, n_terms=args.nprony,
            )
            _, pmean_p = calc_power(h_pont_opt['v'], cpto)

        # Record
        p_mean_fixed_all.append(pmean_f)
        p_mean_latch_all.append(pmean_l)
        p_mean_pont_all.append(pmean_p)

        cw_f = capture_width(pmean_f, J)
        cw_l = capture_width(pmean_l, J)
        cw_p = capture_width(pmean_p, J)

        cw_fixed.append(cw_f)
        cw_latch.append(cw_l)
        cw_pont.append(cw_p)

        cwr_fixed.append(capture_width_ratio(cw_f, diameter))
        cwr_latch.append(capture_width_ratio(cw_l, diameter))
        cwr_pont.append(capture_width_ratio(cw_p, diameter))

        print(f'  P_mean  —  fixed: {pmean_f/1e3:.2f} kW  |  latch: {pmean_l/1e3:.2f} kW'
              + (f'  |  pont: {pmean_p/1e3:.2f} kW' if include_pont else ''))
        print(f'  CW      —  fixed: {cw_f:.2f} m    |  latch: {cw_l:.2f} m'
              + (f'    |  pont: {cw_p:.2f} m' if include_pont else ''))
        print(f'  CWR     —  fixed: {cwr_fixed[-1]:.3f}     |  latch: {cwr_latch[-1]:.3f}'
              + (f'     |  pont: {cwr_pont[-1]:.3f}' if include_pont else ''))

    
    '''Capture width plots'''

    Tps = np.array(Tps)

    # 1.  Capture Width vs Tp
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_title('Capture Width vs Peak Period', fontsize=15)
    ax1.plot(Tps, cw_fixed, 'o-', label='Linear Control', lw=1.8, markersize=7)
    ax1.plot(Tps, cw_latch, 's-', label='Stepwise Latch', lw=1.8, markersize=7)
    if include_pont:
        ax1.plot(Tps, cw_pont, 'D--', label='Pontryagin Latch', lw=1.8, markersize=7)
    ax1.set_xlabel('Peak Period  Tp [s]', fontsize=13)
    ax1.set_ylabel('Capture Width  CW [m]', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    if args.save:
        fig1.savefig('capture_width_vs_Tp.png', dpi=300, bbox_inches='tight')
        print('\nSaved: capture_width_vs_Tp.png')

    # 2.  Capture Width Ratio vs Tp
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.set_title('Capture Width Ratio vs Peak Period', fontsize=15)
    ax2.plot(Tps, cwr_fixed, 'o-', label='Linear Control', lw=1.8, markersize=7)
    ax2.plot(Tps, cwr_latch, 's-', label='Stepwise Latch', lw=1.8, markersize=7)
    if include_pont:
        ax2.plot(Tps, cwr_pont, 'D--', label='Pontryagin Latch', lw=1.8, markersize=7)
    ax2.set_xlabel('Peak Period  Tp [s]', fontsize=13)
    ax2.set_ylabel('Capture Width Ratio  CWR = CW / D  [-]', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if args.save:
        fig2.savefig('capture_width_ratio_vs_Tp.png', dpi=300, bbox_inches='tight')
        print('Saved: capture_width_ratio_vs_Tp.png')

    # 3.  Grouped bar chart of CWR per sea state
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.set_title('Capture Width Ratio by Sea State', fontsize=15)

    labels = [f'Tp={t:.1f}\nHs={h:.1f}' for t, h in
              zip(args.peakperiods, args.significantwaveheights)]
    x = np.arange(len(labels))
    n_bars = 3 if include_pont else 2
    w = 0.8 / n_bars

    b1 = ax3.bar(x - w * (n_bars - 1) / 2, cwr_fixed, w,
                 label='Fixed Control', color='C0', edgecolor='black', linewidth=0.5)
    b2 = ax3.bar(x - w * (n_bars - 1) / 2 + w, cwr_latch, w,
                 label='Stepwise Latch', color='C1', edgecolor='black', linewidth=0.5)
    bar_groups = [b1, b2]
    if include_pont:
        b3 = ax3.bar(x - w * (n_bars - 1) / 2 + 2 * w, cwr_pont, w,
                     label='Pontryagin Latch', color='C2', edgecolor='black', linewidth=0.5)
        bar_groups.append(b3)

    for bars in bar_groups:
        for b in bars:
            ax3.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01,
                     f'{b.get_height():.3f}', ha='center', fontsize=9)

    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=10)
    ax3.set_ylabel('CWR  [-]', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    if args.save:
        fig3.savefig('capture_width_ratio_bars.png', dpi=300, bbox_inches='tight')
        print('Saved: capture_width_ratio_bars.png')

    # summary prints
    print('\n' + '='*80)
    print(f'{"Tp [s]":>8} {"Hs [m]":>8} {"J [W/m]":>10} '
          f'{"CW_fix [m]":>11} {"CW_lat [m]":>11} '
          + (f'{"CW_pon [m]":>11} ' if include_pont else '')
          + f'{"CWR_fix":>9} {"CWR_lat":>9}'
          + (f' {"CWR_pon":>9}' if include_pont else ''))
    print('-'*80)
    for i in range(n_ss):
        line = (f'{Tps[i]:8.2f} {args.significantwaveheights[i]:8.2f} '
                f'{J_vals[i]:10.1f} '
                f'{cw_fixed[i]:11.3f} {cw_latch[i]:11.3f} ')
        if include_pont:
            line += f'{cw_pont[i]:11.3f} '
        line += f'{cwr_fixed[i]:9.4f} {cwr_latch[i]:9.4f}'
        if include_pont:
            line += f' {cwr_pont[i]:9.4f}'
        print(line)
    print('='*80)

    if not args.save:
        plt.show()

    plt.close('all')
    print('\nDone.')
