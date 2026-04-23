import numpy as np
import matplotlib.pyplot as plt
import capytaine as cpt
import os
import sys

# Ensure we can import from local directories
sys.path.append(os.getcwd())

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

def energy_balance_check(history, C_pto, K_heave, M_eff, pto_force_max=np.inf):
    """Calculates energy flow to verify physical consistency."""
    t = np.array(history['t'])
    x = np.array(history['x'])
    v = np.array(history['v'])
    f_ex = np.array(history['F_ex'])
    
    # Re-calculate forces for energy integration
    # Note: Memory force is not stored, but we can infer it from the energy balance
    # Or just check if (Work In - Work Out) matches Change in State.
    
    # 1. Work In (Excitation)
    work_ex = np.trapezoid(f_ex * v, t)
    
    # 2. Work Out (PTO)
    f_pto = np.clip(C_pto * v, -pto_force_max, pto_force_max)
    work_pto = np.trapezoid(f_pto * v, t)
    
    # 3. Change in Potential (Stiffness)
    delta_pe = 0.5 * K_heave * (x[-1]**2 - x[0]**2)
    
    # 4. Change in Kinetic (Mass + Added Mass)
    delta_ke = 0.5 * M_eff * (v[-1]**2 - v[0]**2)
    
    # For a balanced system: Work_Ex = Work_PTO + Work_Radiation + Work_Viscous + Delta(PE+KE)
    # We'll report the 'Net Energy' which includes Radiation and Viscous losses.
    net_dissipated = work_ex - work_pto - delta_pe - delta_ke
    
    return {
        'work_in': work_ex,
        'work_pto': work_pto,
        'delta_state': delta_pe + delta_ke,
        'net_dissipated': net_dissipated
    }

def main():
    # parameters
    # Resonant period for 10m diameter sphere (M+Ainf ~ 520t, K ~ 770kN/m)
    peakperiod = 5.2 
    significantwaveheight = 2.5
    buoymass = 261800.0 # neutrally buoyant half-sphere
    radius = 5.0
    cpto_centre = 80000.0 # Increased damping for resonance
    kpto = 0.0
    tspan = 60
    dt = 0.05
    seed = 42
    latch_gain_scalar = 0.5
    force_limit = 100000.0 # 100 kN realistic limit
    
    print(f"--- QuickCheck Simulation Parameters (Resonant Case) ---")
    print(f"Tp: {peakperiod}s (Resonant), Hs: {significantwaveheight}m")
    print(f"Mass: {buoymass}kg, C_pto: {cpto_centre} Ns/m")

    # 1. Setup Buoy and Waves
    buoy = generate_buoy(radius=radius, mass=buoymass)
    omegas, delta_omega = generate_frequencies(N=40, Tp=peakperiod)
    wave_amplitudes = jonswap_frequency_amplitudes(omegas, delta_omega, Hs=significantwaveheight, Tp=peakperiod)

    # 2. BEM Solve
    capytaine_dataset = solve_with_capytaine(body=buoy, omegas=omegas, wave_direction=np.pi, water_depth=np.inf, water_density=1000.0)

    # 3. Get Cummins Components
    A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, B_heave = get_cummins_components(
        body=buoy, capytaine_dataset=capytaine_dataset, wave_direction=np.pi, 
        wave_amplitudes=wave_amplitudes, omegas=omegas, seed=seed
    )
    M_eff = buoymass + A_heave_inf

    # 4. Solvers
    # 4a. Cummins Fixed Control
    print("\nRunning Fixed Control...")
    history_fixed = solve_cummins_stepwise_no_control_limited(
        buoy, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, 
        C_pto=cpto_centre, K_pto=kpto, t_span=[0, tspan], dt=dt
    )

    # 4b. Cummins Stepwise Latch (Unlimited)
    print("Running Stepwise Latch (Unlimited Force)...")
    history_latch = solve_cummins_stepwise_latch_limited(
        buoy, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, 
        C_pto=cpto_centre, K_pto=kpto, t_span=[0, tspan], dt=dt
    )
    
    # 4c. Cummins Stepwise Latch (FORCE LIMITED)
    print(f"Running Stepwise Latch (Force Limited @ {force_limit/1000:.0f}kN)...")
    history_latch_lim = solve_cummins_stepwise_latch_limited(
        buoy, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, 
        C_pto=cpto_centre, K_pto=kpto, t_span=[0, tspan], dt=dt, pto_force_max=force_limit
    )

    # 4d. Pontryagin Latch
    print("Running Pontryagin Latch...")
    t_grid = np.arange(0.0, tspan + dt, dt)
    _, forcing_interp = build_excitation_time_series(F_ex_time, t_grid)
    latch_gain = latch_gain_scalar * 1e3 * M_eff
    params = (buoymass, A_heave_inf, cpto_centre, latch_gain, K_heave, radius)
    history_pont, history_pont_fixed, _ = solve_pontryagin_latching_limited(
        omega=omegas, radiation_damping=B_heave, t_grid=t_grid, params=params, 
        forcing=forcing_interp, max_iter=20, n_terms=6
    )

    # 5. Extract Power
    p_inst_fixed, p_mean_fixed = calc_power_absorbed(history_fixed)
    p_inst_latch, p_mean_latch = calc_power_absorbed(history_latch)
    p_inst_latch_lim, p_mean_latch_lim = calc_power_absorbed(history_latch_lim)
    p_inst_pont, p_mean_pont = calc_power(history_pont["v"], cpto_centre)
    p_inst_pont_fixed, p_mean_pont_fixed = calc_power(history_pont_fixed["v"], cpto_centre)

    # 6. Energy Balance Reports
    print(f"\n--- Energy Balance Verification ---")
    for name, hist, fmax in [("Fixed", history_fixed, np.inf), ("Latch (Unlim)", history_latch, np.inf), ("Latch (Lim)", history_latch_lim, force_limit)]:
        eb = energy_balance_check(hist, cpto_centre, K_heave, M_eff, fmax)
        leak_ratio = eb['net_dissipated'] / eb['work_in'] if eb['work_in'] != 0 else 0
        print(f"{name:12s} | Dissipated (Rad+Visc): {eb['net_dissipated']/1e3:7.1f} kJ | Net Power: {p_mean_fixed if name=='Fixed' else (p_mean_latch if name=='Latch (Unlim)' else p_mean_latch_lim):.1f} W")

    print(f"\n--- Power Comparison ---")
    print(f"Fixed Control (Cummins):   {p_mean_fixed:.2f} W")
    print(f"Fixed Control (Pontr Ref): {p_mean_pont_fixed:.2f} W")
    print(f"Stepwise Latch (Unlim):   {p_mean_latch:.2f} W")
    print(f"Stepwise Latch (Lim100k): {p_mean_latch_lim:.2f} W")
    print(f"Pontryagin Latch (Opt):   {p_mean_pont:.2f} W")

    # 7. Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # 7a. Displacement Plot
    axes[0].set_title(f"Resonant Displacement (z) @ Tp={peakperiod}s", fontsize=14)
    axes[0].plot(history_fixed['t'], history_fixed['x'], label='Fixed Control (Cummins)', alpha=0.5)
    axes[0].plot(history_pont_fixed['t'], history_pont_fixed['x'], label='Fixed Control (Pontr Ref)', ls=':', color='blue', alpha=0.7)
    axes[0].plot(history_latch['t'], history_latch['x'], label='Stepwise Latch (Unlimited)', alpha=0.8)
    axes[0].plot(history_latch_lim['t'], history_latch_lim['x'], label=f'Stepwise Latch (Lim {force_limit/1e3:.0f}kN)', ls='-.', color='black')
    axes[0].plot(history_pont['t'], history_pont['x'], label='Pontryagin Latch', ls='--', alpha=0.8)
    axes[0].axhline(y=radius, color='r', linestyle=':', label='Physical Limit')
    axes[0].axhline(y=-radius, color='r', linestyle=':')
    axes[0].set_ylabel("z (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right', ncol=2)

    # 7b. Power Plot
    axes[1].set_title("Instantaneous Power Absorbed", fontsize=14)
    axes[1].plot(history_fixed['t'][50:], p_inst_fixed, label='Fixed Control', alpha=0.5)
    axes[1].plot(history_latch['t'][50:], p_inst_latch, label='Latch (Unlimited)', alpha=0.8)
    axes[1].plot(history_latch_lim['t'][50:], p_inst_latch_lim, label=f'Latch (Lim {force_limit/1e3:.0f}kN)', color='black', lw=2)
    axes[1].set_ylabel("Power (W)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    
    # 7c. Force Plot
    axes[2].set_title("PTO Force (Clipping/Saturation)", fontsize=14)
    f_pto_unlim = cpto_centre * np.array(history_latch['v'])
    f_pto_lim = np.clip(cpto_centre * np.array(history_latch_lim['v']), -force_limit, force_limit)
    axes[2].plot(history_latch['t'], f_pto_unlim/1e3, label='Force (Unlimited)', alpha=0.5)
    axes[2].plot(history_latch_lim['t'], f_pto_lim/1e3, label=f'Force (Lim {force_limit/1e3:.0f}kN)', color='black')
    axes[2].set_ylabel("Force (kN)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('QuickCheck_Results.png')
    print("\nResults saved to 'QuickCheck_Results.png'")

if __name__ == '__main__':
    main()
