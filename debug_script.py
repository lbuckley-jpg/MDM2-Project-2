import numpy as np
import matplotlib.pyplot as plt
import capytaine as cpt
import argparse
from SharedCapytaineFunctions import (
    generate_frequencies,
    jonswap_frequency_amplitudes,
    generate_buoy,
    solve_with_capytaine,
    get_cummins_components,
    solve_cummins_stepwise_no_control,
    calc_power_absorbed,
)
from Latching.LatchingFunctions import solve_cummins_stepwise_latch

cpt.set_logging('WARNING')

def main():
    peakperiod = 10.0
    significantwaveheight = 2.5
    buoymass = 5000.0
    cpto = 15000.0
    kpto = 0.0
    
    buoy = generate_buoy(radius=5.0, mass=buoymass)
    omegas, delta_omega = generate_frequencies(N=40, Tp=peakperiod)
    wave_amplitudes = jonswap_frequency_amplitudes(omegas, delta_omega, Hs=significantwaveheight, Tp=peakperiod)
    
    capytaine_dataset = solve_with_capytaine(body=buoy, omegas=omegas, wave_direction=np.pi, water_depth=np.inf, water_density=1000.0)
    
    A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, B_heave = get_cummins_components(
        body=buoy, capytaine_dataset=capytaine_dataset, wave_direction=np.pi, 
        wave_amplitudes=wave_amplitudes, omegas=omegas, seed=123
    )
    
    history_latch = solve_cummins_stepwise_latch(
        buoy, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, 
        C_pto=cpto*1.2, K_pto=kpto, t_span=[0, 60]
    )
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(history_latch['t'], history_latch['x'], label='x (Latch)')
    axs[0].set_title("Displacement")
    axs[0].legend()
    
    axs[1].plot(history_latch['t'], history_latch['v'], label='v (Latch)')
    axs[1].set_title("Velocity")
    axs[1].legend()
    
    p_inst = np.array(history_latch['c_pto']) * np.array(history_latch['v'])**2
    axs[2].plot(history_latch['t'], p_inst, label='P_inst (Latch)')
    axs[2].set_title("Instantaneous Power")
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig('debug_latch.png')
    print(f"Max velocity: {np.max(history_latch['v']):.2f}")
    print(f"Min velocity: {np.min(history_latch['v']):.2f}")
    print(f"Max power: {np.max(p_inst):.2f}")
    
if __name__ == '__main__':
    main()
