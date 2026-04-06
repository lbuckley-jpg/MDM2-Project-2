from tokenize import generate_tokens
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import capytaine as cpt

import argparse
import os
import time

cpt.set_logging('WARNING')

bem_solver = cpt.BEMSolver()


from Functions import generate_frequencies, jonswap_frequency_amplitudes, generate_buoy, plot_history, solve_with_capytaine, get_cummins_components, solve_cummins_stepwise_latch, solve_cummins_stepwise_no_latch, calc_power_absorbed, plot_power


if __name__ == '__main__':
    
    # argument parser

    parser = argparse.ArgumentParser(description="Run Wave Simulation")

    # script run parameters

    parser.add_argument("--save", type=bool, required = False)
    parser.add_argument("--tspan", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)

    # parameters for frequency generation
    parser.add_argument("--nfreqcomponents", type=int, required=False)
    parser.add_argument("--peakperiod", type=float, required=False)
    parser.add_argument("--significantwaveheight", type=float, required=False) # the mean height of the top third of largest waves

    # parameters for the buoy
    parser.add_argument("--buoymass", type=float, required=False)
    parser.add_argument("--buoyradius", type=float, required=False)

    # parameters for the water

    parser.add_argument("--waterdensity", type=float, required=False)
    parser.add_argument("--waterdepth", type=float, required=False)

    # paramters for the wave
    parser.add_argument("--wavedirection", type=float, required=False)
    parser.add_argument("--waveamplitude", type=float, required=False)

    # parameters for power take off

    parser.add_argument("--cpto", type=float, required = False)
    parser.add_argument("--kpto", type=float, required = False)

    # define args from parseer
    args = parser.parse_args()

    '''---------------run code------------------'''

    #T0 = 12.0                 # wave period [s]
    #omega0 = 2 * np.pi / T0   # rad/s
    #a0 = 2.0                  # wave amplitude [m]

    # create buoy
    buoy = generate_buoy(radius=args.buoyradius, mass=args.buoymass)

    # generate frequencies
    
    omegas, delta_omega = generate_frequencies(N=args.nfreqcomponents, Tp=args.peakperiod)
    #omegas = np.array([omega0])
    # generate amplitudes for each frequency

    wave_amplitudes = jonswap_frequency_amplitudes(omegas, delta_omega, Hs= args.significantwaveheight, Tp= args.peakperiod)
    #wave_amplitudes = np.array([a0])

    # solve with capytaine

    capytaine_dataset = solve_with_capytaine(body=buoy, omegas=omegas, wave_direction=args.wavedirection, water_depth=args.waterdepth, water_density=args.waterdensity)

    # get cummins stuff

    A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot = get_cummins_components(body=buoy, capytaine_dataset=capytaine_dataset, wave_direction=args.wavedirection, wave_amplitudes=wave_amplitudes, omegas=omegas, seed=args.seed)

    # solve cummins equation

    history_latch = solve_cummins_stepwise_latch(buoy, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, C_pto=args.cpto, K_pto=args.kpto, t_span=[0,args.tspan])

    history_no_latch = solve_cummins_stepwise_no_latch(buoy, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, C_pto=args.cpto, K_pto=args.kpto, t_span=[0,args.tspan])

    # calculate power 

    p_inst_latch, p_mean_latch = calc_power_absorbed(history_latch, args.cpto)

    p_inst_no_latch, p_mean_no_latch = calc_power_absorbed(history_no_latch, args.cpto)

    # plot the history's

    plot_history(history_latch, history_no_latch, F_ex_time)

    # plot power

    plot_power(history_latch, history_no_latch, p_inst_latch, p_inst_no_latch)

    # print power

    print(f'Average power absorption without latching: {p_mean_no_latch}')
    print(f'Average power absoption with latching: {p_mean_latch}')











