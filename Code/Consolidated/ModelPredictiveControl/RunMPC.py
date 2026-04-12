import numpy as np

import capytaine as cpt

import argparse
import os
import sys

cpt.set_logging('WARNING')

bem_solver = cpt.BEMSolver()


_consolidated = os.path.dirname(os.path.dirname(__file__))
sys.path.append(_consolidated)
sys.path.append(os.path.join(_consolidated, "Pontryagin"))

from SharedCapytaineFunctions import (
    generate_frequencies,
    jonswap_frequency_amplitudes,
    generate_buoy,
    solve_with_capytaine,
    get_cummins_components,
    calc_power_absorbed,
    solve_cummins_stepwise_no_control
)

from MPCFunctions import (
    discrete_matrices_for_mpc_from_prony,
    solve_mpc,
    solve_cummins_stepwise_mpc,
    calc_power_absorbed_mpc,
    plot_history,
    plot_power
)

from PontryaginFunctions import (
    fit_prony_coefficients
)

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

    # parameters for mpc

    parser.add_argument("--nprony", type=int, default=6, help="Number of Prony terms (state size 2 + 2*n).")
    parser.add_argument("--mpc_dt", type=float, default=None, help="MPC sampling time; default uses MPCFunctions.dt.")
    parser.add_argument("--mpc_N", type=int, default=None, help="MPC horizon steps; default uses MPCFunctions.N.")

    # define args from parseer
    args = parser.parse_args()


    '''-------------Get the caytaine stuff----------------'''

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

    '''-------------Find optimal Control using MPC---------------------'''

    B_heave = capytaine_dataset["radiation_damping"].sel(
        radiating_dof="Heave", influenced_dof="Heave"
    ).values

    # first use prony's method to create a linear approximation to the radiation kernel
    prony_coeffs, _ = fit_prony_coefficients(
        t_grid=t_kernel,
        omega=omegas,
        radiation_damping=B_heave,
        mass=buoy.mass,
        added_mass_inf=A_heave_inf,
        pto_damping=args.cpto,
        hydrostatic_stiffness=K_heave,
        n_terms=args.nprony,
    )  

    # create the discrete matrices used to solve mpc
    Ad, Bd = discrete_matrices_for_mpc_from_prony(
        prony_coeffs,
        mass=buoy.mass,
        added_mass_inf=A_heave_inf,
        C_pto=args.cpto,
        K_heave=K_heave,
        dt_sample=args.mpc_dt,
    )

    t0 = 0.0
    x0 = np.zeros(Ad.shape[0])
    wave_pred = np.array([F_ex_time(t0 + args.mpc_dt * k) for k in range(args.mpc_N)])


    '''--------solve cummins equation with mpc---------'''
    x0 = np.zeros(Ad.shape[0])

    history_mpc = solve_cummins_stepwise_mpc(buoy, A_heave_inf, prony_coeffs, t_kernel, kernel, K_heave, F_ex_time, [0,args.tspan], Ad=Ad, Bd=Bd, x0=x0,dt=args.mpc_dt, n_horizon=args.mpc_N)

    history_no_control = solve_cummins_stepwise_no_control(body=buoy, A_heave_inf=A_heave_inf, t_kernel=t_kernel, kernel=kernel, K_heave=K_heave, F_ex_time=F_ex_time, F_ex_time_dot=F_ex_time_dot, C_pto=args.cpto, K_pto=args.kpto, t_span=[0, args.tspan], dt=0.05)
    

    '''---------calc power and plot-------------'''

    p_inst_mpc, p_mean_mpc = calc_power_absorbed_mpc(history_mpc)
    p_inst_no_control, p_mean_no_control = calc_power_absorbed(history_no_control)

    plot_history(history_mpc, history_no_control)
    plot_power(history_mpc, history_no_control, p_inst_mpc, p_inst_no_control)


