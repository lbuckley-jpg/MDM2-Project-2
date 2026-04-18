import numpy as np
import matplotlib.pyplot as plt

import capytaine as cpt

import argparse
import os
import sys

cpt.set_logging('WARNING')

bem_solver = cpt.BEMSolver()


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


if __name__ == '__main__':
    
    # argument parser

    parser = argparse.ArgumentParser(description="Run Wave Simulation")

    # script run parameters
    parser.add_argument("--save", action='store_true', help="Save plots")
    parser.add_argument("--tspan", type=int, required=True)
    parser.add_argument("--dt", type=float, default=0.05, help="Time step for output grid [s].")

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runs", type=int, default=1, help='number of runs for each sea state')

    # parameters for frequency generation with multiple sea states
    parser.add_argument("--nfreqcomponents", type=int, default=40, required=False)
    parser.add_argument("--peakperiods", type=float, nargs='+', required=False)
    parser.add_argument("--significantwaveheights", type=float, nargs='+', required=False) # the mean height of the top third of largest waves

    # parameters for the buoy
    parser.add_argument("--buoymasses", type=float, nargs='+', required=False)
    parser.add_argument("--buoyradius", type=float, default=5.0, required=False)

    # parameters for the water
    parser.add_argument("--waterdensity", type=float, default=1000.0, required=False)
    parser.add_argument("--waterdepth", type=float, default=np.inf, required=False)

    # paramters for the wave
    parser.add_argument("--wavedirection", type=float, default=np.pi, required=False)
    parser.add_argument("--waveamplitude", type=float, required=False)

    # parameters for power take off tuned for peak frequency
    parser.add_argument("--cptos", type=float, nargs='+', required = False)
    parser.add_argument("--kpto", type=float, default=0.0, required = False)

    # pontryagin parameters
    parser.add_argument("--latchgain", type=float, default=0.5, required=False)
    parser.add_argument("--maxiter", type=int, default=20, required=False)
    parser.add_argument("--nprony", type=int, default=6, required=False)

    # define args from parseer
    args = parser.parse_args()

    '''---------------run code------------------'''

    ######## loop through peak frequencies that we have analysed

    for peakperiod, significantwaveheight, buoymass, cpto_centre in zip(args.peakperiods, args.significantwaveheights, args.buoymasses, args.cptos):

        # create buoy
        buoy = generate_buoy(radius=args.buoyradius, mass=buoymass)

        # generate frequencies
        omegas, delta_omega = generate_frequencies(N=args.nfreqcomponents, Tp=peakperiod)
        #omegas = np.array([omega0])
        # generate amplitudes for each frequency


        wave_amplitudes = jonswap_frequency_amplitudes(omegas, delta_omega, Hs= significantwaveheight, Tp = peakperiod)
        #wave_amplitudes = np.array([a0])

        # solve with capytaine
        capytaine_dataset = solve_with_capytaine(body=buoy, omegas=omegas, wave_direction=args.wavedirection, water_depth=args.waterdepth, water_density=args.waterdensity)

        ####### loop through the runs for this wave frequency energy spectrum

        # sweep multipliers for c_pto to tune for non-linear latching effects
        cpto_multipliers = [0.8, 0.9, 1.0, 1.1, 1.2]
        
        # dictionary to store sweep results
        sweep_results = {}

        for cpto_multiplier in cpto_multipliers:

            cpto = cpto_centre * cpto_multiplier

            # create dictionary to store results for this specific cpto multiplier
            run_results = {

                'latch_control': {
                    'run_power_means': []

                },

                'fixed_control': {
                    'run_power_means': []

                },

                'pontryagin_latch_control' : {
                    'run_power_means': []

                }
            }

            for run in range(args.runs):

                # generate a unique seed for each run to change phase
                run_seed = args.seed + run

                # get cummins stuff
        
                A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, B_heave = get_cummins_components(body=buoy, capytaine_dataset=capytaine_dataset, wave_direction=args.wavedirection, wave_amplitudes=wave_amplitudes, omegas=omegas, seed=run_seed)

                # solve cummins equation
                history_latch_control = solve_cummins_stepwise_latch_limited(buoy, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, C_pto=cpto, K_pto=args.kpto, t_span=[0,args.tspan])

                history_fixed_control = solve_cummins_stepwise_no_control_limited(buoy, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, C_pto=cpto, K_pto=args.kpto, t_span=[0,args.tspan])

                # calculate power 

                p_latch_control_inst, p_latch_run_mean = calc_power_absorbed(history_latch_control)

                p_fixed_control_inst, p_fixed_run_mean = calc_power_absorbed(history_fixed_control)

                # record results

                run_results['latch_control']['run_power_means'].append(p_latch_run_mean)
                run_results['fixed_control']['run_power_means'].append(p_fixed_run_mean)

                '''-----Pontryagin Stuff ----------------'''

                # create time grid
                t_grid = np.arange(0.0, args.tspan + args.dt, args.dt)

                # build excitation time series
                forcing_values, forcing_interp = build_excitation_time_series(F_ex_time, t_grid)

                # extract radiation damping and added mass for Pontryagin memory model.
                
                # Must multiply by a massive scalar, otherwise 'u=1' acts as a weak parasitic damper 
                # (since 0.5*mass ~ 50k Ns/m) which dissipates energy rather than rigorously stopping motion.
                latch_gain = args.latchgain * 1e3 * (buoymass + A_heave_inf)

                params = (buoymass, A_heave_inf, cpto, latch_gain, K_heave, args.buoyradius)

                # solve optimal latch policy and baseline no-latch case.
                history_opt, _, _ = solve_pontryagin_latching_limited(omega=omegas, radiation_damping=B_heave, t_grid=t_grid, params=params, forcing=forcing_interp, max_iter=args.maxiter, n_terms=args.nprony)

                # compare absorbed power.
                p_inst_opt, p_mean_opt = calc_power(history_opt["v"], cpto)

                # record pontryagin results
                run_results['pontryagin_latch_control']['run_power_means'].append(p_mean_opt)

        # store sweep results for this modifier after finishing all runs
            sweep_results[cpto_multiplier] = run_results

        '''-----Analysis & Plotting ----------------'''

        # extract overall expectations for each control type
        mean_powers_fixed = []
        mean_powers_latch = []
        mean_powers_pontryagin = []

        for mult in cpto_multipliers:
            # calculate the 'mean of expected means' traversing phases
            val_fixed = np.mean(sweep_results[mult]['fixed_control']['run_power_means'])
            val_latch = np.mean(sweep_results[mult]['latch_control']['run_power_means'])
            val_pontryagin = np.mean(sweep_results[mult]['pontryagin_latch_control']['run_power_means'])

            mean_powers_fixed.append(val_fixed)
            mean_powers_latch.append(val_latch)
            mean_powers_pontryagin.append(val_pontryagin)

        # print best values to console natively
        best_fixed_idx = np.argmax(mean_powers_fixed)
        best_latch_idx = np.argmax(mean_powers_latch)
        best_pontryagin_idx = np.argmax(mean_powers_pontryagin)

        print(f"\n--- Results for Peak Period: {peakperiod}s ---")
        print(f"Optimal Fixed C_PTO Mult: {cpto_multipliers[best_fixed_idx]} -> Max Expected Power: {mean_powers_fixed[best_fixed_idx]:.2f} W")
        print(f"Optimal Latch C_PTO Mult: {cpto_multipliers[best_latch_idx]} -> Max Expected Power: {mean_powers_latch[best_latch_idx]:.2f} W")
        print(f"Optimal Pontryagin C_PTO Mult: {cpto_multipliers[best_pontryagin_idx]} -> Max Expected Power: {mean_powers_pontryagin[best_pontryagin_idx]:.2f} W")

        # executing plots
        plt.figure(figsize=(10,6))
        plt.title(f'Expected Power vs C_PTO (Tp={peakperiod}s, Hs={significantwaveheight}m)', fontsize=16)
        
        plt.plot(cpto_multipliers, mean_powers_fixed, label='fixed control', marker='o')
        plt.plot(cpto_multipliers, mean_powers_latch, label='latch control', marker='s')
        plt.plot(cpto_multipliers, mean_powers_pontryagin, label='pontryagin latch', marker='^')
        
        # highlighting stars across optimal point coordinates
        plt.scatter(cpto_multipliers[best_fixed_idx], mean_powers_fixed[best_fixed_idx], color='blue', s=200, marker='*')
        plt.scatter(cpto_multipliers[best_latch_idx], mean_powers_latch[best_latch_idx], color='orange', s=200, marker='*')
        plt.scatter(cpto_multipliers[best_pontryagin_idx], mean_powers_pontryagin[best_pontryagin_idx], color='green', s=200, marker='*')

        plt.xlabel('c_pto multiplier (vs linear peak optimum)', fontsize=14)
        plt.ylabel('expected average absorbed power [w]', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # process figure outputs
        if args.save:
            save_name = f'cpto_optimization_Tp_{peakperiod}_Hs_{significantwaveheight}.png'
            plt.savefig(save_name, dpi=300, bbox_inches='tight')
            print(f"saved figure: {save_name}")
        else:
            plt.show()

        plt.close()
