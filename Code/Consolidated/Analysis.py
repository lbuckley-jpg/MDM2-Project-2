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
    parser.add_argument("--nfreqcomponents", type=int, default = 60, required=False)
    parser.add_argument("--peakperiods", type=float, nargs='+', required=False)
    parser.add_argument("--significantwaveheights", type=float, nargs='+', required=False) # the mean height of the top third of largest waves

    # parameters for the buoy
    parser.add_argument("--buoymasses", type=float, nargs='+', required=False)
    parser.add_argument("--buoyradius", type=float, default=2.0, required=False)
    parser.add_argument("--buoyheight", type=float, default=4.0, required=False)
    parser.add_argument("--buoydraft", type=float, default=2.0, required=False)

    # parameters for the water
    parser.add_argument("--waterdensity", type=float, default=1025.0, required=False)
    parser.add_argument("--waterdepth", type=float, default=np.inf, required=False)

    # paramters for the wave
    parser.add_argument("--wavedirection", type=float, default=np.pi, required=False)
    parser.add_argument("--waveamplitude", type=float, required=False)

    # parameters for power take off tuned for peak frequency
    parser.add_argument("--cptos", type=float, nargs='+', required = False)
    parser.add_argument("--kpto", type=float, default=0.0, required = False)

    # pontryagin parameters
    parser.add_argument("--latchgain", type=float, default=0.5, required=False)
    parser.add_argument("--maxiter", type=int, default=30, required=False)
    parser.add_argument("--nprony", type=int, default=6, required=False)
    parser.add_argument("--ptoforcemax", type=float, default=np.inf, help="Maximum PTO force [N]")

    # define args from parseer
    args = parser.parse_args()

    '''---------------run code------------------'''

    ######## loop through peak frequencies that we have analysed
    
    global_sweep_data = {}
    
    peak_powers_linear = []
    peak_powers_latch = []
    peak_powers_pont_latch = []
    sea_state_labels = []

    for peakperiod, significantwaveheight, buoymass, cpto_centre in zip(args.peakperiods, args.significantwaveheights, args.buoymasses, args.cptos):

        # create buoy
        buoy = generate_buoy(radius=args.buoyradius, mass=buoymass, height=args.buoyheight, draft=args.buoydraft)

        # generate frequencies
        omegas, delta_omega = generate_frequencies(N=args.nfreqcomponents, Tp=peakperiod)
        #omegas = np.array([omega0])
        # generate amplitudes for each frequency


        wave_amplitudes = jonswap_frequency_amplitudes(omegas, delta_omega, Hs= significantwaveheight, Tp = peakperiod)
        #wave_amplitudes = np.array([a0])

        # solve with capytaine
        capytaine_dataset = solve_with_capytaine(body=buoy, omegas=omegas, wave_direction=args.wavedirection, water_depth=args.waterdepth, water_density=args.waterdensity)

        ####### loop through the runs for this wave frequency energy spectrum

        # sweep multipliers logarithmicly: from 0.05x to 5.0x
        cpto_multipliers = np.logspace(np.log10(0.05), np.log10(5.0), 11)
        
        # To store best performing run data for high-fidelity plot
        best_latch_stats = {'power': -np.inf, 'history': None, 'cpto': 0.0, 'K_heave': 0.0}
        
        # dictionary to store sweep results
        sweep_results = {}
        
        # To store high-res data for representative plot (cpto=1.0, run=0)
        rep_histories = {}

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
                },
                'prony_fix' : {
                    'run_power_means': []
                }
            }

            for run in range(args.runs):

                # generate a unique seed for each run to change phase
                run_seed = args.seed + run

                # get cummins stuff
        
                A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, B_heave = get_cummins_components(body=buoy, capytaine_dataset=capytaine_dataset, wave_direction=args.wavedirection, wave_amplitudes=wave_amplitudes, omegas=omegas, seed=run_seed)

                # solve cummins equation
                history_latch_control = solve_cummins_stepwise_latch_limited(buoy, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, B_pto=cpto, K_pto=args.kpto, t_span=[0,args.tspan], pto_force_max=args.ptoforcemax)

                history_fixed_control = solve_cummins_stepwise_no_control_limited(buoy, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, b_pto=cpto, K_pto=args.kpto, t_span=[0,args.tspan], pto_force_max=args.ptoforcemax)

                # calculate power 
                p_latch_control_inst, p_latch_run_mean = calc_power_absorbed(history_latch_control)
                p_fixed_control_inst, p_fixed_run_mean = calc_power_absorbed(history_fixed_control)

                # Capture 'Best Overall Latch' run for displacing later
                if p_latch_run_mean > best_latch_stats['power']:
                    best_latch_stats['power'] = p_latch_run_mean
                    best_latch_stats['history'] = history_latch_control
                    best_latch_stats['fixed_ref'] = history_fixed_control
                    best_latch_stats['cpto'] = cpto
                    best_latch_stats['K_heave'] = K_heave

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
                history_opt, history_no, _ = solve_pontryagin_latching_limited(omega=omegas, radiation_damping=B_heave, t_grid=t_grid, params=params, forcing=forcing_interp, max_iter=args.maxiter, n_terms=args.nprony)

                # compare absorbed power.
                p_inst_opt, p_mean_opt = calc_power(history_opt["v"], cpto)
                p_inst_no, p_mean_no = calc_power(history_no["v"], cpto)

                # record pontryagin results
                run_results['pontryagin_latch_control']['run_power_means'].append(p_mean_opt)
                run_results['prony_fix']['run_power_means'].append(p_mean_no)
                
                if p_latch_run_mean >= best_latch_stats['power']:
                    # Also capture Pontryagin for the SAME run if it's the best latch
                    best_latch_stats['pont'] = history_opt

        # store sweep results for this modifier after finishing all runs
            sweep_results[cpto_multiplier] = run_results

        '''-----Analysis & Plotting ----------------'''

        '''-----Analysis & Plotting ----------------'''

        mean_powers_fixed = []
        mean_powers_latch = []
        mean_powers_pontryagin = []
        mean_powers_prony_fix = []
        
        var_powers_fixed = []
        var_powers_latch = []
        var_powers_pontryagin = []

        for mult in cpto_multipliers:
            results = sweep_results[mult]
            
            # calculate the 'mean of expected means' traversing phases
            val_fixed = np.mean(results['fixed_control']['run_power_means'])
            val_latch = np.mean(results['latch_control']['run_power_means'])
            val_pontryagin = np.mean(results['pontryagin_latch_control']['run_power_means'])
            val_prony_fix = np.mean(results['prony_fix']['run_power_means'])

            mean_powers_fixed.append(val_fixed)
            mean_powers_latch.append(val_latch)
            mean_powers_pontryagin.append(val_pontryagin)
            mean_powers_prony_fix.append(val_prony_fix)
            
            # calculate variances if multiple runs exist
            if args.runs > 1:
                var_powers_fixed.append(np.var(results['fixed_control']['run_power_means']))
                var_powers_latch.append(np.var(results['latch_control']['run_power_means']))
                var_powers_pontryagin.append(np.var(results['pontryagin_latch_control']['run_power_means']))

        # print best values to console natively
        best_fixed_idx = np.argmax(mean_powers_fixed)
        best_latch_idx = np.argmax(mean_powers_latch)
        best_pontryagin_idx = np.argmax(mean_powers_pontryagin)

        # Collect data for the Comparative global Plot (Log-X)
        global_sweep_data[f"SS{len(global_sweep_data)+1}: Tp={peakperiod}s, Hs={significantwaveheight}m"] = {
            'cptos': np.array(cpto_multipliers) * cpto_centre,
            'powers': np.array(mean_powers_latch)
        }

        # Collect peak values for consolidated bar chart
        peak_powers_linear.append(max(mean_powers_fixed))
        peak_powers_latch.append(max(mean_powers_latch))
        peak_powers_pont_latch.append(max(mean_powers_pontryagin))
        sea_state_labels.append(f"Tp={peakperiod}\nHs={significantwaveheight}")

        print(f"\n--- Statistical Results for Peak Period: {peakperiod}s ---")
        print(f"Optimal Fixed C_PTO Mult: {cpto_multipliers[best_fixed_idx]:4.1f} | p_mean: {mean_powers_fixed[best_fixed_idx]:8.1f} W" + (f" | p_var: {var_powers_fixed[best_fixed_idx]:.1f}" if args.runs > 1 else ""))
        print(f"Optimal Latch C_PTO Mult: {cpto_multipliers[best_latch_idx]:4.1f} | p_mean: {mean_powers_latch[best_latch_idx]:8.1f} W" + (f" | p_var: {var_powers_latch[best_latch_idx]:.1f}" if args.runs > 1 else ""))
        print(f"Optimal Pontryagin Latch: {cpto_multipliers[best_pontryagin_idx]:4.1f} | p_mean: {mean_powers_pontryagin[best_pontryagin_idx]:8.1f} W" + (f" | p_var: {var_powers_pontryagin[best_pontryagin_idx]:.1f}" if args.runs > 1 else ""))
        print(f"Prony's Fix Baseline:    {cpto_multipliers[np.argmax(mean_powers_prony_fix)]:4.1f} | p_mean: {mean_powers_prony_fix[np.argmax(mean_powers_prony_fix)]:8.1f} W")

        # 1 displacement plot of the best run found
        if best_latch_stats['history'] is not None:
            plt.figure(figsize=(12, 6))
            plt.title(f'Optimal Latch Performance (C_PTO={best_latch_stats["cpto"]/1e3:.1f}kN, Tp={peakperiod}s)', fontsize=16)
            
            h_latch = best_latch_stats['history']
            h_fixed = best_latch_stats['fixed_ref']
            h_pont = best_latch_stats.get('pont', None)
            
            # Scaled excitation Force
            t_fixed = np.array(h_fixed['t'])
            f_ex_scaled = np.array(h_fixed['F_ex']) / best_latch_stats['K_heave']
            plt.plot(t_fixed, f_ex_scaled, label='Scaled Excitation Force (F_ex/K)', color='black', ls=':', alpha=0.4)
            
            plt.plot(t_fixed, h_fixed['x'], label='Fixed Control (Cummins)', alpha=0.7)
            plt.plot(h_latch['t'], h_latch['x'], label='Stepwise Latch (Best)', alpha=0.9)
            if h_pont is not None:
                plt.plot(h_pont['t'], h_pont['x'], label='Pontryagin Latch', ls='--', alpha=0.9)
            
            plt.xlabel('Time [s]', fontsize=14)
            plt.ylabel('Displacement z [m]', fontsize=14)
            plt.legend(loc='upper right', fontsize=10, ncol=2)
            plt.grid(True, alpha=0.3)
            
            if args.save:
                rep_name = f'displacement_history_Tp_{peakperiod}_Hs_{significantwaveheight}.png'
                plt.savefig(rep_name, dpi=300, bbox_inches='tight')
                print(f"Saved displacement plot: {rep_name}")

        if not args.save:
            plt.show()

        plt.close('all')

    # 2 CONSOLIDATED OPTIMIZATION PLOT of the bpto (Comparative Log-Search)
    if global_sweep_data:
        plt.figure(figsize=(10, 8))
        plt.title('Comparison of Latching Performance across Sea States', fontsize=16)
        
        for label, data in global_sweep_data.items():
            plt.semilogx(data['Bptos'], data['powers'] / 1e3, 'o-', label=label)
            
        plt.xlabel(r"$C_{PTO}$ [Ns/m]", fontsize=14)
        plt.ylabel("Mean absorbed power [kW]", fontsize=14)
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(fontsize=9)
        
        if args.save:
            plt.savefig('optimization_comparative.png', dpi=300, bbox_inches='tight')
            print("Saved consolidated optimization plot: optimization_comparative.png")
        else:
            plt.show()
        plt.close()

    # 3. CONSOLIDATED BAR CHART
    if peak_powers_linear:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        x = np.arange(len(sea_state_labels))
        w = 0.25
        
        b1 = ax3.bar(x - w, np.array(peak_powers_linear)/1e3, w, label="linear control", color='C0')
        b2 = ax3.bar(x, np.array(peak_powers_latch)/1e3, w, label="stepwise latch", color='C1')
        b3 = ax3.bar(x + w, np.array(peak_powers_pont_latch)/1e3, w, label="pontryagin latch", color='C2')
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(sea_state_labels, fontsize=10)
        ax3.set_ylabel("Peak mean absorbed power [kW]", fontsize=12)
        ax3.set_title("Best control performance per sea state", fontsize=16)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add labels on top of bars
        for bars in [b1, b2, b3]:
            for b in bars:
                ax3.text(b.get_x() + b.get_width()/2, b.get_height()*1.01,
                         f"{b.get_height():.2f}", ha='center', fontsize=9)
        
        plt.tight_layout()
        if args.save:
            save_name_bar = 'peak_performance_comparison.png'
            plt.savefig(save_name_bar, dpi=300, bbox_inches='tight')
            print(f"Saved bar chart: {save_name_bar}")
        else:
            plt.show()
