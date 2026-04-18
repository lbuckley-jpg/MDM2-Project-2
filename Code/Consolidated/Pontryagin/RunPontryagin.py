import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from SharedCapytaineFunctions import (
    generate_frequencies,
    generate_buoy,
    get_cummins_components,
    jonswap_frequency_amplitudes,
    solve_with_capytaine,
)
from PontryaginFunctions import (
    build_excitation_time_series,
    calc_power,
    plot_pontryagin_results,
    solve_no_latch,
    solve_pontryagin_latching,
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run Pontryagin latching control simulation."
    )

    # Runtime controls.
    parser.add_argument("--tspan", type=float, default=300.0, help="Total simulation time [s].")
    parser.add_argument("--dt", type=float, default=0.05, help="Time step for output grid [s].")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for wave phases.")
    parser.add_argument("--plot", action="store_true", help="Plot latch, motion and power.")

    # Frequency and sea state.
    parser.add_argument("--nfreqcomponents", type=int, default=52, help="Number of frequency components.")
    parser.add_argument("--peakperiod", type=float, default=12.0, help="Peak wave period Tp [s].")
    parser.add_argument("--significantwaveheight", type=float, default=2.0, help="Significant wave height Hs [m].")

    # Buoy and water.
    parser.add_argument("--buoymass", type=float, default=314000.0, help="Buoy mass [kg].")
    parser.add_argument("--buoyradius", type=float, default=5.0, help="Buoy radius [m].")
    parser.add_argument("--waterdensity", type=float, default=1025.0, help="Water density [kg/m^3].")
    parser.add_argument("--waterdepth", type=float, default=np.inf, help="Water depth [m].")
    parser.add_argument("--wavedirection", type=float, default=0.0, help="Wave direction [rad].")

    # PTO and latching.
    parser.add_argument("--cpto", type=float, default=50000.0, help="PTO damping coefficient [N s/m].")
    parser.add_argument("--latchgain", type=float, default=80.0, help="Latch gain multiplier (G = latchgain * (m + m_inf)).")
    parser.add_argument("--nprony", type=int, default=7, help="Number of Prony terms.")
    parser.add_argument("--maxiter", type=int, default=100, help="Max Pontryagin iterations.")


    args = parser.parse_args()

    # 1) Build buoy and solve hydrodynamics with shared Capytaine helpers.
    
    buoy = generate_buoy(radius=args.buoyradius, mass=args.buoymass)

    
    omegas, delta_omega = generate_frequencies(N=args.nfreqcomponents, Tp=args.peakperiod)
    

    wave_amplitudes = jonswap_frequency_amplitudes(omega=omegas, delta_omega=delta_omega, Hs=args.significantwaveheight, Tp=args.peakperiod)

   
    capytaine_dataset = solve_with_capytaine(body=buoy, omegas=omegas, wave_direction=args.wavedirection, water_depth=args.waterdepth, water_density=args.waterdensity)

    # 2) Use shared force generation, then sample onto a regular simulation grid.

    A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot = get_cummins_components(body=buoy, capytaine_dataset=capytaine_dataset, wave_direction=args.wavedirection, wave_amplitudes=wave_amplitudes, omegas=omegas, seed=args.seed)

    # create a time grid
    t_grid = np.arange(0.0, args.tspan + args.dt, args.dt)

    forcing_values, forcing_interp = build_excitation_time_series(F_ex_time, t_grid)

    # 3) Extract radiation damping and added mass for Pontryagin memory model.


    radiation_damping = capytaine_dataset["radiation_damping"].sel(
        radiating_dof="Heave", influenced_dof="Heave"
    ).values

    added_mass_inf = float(
        capytaine_dataset["added_mass"].sel(radiating_dof="Heave", influenced_dof="Heave").values[-1]
    )

    hydrostatic_stiffness = float(
        buoy.hydrostatic_stiffness.sel(influenced_dof="Heave", radiating_dof="Heave")
    )
    
    latch_gain = args.latchgain * (args.buoymass + A_heave_inf)

    params = (args.buoymass, A_heave_inf, args.cpto, latch_gain, hydrostatic_stiffness)

    # 4) Solve optimal latch policy and baseline no-latch case.
    history_opt, _, _ = solve_pontryagin_latching(
        omega=omegas,
        radiation_damping=radiation_damping,
        t_grid=t_grid,
        params=params,
        forcing=forcing_interp,
        max_iter=args.maxiter,
        n_terms=args.nprony,
    )
    
    print("solving no-latch baseline")
    history_no = solve_no_latch(
        omega=omegas,
        radiation_damping=radiation_damping,
        t_grid=t_grid,
        params=params,
        forcing=forcing_interp,
        n_terms=args.nprony,
    )

    # 5) Compare absorbed power.
    print("calculating absorbed power")
    p_inst_opt, p_mean_opt = calc_power(history_opt["v"], args.cpto)
    p_inst_no, p_mean_no = calc_power(history_no["v"], args.cpto)

    print(f"Average absorbed power without latching: {p_mean_no:.3f} W")
    print(f"Average absorbed power with Pontryagin latching: {p_mean_opt:.3f} W")
    print(f"Power improvement ratio: {(p_mean_opt / max(p_mean_no, 1e-12)):.3f}x")

    if args.plot:
        print("plotting results")
        plot_pontryagin_results(
            history_opt=history_opt,
            history_no_latch=history_no,
            forcing_values=forcing_values,
            p_inst_opt=p_inst_opt,
            p_inst_no=p_inst_no,
        )


