import argparse
import numpy as np
import capytaine as cpt
import sys
import os

cpt.set_logging("WARNING")
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from SharedCapytaineFunctions import (
    generate_frequencies, jonswap_frequency_amplitudes, generate_buoy,
    solve_with_capytaine, get_cummins_components,
    solve_cummins_stepwise_no_control, calc_power_absorbed,
)
from Kalman.KalmanFunctions import (
    solve_cummins_stepwise_adaptive_kalman,
    calculate_variable_damping_power as kalman_adapt_power,
)
from AutoRegression.AdaptiveARFunctions import (
    solve_cummins_stepwise_adaptive_ar,
    calculate_variable_damping_power as ar_adapt_power,
)
from MPC.MPCFunctions import (
    solve_cummins_mpc_ar,
    solve_cummins_mpc_kalman,
    calculate_mpc_power,
    plot_results_mpc,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Five-way WEC damping comparison including MPC")

    parser.add_argument("--tspan",                 type=int,   required=True)
    parser.add_argument("--seed",                  type=int,   required=True)
    parser.add_argument("--nfreqcomponents",       type=int,   default=40)
    parser.add_argument("--peakperiod",            type=float, default=12.0)
    parser.add_argument("--significantwaveheight", type=float, default=2.0)
    parser.add_argument("--buoymass",              type=float, default=5000)
    parser.add_argument("--buoyradius",            type=float, default=5)
    parser.add_argument("--waterdensity",          type=float, default=1000)
    parser.add_argument("--waterdepth",            type=float, default=np.inf)
    parser.add_argument("--wavedirection",         type=float, default=3.14159)
    parser.add_argument("--cpto",                  type=float, default=1e5)
    parser.add_argument("--kpto",                  type=float, default=0.0)
    # adaptive shared
    parser.add_argument("--adaptivegain",          type=float, default=5e4)
    parser.add_argument("--cptomin",               type=float, default=1e4)
    parser.add_argument("--cptomax",               type=float, default=5e5)
    # AR
    parser.add_argument("--arorder",               type=int,   default=10)
    parser.add_argument("--arlambda",              type=float, default=0.97)
    # MPC
    parser.add_argument("--nhorizon",              type=int,   default=40,
                        help="MPC horizon length in steps (default 40 = 2 s at dt=0.05)")
    parser.add_argument("--solveevery",            type=int,   default=20,
                        help="Re-solve optimisation every N steps (default 5 = 0.25 s)")
    parser.add_argument("--xmax",                  type=float, default=float("inf"),
                        help="Stroke limit [m]. Omit for unconstrained.")

    args = parser.parse_args()

    # --- shared setup (run once, used by all five solvers) --------------------
    buoy            = generate_buoy(radius=args.buoyradius, mass=args.buoymass)
    omegas, dw      = generate_frequencies(args.nfreqcomponents, args.peakperiod)
    wave_amplitudes = jonswap_frequency_amplitudes(omegas, dw,
                        Hs=args.significantwaveheight, Tp=args.peakperiod)
    dataset         = solve_with_capytaine(buoy, omegas, args.wavedirection,
                        args.waterdepth, args.waterdensity)
    A_inf, t_kernel, kernel, K_heave, F_ex, F_ex_dot = get_cummins_components(
        buoy, dataset, args.wavedirection, wave_amplitudes, omegas, args.seed
    )

    common = dict(
        body=buoy, A_heave_inf=A_inf, t_kernel=t_kernel, kernel=kernel,
        K_heave=K_heave, F_ex_time=F_ex, K_pto=args.kpto,
        t_span=[0, args.tspan], dt=0.05,
    )

    # --- 1. constant damping baseline ----------------------------------------
    print("\n── Constant ──")
    h_const              = solve_cummins_stepwise_no_control(
        **common, F_ex_time_dot=F_ex_dot, C_pto=args.cpto
    )
    p_const, pm_const    = calc_power_absorbed(h_const)

    # --- 2. Kalman adaptive ---------------------------------------------------
    print("\n── Kalman adaptive ──")
    h_kal_ad             = solve_cummins_stepwise_adaptive_kalman(
        **common, C_pto_base=args.cpto, adaptive_gain=args.adaptivegain,
        cpto_min=args.cptomin, cpto_max=args.cptomax, seed=args.seed
    )
    p_kal_ad, pm_kal_ad  = kalman_adapt_power(h_kal_ad)

    # --- 3. AR adaptive -------------------------------------------------------
    print("\n── AR adaptive ──")
    h_ar_ad              = solve_cummins_stepwise_adaptive_ar(
        **common, C_pto_base=args.cpto, adaptive_gain=args.adaptivegain,
        cpto_min=args.cptomin, cpto_max=args.cptomax,
        ar_order=args.arorder, ar_lambda=args.arlambda, seed=args.seed
    )
    p_ar_ad, pm_ar_ad    = ar_adapt_power(h_ar_ad)

    # --- 4. Kalman MPC --------------------------------------------------------
    print("\n── Kalman MPC ──")
    h_kal_mpc            = solve_cummins_mpc_kalman(
        **common, C_pto_base=args.cpto, cpto_min=args.cptomin, cpto_max=args.cptomax,
        x_max=args.xmax, n_horizon=args.nhorizon, solve_every=args.solveevery,
        seed=args.seed
    )
    p_kal_mpc, pm_kal_mpc = calculate_mpc_power(h_kal_mpc)

    # --- 5. AR MPC ------------------------------------------------------------
    print("\n── AR MPC ──")
    h_ar_mpc             = solve_cummins_mpc_ar(
        **common, C_pto_base=args.cpto, cpto_min=args.cptomin, cpto_max=args.cptomax,
        x_max=args.xmax, n_horizon=args.nhorizon, solve_every=args.solveevery,
        ar_order=args.arorder, ar_lambda=args.arlambda, seed=args.seed
    )
    p_ar_mpc, pm_ar_mpc  = calculate_mpc_power(h_ar_mpc)

    # --- results table --------------------------------------------------------
    print("\n" + "=" * 55)
    print(f"  Constant        : {pm_const:.2f} W  (baseline)")
    print(f"  Kalman adaptive : {pm_kal_ad:.2f} W  ({100*(pm_kal_ad -pm_const)/pm_const:+.1f}%)")
    print(f"  AR adaptive     : {pm_ar_ad:.2f} W  ({100*(pm_ar_ad  -pm_const)/pm_const:+.1f}%)")
    print(f"  Kalman MPC      : {pm_kal_mpc:.2f} W  ({100*(pm_kal_mpc-pm_const)/pm_const:+.1f}%)")
    print(f"  AR MPC          : {pm_ar_mpc:.2f} W  ({100*(pm_ar_mpc -pm_const)/pm_const:+.1f}%)")
    print(f"  Kalman MPC vs adaptive : {100*(pm_kal_mpc-pm_kal_ad)/pm_kal_ad:+.2f}%")
    print(f"  AR MPC     vs adaptive : {100*(pm_ar_mpc -pm_ar_ad )/pm_ar_ad :+.2f}%")
    print("=" * 55)

    plot_results_mpc(
        h_const, h_kal_ad, h_ar_ad, h_kal_mpc, h_ar_mpc,
        p_const, p_kal_ad, p_ar_ad, p_kal_mpc, p_ar_mpc,
    )