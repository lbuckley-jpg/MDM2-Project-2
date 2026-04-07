"""
RunARComparison.py
==================
Runs three simulations in sequence and produces a three-way comparison:

    1. Constant (linear) damping           — solve_cummins_stepwise_no_latch
    2. Kalman adaptive damping             — solve_cummins_stepwise_adaptive_kalman
    3. AR adaptive damping                 — solve_cummins_stepwise_adaptive_ar

Usage (mirrors RunKalmanAdaptive.py exactly, plus two new AR flags):

    python RunARComparison.py \
        --tspan 300 --seed 42 \
        --nfreqcomponents 40 \
        --peakperiod 12.0 \
        --significantwaveheight 2.0 \
        --buoymass 5000 --buoyradius 5 \
        --waterdensity 1000 --waterdepth inf \
        --wavedirection 3.14159 \
        --cpto 1e5 --kpto 0.0 \
        --arorder 10 --arlambda 0.97

New flags
---------
--arorder    int    AR model order (default 10)
--arlambda   float  RLS forgetting factor (default 0.97; range 0.9–1.0)
"""

import argparse
import numpy as np
import capytaine as cpt

cpt.set_logging("WARNING")

from Functions import (
    generate_frequencies,
    jonswap_frequency_amplitudes,
    generate_buoy,
    solve_with_capytaine,
    get_cummins_components,
    solve_cummins_stepwise_no_latch,
    calc_power_absorbed,
)

from AdaptiveKalman import (
    solve_cummins_stepwise_adaptive_kalman,
    calculate_variable_damping_power as kalman_power,
)

from AdaptiveAR import (
    solve_cummins_stepwise_adaptive_ar,
    calculate_variable_damping_power as ar_power,
    plot_results_ar,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Three-way WEC damping comparison")

    # timing / reproducibility
    parser.add_argument("--tspan",   type=int,   required=True)
    parser.add_argument("--seed",    type=int,   required=True)

    # wave / spectrum
    parser.add_argument("--nfreqcomponents",      type=int,   default=40)
    parser.add_argument("--peakperiod",            type=float, default=12.0)
    parser.add_argument("--significantwaveheight", type=float, default=2.0)

    # buoy geometry
    parser.add_argument("--buoymass",   type=float, default=5000)
    parser.add_argument("--buoyradius", type=float, default=5)

    # water properties
    parser.add_argument("--waterdensity", type=float, default=1000)
    parser.add_argument("--waterdepth",   type=float, default=np.inf)

    # wave direction
    parser.add_argument("--wavedirection", type=float, default=3.14159)

    # PTO
    parser.add_argument("--cpto", type=float, default=1e5)
    parser.add_argument("--kpto", type=float, default=0.0)

    # adaptive (shared)
    parser.add_argument("--adaptivegain", type=float, default=5e4,
                        help="Scales |v_predicted| to delta_c_pto (same for both methods)")
    parser.add_argument("--cptomin", type=float, default=1e4)
    parser.add_argument("--cptomax", type=float, default=5e5)

    # AR-specific
    parser.add_argument("--arorder",  type=int,   default=10,
                        help="AR model order p (number of lagged velocity values)")
    parser.add_argument("--arlambda", type=float, default=0.97,
                        help="RLS forgetting factor (0.9–1.0; lower = faster adaptation)")

    args = parser.parse_args()

    # ── build buoy ─────────────────────────────────────────────────────────────
    buoy = generate_buoy(radius=args.buoyradius, mass=args.buoymass)

    # ── generate JONSWAP wave components ───────────────────────────────────────
    omegas, delta_omega = generate_frequencies(args.nfreqcomponents, args.peakperiod)
    wave_amplitudes = jonswap_frequency_amplitudes(
        omegas, delta_omega,
        Hs=args.significantwaveheight,
        Tp=args.peakperiod,
    )

    # ── BEM hydrodynamics via Capytaine ────────────────────────────────────────
    dataset = solve_with_capytaine(
        body           = buoy,
        omegas         = omegas,
        wave_direction = args.wavedirection,
        water_depth    = args.waterdepth,
        water_density  = args.waterdensity,
    )

    A_inf, t_kernel, kernel, K_heave, F_ex, F_ex_dot = get_cummins_components(
        body              = buoy,
        capytaine_dataset = dataset,
        wave_direction    = args.wavedirection,
        wave_amplitudes   = wave_amplitudes,
        omegas            = omegas,
        seed              = args.seed,
    )

    # ── 1. Constant damping ────────────────────────────────────────────────────
    print("\n── Constant damping ──")
    history_const = solve_cummins_stepwise_no_latch(
        body         = buoy,
        A_heave_inf  = A_inf,
        t_kernel     = t_kernel,
        kernel       = kernel,
        K_heave      = K_heave,
        F_ex_time    = F_ex,
        F_ex_time_dot= F_ex_dot,
        C_pto        = args.cpto,
        K_pto        = args.kpto,
        t_span       = [0, args.tspan],
        dt           = 0.05,
    )
    p_const, p_mean_const = calc_power_absorbed(history_const, args.cpto)

    # ── 2. Kalman adaptive ─────────────────────────────────────────────────────
    print("\n── Kalman adaptive ──")
    history_kalman = solve_cummins_stepwise_adaptive_kalman(
        body          = buoy,
        A_heave_inf   = A_inf,
        t_kernel      = t_kernel,
        kernel        = kernel,
        K_heave       = K_heave,
        F_ex_time     = F_ex,
        C_pto_base    = args.cpto,
        K_pto         = args.kpto,
        t_span        = [0, args.tspan],
        dt            = 0.05,
        adaptive_gain = args.adaptivegain,
        cpto_min      = args.cptomin,
        cpto_max      = args.cptomax,
        seed          = args.seed,
    )
    p_kalman, p_mean_kalman = kalman_power(history_kalman)

    # ── 3. AR adaptive ─────────────────────────────────────────────────────────
    print("\n── AR adaptive ──")
    history_ar = solve_cummins_stepwise_adaptive_ar(
        body          = buoy,
        A_heave_inf   = A_inf,
        t_kernel      = t_kernel,
        kernel        = kernel,
        K_heave       = K_heave,
        F_ex_time     = F_ex,
        C_pto_base    = args.cpto,
        K_pto         = args.kpto,
        t_span        = [0, args.tspan],
        dt            = 0.05,
        adaptive_gain = args.adaptivegain,
        cpto_min      = args.cptomin,
        cpto_max      = args.cptomax,
        ar_order      = args.arorder,
        ar_lambda     = args.arlambda,
        seed          = args.seed,
    )
    p_ar, p_mean_ar = ar_power(history_ar)

    # ── results summary ────────────────────────────────────────────────────────
    print("\n" + "═" * 45)
    print("RESULTS")
    print("═" * 45)
    print(f"  Constant damping : {p_mean_const:.2f} W")
    print(f"  Kalman adaptive  : {p_mean_kalman:.2f} W   "
          f"({100*(p_mean_kalman - p_mean_const)/p_mean_const:+.1f}% vs constant)")
    print(f"  AR adaptive      : {p_mean_ar:.2f} W   "
          f"({100*(p_mean_ar - p_mean_const)/p_mean_const:+.1f}% vs constant)")
    print(f"  AR vs Kalman     : {100*(p_mean_ar - p_mean_kalman)/p_mean_kalman:+.2f}%")
    print("═" * 45)

    # ── plots ──────────────────────────────────────────────────────────────────
    plot_results_ar(history_const, history_kalman, history_ar,
                    p_const, p_kalman, p_ar)
