import argparse
import numpy as np
import capytaine as cpt


cpt.set_logging("WARNING")

#importing all the group functions - this is the original model stuff
from Functions import (
    generate_frequencies,
    jonswap_frequency_amplitudes,
    generate_buoy,
    solve_with_capytaine,
    get_cummins_components,
    solve_cummins_stepwise_no_latch,
    calc_power_absorbed,
)

# importing Kalman + adaptive damping
from AdaptiveKalman import (
    solve_cummins_stepwise_adaptive_kalman,
    calculate_variable_damping_power,
    plot_results,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # simulation length + random seed
    parser.add_argument("--tspan", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)

    # wave setup (JONSWAP parameters)
    parser.add_argument("--nfreqcomponents", type=int, default=40)
    parser.add_argument("--peakperiod", type=float, default=12.0)
    parser.add_argument("--significantwaveheight", type=float, default=2.0)

    # buoy properties
    parser.add_argument("--buoymass", type=float, default=5000)
    parser.add_argument("--buoyradius", type=float, default=5)

    # water properties
    parser.add_argument("--waterdensity", type=float, default=1000)
    parser.add_argument("--waterdepth", type=float, default=np.inf)

    # wave direction (pi = head-on waves)
    parser.add_argument("--wavedirection", type=float, default=3.14159)

    # PTO parameters
    parser.add_argument("--cpto", type=float, default=1e5)  # damping
    parser.add_argument("--kpto", type=float, default=0.0)  # stiffness (not really used here)

    args = parser.parse_args()


    # CREATING BUOY MODEL
    # generates the floating body geometry + mass
    buoy = generate_buoy(radius=args.buoyradius, mass=args.buoymass)


    # GENERATING WAVES
    # creates frequency components
    omegas, delta_omega = generate_frequencies(args.nfreqcomponents, args.peakperiod)

    # converts JONSWAP spectrum into amplitudes for each frequency
    wave_amplitudes = jonswap_frequency_amplitudes(
        omegas,
        delta_omega,
        Hs=args.significantwaveheight,
        Tp=args.peakperiod,
    )


    # HYDRODYNAMICS (CAPYTAINE)
    # this is where we solve the wave-body interaction
    # gives us added mass, radiation damping etc
    dataset = solve_with_capytaine(
        body=buoy,
        omegas=omegas,
        wave_direction=args.wavedirection,
        water_depth=args.waterdepth,
        water_density=args.waterdensity,
    )


    # BUILD TIME-DOMAIN MODEL
    # converts frequency-domain data into Cummins equation form
    A_inf, t_kernel, kernel, K_heave, F_ex, F_ex_dot = get_cummins_components(
        body=buoy,
        capytaine_dataset=dataset,
        wave_direction=args.wavedirection,
        wave_amplitudes=wave_amplitudes,
        omegas=omegas,
        seed=args.seed,
    )

     # BASELINE CASE (CONSTANT DAMPING)
    # runs the simulation with fixed damping
    history_const = solve_cummins_stepwise_no_latch(
        body=buoy,
        A_heave_inf=A_inf,
        t_kernel=t_kernel,
        kernel=kernel,
        K_heave=K_heave,
        F_ex_time=F_ex,
        F_ex_time_dot=F_ex_dot,
        C_pto=args.cpto,
        K_pto=args.kpto,
        t_span=[0, args.tspan],
        dt=0.05,
    )

    # calculates power for constant damping
    p_const, p_mean_const = calc_power_absorbed(history_const, args.cpto)


    # KALMAN + ADAPTIVE DAMPING
    history_adapt = solve_cummins_stepwise_adaptive_kalman(
        body=buoy,
        A_heave_inf=A_inf,
        t_kernel=t_kernel,
        kernel=kernel,
        K_heave=K_heave,
        F_ex_time=F_ex,
        C_pto_base=args.cpto,
        K_pto=args.kpto,
        t_span=[0, args.tspan],
        dt=0.05,
        seed=args.seed,
    )

    # calculates power for adaptive case
    p_adapt, p_mean_adapt = calculate_variable_damping_power(history_adapt)

    # Results

    print("\nRESULTS:")
    print(f"Constant power: {p_mean_const:.2f} W")
    print(f"Adaptive power: {p_mean_adapt:.2f} W")
    print(f"Improvement: {100*(p_mean_adapt - p_mean_const)/p_mean_const:.2f}%")

    # Plots
    # shows:
    # - power comparison
    # - Kalman estimation
    # - adaptive damping behaviour
    plot_results(history_const, history_adapt, p_const, p_adapt)