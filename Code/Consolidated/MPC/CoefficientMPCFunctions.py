import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__))) # 

from AutoRegression.AdaptiveARFunctions import ARPredictor, moving_average, N_HORIZON
# imports the AR predictor class, the smoothing helper, and the horizon constant
# from the AutoRegression subfolder

from Kalman.KalmanFunctions import SimpleKalmanFilter
# imports a kalman filter class from the Kalman subfolder 

def _clip(value, lower, upper): # clamping helper to keep c_pto values within bounds
    return max(lower, min(upper, value))


def _simulate_horizon(x0, v0, c_seq, M, K_heave, F_ex_time, t0, dt):

    N   = len(c_seq) # number of steps to simulate = length of damping sequence
    x_t = np.empty(N) # pre-allocate output array for predicted displacements
    v_t = np.empty(N) # pre-allocate output array for predicted velocities
    x, v = x0, v0 # initialize state with current conditions

    for k in range(N):
        t_k = t0 + k * dt # time at step k
        c_k = c_seq[k] # damping coefficient at step k

        def rhs(t, state, c=c_k):
            xs, vs = state
            return [vs, (F_ex_time(t) - c_k * vs - K_heave * xs) / M]
            # simplified cummins rhs for horizon simulation
            # dx/dt = v
            # dv/dt = (F_ex(t) - c_k * v - K_heave * x) / M
        sol   = solve_ivp(rhs, [t_k, t_k + dt], [x, v], method="RK45", max_step=dt)
        x     = sol.y[0, -1] # displacement at end of step k
        v     = sol.y[1, -1] # velocity at end of step k
        x_t[k] = x # store vals for output
        v_t[k] = v

    return x_t, v_t


def _mpc_optimise(x0, v0, M, K_heave, F_ex_time, t0, dt,
                cpto_min, cpto_max, x_max, N_horizon, c_init):
    
    def objective(c_seq):
        _, v_t = _simulate_horizon(x0, v0, c_seq, M, K_heave, F_ex_time, t0, dt)
        return -np.sum(c_seq * v_t**2) * dt   # total predicted power absorbed over horizon (negative for maximisation)

    bounds = [(cpto_min, cpto_max)] * N_horizon
    # produces a list of N_horizon identical (min, max) tuples

    if x_max < np.inf:
        def stroke_constraint(c_seq):
            x_t, _ = _simulate_horizon(x0, v0, c_seq, M, K_heave, F_ex_time, t0, dt)
            return x_max - np.max(np.abs(x_t))   # >= 0 when constraint satisfied
            # the worst-case predicted displacement must not exceed x_max

        constraints = [{"type": "ineq", "fun": stroke_constraint}]
        method      = "SLSQP" # Sequential Least Squares Programming supports nonlinear constraints
    else:
        constraints = []
        method      = "L-BFGS-B"

    result = minimize(
        objective, x0=c_init, method=method,
        bounds=bounds, constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-7} # maximum optimiser iterations before giving up
        # convergence tolerance on the objective value
    )
    return result.x


def solve_cummins_mpc_ar(
    body, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time,
    C_pto_base, K_pto, t_span,
    dt=0.05, cpto_min=1e4, cpto_max=5e5, x_max=np.inf,
    ar_order=10, ar_lambda=0.97,
    n_horizon=N_HORIZON, solve_every=5, seed=42,
):
    
    print("Running AR-MPC simulation")

    M  = body.mass + A_heave_inf # effective inertia is physical mass + hydrodynamic added mass
    ar = ARPredictor(order=ar_order, lam=ar_lambda) # create AR predictor

    history = {
        "t": [0.0], "x": [0.0], "v": [0.0],
        "x_est": [0.0], "c_pto": [C_pto_base],
    } # output storage with initial conditions

    t_now = 0.0; x_now = 0.0; v_now = 0.0 # simulation state
    c_pto = C_pto_base # current damping value applied to ODE
    c_seq = np.full(n_horizon, C_pto_base) 
    step  = 0 # step counter used to trigger optimisation

    while t_now < t_span[1]:
        t_next = t_now + dt

        def rhs(t, state, c=c_pto):
            x, v = state
            return [v, (F_ex_time(t) - c_pto * v - K_heave * x) / M]

        sol   = solve_ivp(rhs, [t_now, t_next], [x_now, v_now]) # advance buoy one dt step
        x_now = sol.y[0, -1]; v_now = sol.y[1, -1]; t_now = t_next
        step += 1

        ar.update(v_now)   # update AR coefficients with new true velocity

        if step % solve_every == 0:
            # only runs every solve_every steps to reduce computational load
            c_init          = np.roll(c_seq, -solve_every)
            # shift the previous optimal sequence left by solve_every positions
            c_init[-solve_every:] = C_pto_base
            # fill the newly exposed tail with the baseline value
            # gives the optimiser a reasonable starting point for the new future steps

            c_seq = _mpc_optimise(
                x_now, v_now, M, K_heave, F_ex_time, t_now, dt,
                cpto_min, cpto_max, x_max, n_horizon, c_init
            ) # find optimal c_seq from current true state

        # apply first element of optimal sequence
        c_pto = float(np.clip(c_seq[0], cpto_min, cpto_max))

        
        v_h   = ar.predict_ahead(n_horizon) # open-loop AR velocity forecast
        x_est = float(np.mean(x_now + np.cumsum(v_h) * dt)) # cumsum integrates the velocity forecast into a displacement trajectory

        history["t"].append(t_now); history["x"].append(x_now)
        history["v"].append(v_now); history["x_est"].append(x_est)
        history["c_pto"].append(c_pto) # store the value actually applied this step

    return history


def solve_cummins_mpc_kalman(
    body, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time,
    C_pto_base, K_pto, t_span,
    dt=0.05, measurement_std=0.05, cpto_min=1e4, cpto_max=5e5, x_max=np.inf,
    n_horizon=N_HORIZON, solve_every=5, seed=42,
):
    
    print("Running Kalman-MPC simulation")

    rng  = np.random.default_rng(seed) # random number generator for measurement noise
    M    = body.mass + A_heave_inf 
    kf   = SimpleKalmanFilter() # Kalman filter instead of AR predictor

    history = {
        "t": [0.0], "x": [0.0], "v": [0.0],
        "x_est": [0.0], "v_est": [0.0], "c_pto": [C_pto_base],
    }

    t_now = 0.0; x_now = 0.0; v_now = 0.0
    c_pto = C_pto_base
    c_seq = np.full(n_horizon, C_pto_base)
    step  = 0
    x_est = 0.0; v_est = 0.0 # Kalman state estimates, updated every step

    while t_now < t_span[1]:
        t_next = t_now + dt

        def rhs(t, state, c=c_pto):
            x, v = state
            return [v, (F_ex_time(t) - c_pto * v - K_heave * x) / M]

        sol   = solve_ivp(rhs, [t_now, t_next], [x_now, v_now])
        x_now = sol.y[0, -1]; v_now = sol.y[1, -1]; t_now = t_next
        step += 1

        # Kalman runs every step — provides clean state estimate for optimiser
        x_meas = x_now + rng.normal(0, measurement_std)
        A_kf   = np.array([[1, dt], [0, 1]])
        B_kf   = np.array([[0], [dt / M]])
        kf.predict(A_kf, B_kf, F_ex_time(t_now), np.eye(2) * 1e-4)
        kf.update(x_meas, np.array([[measurement_std**2]]))
        x_est  = kf.x[0, 0]
        v_est  = kf.x[1, 0]

        if step % solve_every == 0:
            c_init          = np.roll(c_seq, -solve_every)
            c_init[-solve_every:] = C_pto_base

            # Use Kalman estimate as initial condition — cleaner than noisy true state
            c_seq = _mpc_optimise(
                x_est, v_est, M, K_heave, F_ex_time, t_now, dt,
                cpto_min, cpto_max, x_max, n_horizon, c_init
            )

        c_pto = float(np.clip(c_seq[0], cpto_min, cpto_max))

        history["t"].append(t_now); history["x"].append(x_now)
        history["v"].append(v_now); history["x_est"].append(x_est)
        history["v_est"].append(v_est); history["c_pto"].append(c_pto)

    return history


def calculate_mpc_power(history, cutoff=50):
    v = np.array(history["v"][cutoff:])
    c = np.array(history["c_pto"][cutoff:])
    p = c * v**2
    return p, np.mean(p)


def plot_results_mpc(
    history_const, history_kalman_adapt, history_ar_adapt,
    history_kalman_mpc, history_ar_mpc,
    p_const, p_kalman_adapt, p_ar_adapt, p_kalman_mpc, p_ar_mpc,
    output_dir="results", show=True,
):
    os.makedirs(output_dir, exist_ok=True)
    cut = 50

    colours = {
        "const":        "tab:blue",
        "kalman_adapt": "tab:orange",
        "ar_adapt":     "tab:green",
        "kalman_mpc":   "tab:red",
        "ar_mpc":       "tab:purple",
    }

    # ── 1. Power — five-way ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(history_const["t"][cut:],        moving_average(p_const),        color=colours["const"],        lw=1.6, label="Constant")
    ax.plot(history_kalman_adapt["t"][cut:], moving_average(p_kalman_adapt), color=colours["kalman_adapt"], lw=1.6, label="Kalman adaptive")
    ax.plot(history_ar_adapt["t"][cut:],     moving_average(p_ar_adapt),     color=colours["ar_adapt"],     lw=1.6, label="AR adaptive")
    ax.plot(history_kalman_mpc["t"][cut:],   moving_average(p_kalman_mpc),   color=colours["kalman_mpc"],   lw=1.6, label="Kalman MPC")
    ax.plot(history_ar_mpc["t"][cut:],       moving_average(p_ar_mpc),       color=colours["ar_mpc"],       lw=1.6, label="AR MPC")
    ax.set_title("Instantaneous absorbed power — five-way comparison", fontsize=13)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Power (W)")
    ax.legend(ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "power_mpc_comparison.png"), dpi=300)

    # ── 2. c_pto — adaptive vs MPC side by side ───────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=False)
    fig.suptitle("PTO damping coefficient — adaptive vs MPC", fontsize=13)

    axes[0, 0].plot(history_kalman_adapt["t"][cut:], history_kalman_adapt["c_pto"][cut:], color=colours["kalman_adapt"], lw=1.4)
    axes[0, 0].set_title("Kalman adaptive"); axes[0, 0].set_ylabel("Damping (Ns/m)"); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(history_kalman_mpc["t"][cut:], history_kalman_mpc["c_pto"][cut:], color=colours["kalman_mpc"], lw=1.4)
    axes[0, 1].set_title("Kalman MPC"); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(history_ar_adapt["t"][cut:], history_ar_adapt["c_pto"][cut:], color=colours["ar_adapt"], lw=1.4)
    axes[1, 0].set_title("AR adaptive"); axes[1, 0].set_ylabel("Damping (Ns/m)"); axes[1, 0].set_xlabel("Time (s)"); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(history_ar_mpc["t"][cut:], history_ar_mpc["c_pto"][cut:], color=colours["ar_mpc"], lw=1.4)
    axes[1, 1].set_title("AR MPC"); axes[1, 1].set_xlabel("Time (s)"); axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "damping_mpc_comparison.png"), dpi=300)

    # ── 3. Displacement — true vs MPC predicted ───────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Displacement: true vs predicted (MPC planning horizon)", fontsize=13)

    t_km = history_kalman_mpc["t"][cut:]
    axes[0].plot(t_km, history_kalman_mpc["x"][cut:],     color="tab:blue",          lw=1.8, label="True")
    axes[0].plot(t_km, history_kalman_mpc["x_est"][cut:], color=colours["kalman_mpc"], lw=1.4, ls="--", label="Kalman MPC estimate")
    axes[0].set_title("Kalman MPC"); axes[0].set_ylabel("Displacement (m)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    t_am = history_ar_mpc["t"][cut:]
    axes[1].plot(t_am, history_ar_mpc["x"][cut:],     color="tab:blue",        lw=1.8, label="True")
    axes[1].plot(t_am, history_ar_mpc["x_est"][cut:], color=colours["ar_mpc"], lw=1.4, ls="--", label="AR MPC estimate")
    axes[1].set_title("AR MPC"); axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Displacement (m)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "displacement_mpc.png"), dpi=300)

    if show:
        plt.show()