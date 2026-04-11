
#ai has been used in the development of this file

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp


def _clip(value, lower, upper): # helper function for constraining values within bounds
    return max(lower, min(upper, value))


def moving_average(y, window=15): 
    y = np.asarray(y, dtype=float) # ensure y is a float array regardless of input type
    if window <= 1 or len(y) < window:
        return y.copy() # if window too small or data too short return
    kernel = np.ones(window) / window # averaging kernel, each element is 1/window
    y_pad  = np.pad(y, (window // 2, window - 1 - window // 2), mode="edge") # pad with edge values to maintain length
    return np.convolve(y_pad, kernel, mode="valid") # convolve to compute rolling mean


class ARPredictor:
    

    def __init__(self, order: int = 10, lam: float = 0.97,
                 delta: float = 100.0, reg_lambda: float = 1e-2):
        self.p          = order # AR order p or how many past velocity values to use
        self.lam        = lam # forgetting factor or how quickly old observations fade
        self.reg_lambda = reg_lambda # ridge regularisation which prevents gain blowing up during calm patches
        self.phi        = np.zeros(order) # AR coefficients which are initialised to zero (no prior knowledge)
        self.P          = np.eye(order) * delta # Recursive Least Squares covariance matrix with a large initial value = diffuse prior (learns fast early on)
        self.buf        = np.zeros(order) # circular buffer storing the p most recent true velocities

    def update(self, v_new: float) -> float:
        
        x     = self.buf.copy() # regressor vector of past velocities
        Px    = self.P @ x # intermediate product for gain calculation
        denom = self.lam + x @ Px + self.reg_lambda   # regularised denominator
        gain  = Px / denom # how much to trust the new observation
        self.phi += gain * (v_new - x @ self.phi) # update AR coefficients based on prediction error
        self.P    = (self.P - np.outer(gain, x) @ self.P) / self.lam # update covariance matrix with forgetting factor
        self.P   += np.eye(self.p) * 1e-8 # small diagonal floor to prevent P collapsing to zero
        self.buf    = np.roll(self.buf, 1) # shift buffer right so the oldest v is dropped
        self.buf[0] = v_new # insert new v at front
        return float(self.phi @ self.buf) # one step prediction based on updated buffer


    def predict_ahead(self, steps: int) -> np.ndarray:
        
        buf     = self.buf.copy() # copy buffer so true state is never modified
        v_rms   = float(np.sqrt(np.mean(buf ** 2))) # root mean squared to show current sea state amplitude
        v_scale = max(4.0 * v_rms, 0.005)   # floor prevents zero-clamp at start, 4x rms
        preds   = np.zeros(steps) # predict next v using current buffer
        for i in range(steps):
            v_i      = float(self.phi @ buf)
            v_i      = np.clip(v_i, -v_scale, v_scale) # clamp incase phi is unstable
            preds[i] = v_i
            buf       = np.roll(buf, 1) # shift buffer right to make room for new prediction
            buf[0]    = v_i # insert prediction at front so next prediction is based on it
        return preds


# Horizon. At dt=0.05 s, 40 steps = 2.0 s ahead.
N_HORIZON = 40

<<<<<<< HEAD
def calculate_variable_damping_power(history, cutoff=50):
    v = np.array(history["v"][cutoff:])
=======

def calculate_variable_damping_power(history, cutoff=50): # skip first 50 steps from rest
    v = np.array(history["v"][cutoff:]) # return time series and mean of absorbed power based on velocity and time-varying damping coefficient
>>>>>>> bd4d594 (Fixed AR Bugs and Commented through Functions code)
    c = np.array(history["c_pto"][cutoff:])
    return c * v ** 2, np.mean(c * v ** 2)


def solve_cummins_stepwise_adaptive_ar( 
    body, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time,
    C_pto_base, K_pto, t_span,
    dt=0.05, adaptive_gain=5e4, cpto_min=1e4, cpto_max=5e5,
    ar_order=10, ar_lambda=0.97, seed=42,
):
    print("Running adaptive AR simulation")

    M  = body.mass + A_heave_inf # effective inertia = buoy mass + added mass
    ar = ARPredictor(order=ar_order, lam=ar_lambda) # create AR predictor with chosen order and forgetting factor

    history = {
        "t":      [0.0], "x":     [0.0], "v":    [0.0],
        "v_pred": [0.0], "x_est": [0.0], "c_pto":[C_pto_base],
    } # seed initial conditions

    t_now = 0.0; x_now = 0.0; v_now = 0.0; c_pto = C_pto_base # simulation state variables

    while t_now < t_span[1]: # main loop for cummins simulation with AR adaptive damping
        t_next = t_now + dt # end time for this step

        def rhs(t, state):
            x, v = state
            return [v, (F_ex_time(t) - c_pto * v - K_heave * x) / M]
            # simplified rhs of cummins equation
            #dx/dt = v
            #dv/dt = (F_ex(t) - c_pto * v - K_heave * x) / M
            # where F_ex is the wave excitation force at time t
            #  c_pto is the current damping coefficient
            # K_heave is the hydrostatic stiffness and M is the effective mass.

        sol   = solve_ivp(rhs, [t_now, t_next], [x_now, v_now]) # integrate 1 dt
        x_now = sol.y[0, -1] # displacement at end of step
        v_now = sol.y[1, -1] # velocity at end of step
        t_now = t_next # update time to end of step

        v_pred_1 = ar.update(v_now) # update AR with true velocity, one-step prediction drives damping law
        c_pto    = _clip(C_pto_base + adaptive_gain * abs(v_pred_1), cpto_min, cpto_max) # adaptive law based on predicted velocity, clipped to prevent instability or negative damping

        v_horizon = ar.predict_ahead(N_HORIZON) # multi-step open-loop displacement estimate
        x_traj    = x_now + np.cumsum(v_horizon) * dt # integrate forecast velocities to get predicted displacement trajectory
        x_est     = float(np.mean(x_traj)) # single representative displacement estimate: mean over horizon

        history["t"].append(t_now) # store time, true state, one-step AR prediction, AR predicted displacement, and damping coefficient for analysis
        history["x"].append(x_now)
        history["v"].append(v_now)
        history["v_pred"].append(v_pred_1)
        history["x_est"].append(x_est)
        history["c_pto"].append(c_pto)

    return history


def plot_results_ar(history_const, history_kalman, history_ar,
                    p_const, p_kalman, p_ar, output_dir="results", show=True):
    os.makedirs(output_dir, exist_ok=True)
    cut = 50 # skips forward 50 steps to cut transients from rest

    # ── 1. Power ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(history_const["t"][cut:],  moving_average(p_const),  label="Constant damping", lw=1.8) # smooths noisy instantaneous power for readibility
    ax.plot(history_kalman["t"][cut:], moving_average(p_kalman), label="Kalman adaptive",  lw=1.8)
    ax.plot(history_ar["t"][cut:],     moving_average(p_ar),     label="AR adaptive",      lw=1.8)
    ax.set_title("Instantaneous absorbed power — three-way comparison", fontsize=13)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Power (W)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "power_comparison.png"), dpi=300)

    # ── 2. Displacement estimation ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(
        f"Displacement estimation — Kalman vs AR\n"
        f"(AR x_est = mean of {N_HORIZON}-step open-loop trajectory, "
        f"{N_HORIZON * 0.05:.1f} s horizon)",
        fontsize=12
    )
    t_k = history_kalman["t"][cut:]
    axes[0].plot(t_k, history_kalman["x"][cut:],
                 label="True displacement", lw=1.8, color="tab:blue")
    axes[0].plot(t_k, history_kalman["x_est"][cut:],
                 label="Kalman estimated displacement", lw=1.4, ls="--", color="tab:orange")
    axes[0].set_title("Kalman filter", fontsize=11)
    axes[0].set_ylabel("Displacement (m)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    t_ar = history_ar["t"][cut:]
    axes[1].plot(t_ar, history_ar["x"][cut:],
                 label="True displacement", lw=1.8, color="tab:blue")
    axes[1].plot(t_ar, history_ar["x_est"][cut:],
                 label=f"AR mean predicted displacement ({N_HORIZON * 0.05:.1f} s horizon)",
                 lw=1.4, ls="--", color="tab:green")
    axes[1].set_title("AR predictor", fontsize=11)
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Displacement (m)")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "displacement_estimation.png"), dpi=300)

    # ── 3. Damping coefficients ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    axes[0].plot(history_kalman["t"][cut:], history_kalman["c_pto"][cut:],
                 lw=1.6, color="tab:orange")
    axes[0].set_title("Kalman adaptive c_pto", fontsize=11)
    axes[0].set_ylabel("Damping (Ns/m)"); axes[0].grid(alpha=0.3)

    axes[1].plot(history_ar["t"][cut:], history_ar["c_pto"][cut:],
                 lw=1.6, color="tab:green")
    axes[1].set_title("AR adaptive c_pto", fontsize=11)
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Damping (Ns/m)")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Adaptive PTO damping coefficient over time", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "damping_comparison.png"), dpi=300)

    if show:
        plt.show()


