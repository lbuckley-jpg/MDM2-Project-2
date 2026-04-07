"""
Adaptive PTO damping driven by a simple linear Kalman state estimate.

The simulation advances a heave oscillator with linear damping, observes noisy
position, estimates [position, velocity] with a Kalman filter, and sets the
next step's PTO damping from the estimated speed. Plotting helpers smooth power
traces and save comparison figures.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# just makes sure values stay within limits (used for damping so it doesn't blow up)
def _clip(value, lower, upper):
    # Clamp a scalar to bounds so adaptive gains cannot push damping out of range.
    return max(lower, min(upper, value))


# simple smoothing for plots so they look nicer and less noisy
def moving_average(y, window=15):
    y = np.asarray(y, dtype=float)
    if window <= 1 or len(y) < window:
        return y.copy()
    kernel = np.ones(window) / window

    y_pad = np.pad(y, (window // 2, window - 1 - window // 2), mode="edge")

    return np.convolve(y_pad, kernel, mode="valid")



# SIMPLE KALMAN FILTER
# this estimates displacement + velocity from noisy measurements
class SimpleKalmanFilter:
    # Discrete-time linear Kalman filter for state x = [position, velocity]^T
    # with position-only measurements z = x_position + noise.

    def __init__(self):
        # state = [displacement, velocity]
        self.x = np.zeros((2, 1))

        # uncertainty matrix
        self.P = np.eye(2)

        # measurement matrix (only measuring displacement)
        self.H = np.array([[1.0, 0.0]])

    def predict(self, A, B, u, Q):
        # prediction step using system model
        self.x = A @ self.x + B * float(u)
        self.P = A @ self.P @ A.T + Q

    def update(self, z, R):
        # correction step using measurement
        z = np.array([[float(z)]])
        y = z - self.H @ self.x  # error between measurement and prediction
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P


# calculates absorbed power when damping is changing over time
def calculate_variable_damping_power(history, cutoff=50):
    # ignore first part (transient behaviour)
    v = np.array(history["v"][cutoff:])
    c = np.array(history["c_pto"][cutoff:])

    # power formula: P = c * v^2
    p = c * v**2
    return p, np.mean(p)



# solves motion AND applies Kalman + adaptive damping
def solve_cummins_stepwise_adaptive_kalman(
    body,
    A_heave_inf,
    t_kernel,
    kernel,
    K_heave,
    F_ex_time,
    C_pto_base,
    K_pto,
    t_span,
    dt=0.05,
    measurement_std=0.05,
    adaptive_gain=5e4,
    cpto_min=1e4,
    cpto_max=5e5,
    seed=42,
):

    print("Running adaptive Kalman simulation")

    rng = np.random.default_rng(seed)

    # effective mass (includes hydrodynamic added mass)
    M = body.mass + A_heave_inf

    # storing results over time
    history = {
        "t": [0],
        "x": [0],        # true displacement
        "v": [0],        # true velocity
        "x_est": [0],    # estimated displacement (Kalman)
        "v_est": [0],    # estimated velocity (Kalman)
        "c_pto": [C_pto_base],  # adaptive damping
    }

    kf = SimpleKalmanFilter()

    # initial conditions
    t_now = 0
    x_now = 0
    v_now = 0
    c_pto = C_pto_base

    # time stepping loop
    while t_now < t_span[1]:
        t_next = t_now + dt

        # simplified Cummins equation
        def rhs(t, state):
            x, v = state
            dvdt = (F_ex_time(t) - c_pto * v - K_heave * x) / M
            return [v, dvdt]

        # solve physics for this time step
        sol = solve_ivp(rhs, [t_now, t_next], [x_now, v_now])

        # update states
        x_now = sol.y[0, -1]
        v_now = sol.y[1, -1]
        t_now = t_next

        # simulate sensor noise (real systems are noisy)
        x_meas = x_now + rng.normal(0, measurement_std)

        # simple discrete model for Kalman
        A = np.array([[1, dt], [0, 1]])
        B = np.array([[0], [dt / M]])

        # Kalman filter steps
        kf.predict(A, B, F_ex_time(t_now), np.eye(2) * 1e-4)
        kf.update(x_meas, np.array([[measurement_std**2]]))

        # estimated states
        x_est = kf.x[0, 0]
        v_est = kf.x[1, 0]


        # Adaptive damping
        # higher velocity = more damping = more energy absorbed
        c_pto = _clip(C_pto_base + adaptive_gain * abs(v_est), cpto_min, cpto_max)

        # results storage
        history["t"].append(t_now)
        history["x"].append(x_now)
        history["v"].append(v_now)
        history["x_est"].append(x_est)
        history["v_est"].append(v_est)
        history["c_pto"].append(c_pto)

    return history

def plot_results(history_const, history_adapt, p_const, p_adapt, output_dir="results", show=True):
    os.makedirs(output_dir, exist_ok=True)

    # Power comparison
    plt.figure(figsize=(10, 5))
    plt.plot(history_const["t"][50:], moving_average(p_const), label="Constant damping", linewidth=2)
    plt.plot(history_adapt["t"][50:], moving_average(p_adapt), label="Adaptive damping", linewidth=2)
    plt.title("Instantaneous Absorbed Power Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "power.png"), dpi=300)

    # kalman performance
    plt.figure(figsize=(10, 5))
    plt.plot(history_adapt["t"], history_adapt["x"], label="True displacement", linewidth=2)
    plt.plot(history_adapt["t"], history_adapt["x_est"], "--", label="Estimated displacement", linewidth=2)
    plt.title("Kalman Filter Estimation")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "kalman.png"), dpi=300)

    # damping behaviour
    plt.figure(figsize=(10, 5))
    plt.plot(history_adapt["t"], history_adapt["c_pto"], linewidth=2)
    plt.title("Adaptive Damping Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Damping (Ns/m)")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "damping.png"), dpi=300)

    if show:
        plt.show()