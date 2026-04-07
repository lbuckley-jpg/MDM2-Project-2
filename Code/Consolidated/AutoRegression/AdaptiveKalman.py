import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def _clip(value, lower, upper):
    return max(lower, min(upper, value))


def moving_average(y, window=15):
    y = np.asarray(y, dtype=float)
    if window <= 1 or len(y) < window:
        return y.copy()
    kernel = np.ones(window) / window
    y_pad = np.pad(y, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


class SimpleKalmanFilter:
    def __init__(self):
        self.x = np.zeros((2, 1))
        self.P = np.eye(2)
        self.H = np.array([[1.0, 0.0]])

    def predict(self, A, B, u, Q):
        self.x = A @ self.x + B * float(u)
        self.P = A @ self.P @ A.T + Q

    def update(self, z, R):
        z = np.array([[float(z)]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P


def calculate_variable_damping_power(history, cutoff=50):
    v = np.array(history["v"][cutoff:])
    c = np.array(history["c_pto"][cutoff:])
    p = c * v**2
    return p, np.mean(p)


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
    M = body.mass + A_heave_inf

    history = {
        "t": [0],
        "x": [0],
        "v": [0],
        "x_est": [0],
        "v_est": [0],
        "c_pto": [C_pto_base],
    }

    kf = SimpleKalmanFilter()

    t_now = 0
    x_now = 0
    v_now = 0
    c_pto = C_pto_base

    while t_now < t_span[1]:
        t_next = t_now + dt

        def rhs(t, state):
            x, v = state
            dvdt = (F_ex_time(t) - c_pto * v - K_heave * x) / M
            return [v, dvdt]

        sol = solve_ivp(rhs, [t_now, t_next], [x_now, v_now])

        x_now = sol.y[0, -1]
        v_now = sol.y[1, -1]
        t_now = t_next

        x_meas = x_now + rng.normal(0, measurement_std)

        A = np.array([[1, dt], [0, 1]])
        B = np.array([[0], [dt / M]])

        kf.predict(A, B, F_ex_time(t_now), np.eye(2) * 1e-4)
        kf.update(x_meas, np.array([[measurement_std**2]]))

        x_est = kf.x[0, 0]
        v_est = kf.x[1, 0]

        c_pto = _clip(C_pto_base + adaptive_gain * abs(v_est), cpto_min, cpto_max)

        history["t"].append(t_now)
        history["x"].append(x_now)
        history["v"].append(v_now)
        history["x_est"].append(x_est)
        history["v_est"].append(v_est)
        history["c_pto"].append(c_pto)

    return history


def plot_results(history_const, history_adapt, p_const, p_adapt, output_dir="results", show=True):
    import os
    os.makedirs(output_dir, exist_ok=True)

    # POWER
    plt.figure(figsize=(10, 5))
    plt.plot(history_const["t"][50:], moving_average(p_const), label="Constant damping", linewidth=2)
    plt.plot(history_adapt["t"][50:], moving_average(p_adapt), label="Adaptive damping", linewidth=2)
    plt.title("Instantaneous Absorbed Power Comparison", fontsize=14)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Power (W)", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "power.png"), dpi=300)

    # KALMAN
    plt.figure(figsize=(10, 5))
    plt.plot(history_adapt["t"], history_adapt["x"], label="True displacement", linewidth=2)
    plt.plot(history_adapt["t"], history_adapt["x_est"], "--", label="Estimated displacement", linewidth=2)
    plt.title("Kalman Filter Displacement Estimation", fontsize=14)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Displacement (m)", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kalman.png"), dpi=300)

    # DAMPING
    plt.figure(figsize=(10, 5))
    plt.plot(history_adapt["t"], history_adapt["c_pto"], linewidth=2)
    plt.title("Adaptive PTO Damping Over Time", fontsize=14)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Damping Coefficient (Ns/m)", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "damping.png"), dpi=300)

    if show:
        plt.show()

