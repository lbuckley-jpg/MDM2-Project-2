# """
# AdaptiveAR.py
# =============
# Adaptive PTO damping using an AutoRegressive (AR) velocity predictor.

# The AR model predicts the buoy velocity one step ahead from a rolling
# window of recent velocity measurements.  The predicted velocity replaces
# the Kalman-estimated velocity used in AdaptiveKalman.py, so the adaptive
# damping law is otherwise identical:

#     c_pto = clip(C_pto_base + adaptive_gain * |v_predicted|, cpto_min, cpto_max)

# AR model recap
# --------------
# An AR(p) model expresses the current value as a linear combination of the
# previous p values plus white noise:

#     v[n] = phi_1 * v[n-1] + phi_2 * v[n-2] + ... + phi_p * v[n-p] + e[n]

# Coefficients phi are estimated online using Recursive Least Squares (RLS),
# which updates the coefficient vector every time step without storing or
# re-fitting the full history.  RLS update equations:

#     gain  = P @ x / (lambda + x^T @ P @ x)
#     phi  += gain * (v_new - x^T @ phi)
#     P     = (P - gain @ x^T @ P) / lambda

# where x = [v[n-1], ..., v[n-p]] is the regressor vector, lambda is a
# forgetting factor (0 < lambda <= 1) that down-weights older observations,
# and P is the covariance matrix of the coefficient estimate.

# One-step-ahead prediction is then:
#     v_pred = phi^T @ x

# This file mirrors the structure of AdaptiveKalman.py exactly so the two
# modules can be used side-by-side with the same runner script.
# """

# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from scipy.integrate import solve_ivp


# # ── helpers shared with AdaptiveKalman ────────────────────────────────────────

# def _clip(value, lower, upper):
#     return max(lower, min(upper, value))


# def moving_average(y, window=15):
#     y = np.asarray(y, dtype=float)
#     if window <= 1 or len(y) < window:
#         return y.copy()
#     kernel = np.ones(window) / window
#     y_pad  = np.pad(y, (window // 2, window - 1 - window // 2), mode="edge")
#     return np.convolve(y_pad, kernel, mode="valid")


# # ── RLS-based AR predictor ────────────────────────────────────────────────────

# class ARPredictor:
#     """
#     Online AR(p) predictor using Recursive Least Squares.

#     Parameters
#     ----------
#     order   : int    AR order p — number of lagged values used
#     lam     : float  forgetting factor (0.95–1.0). Lower = faster adaptation,
#                      higher = more stable on slowly varying seas.
#     delta   : float  initial diagonal value of covariance matrix P.
#                      Large delta = diffuse prior (learns quickly from scratch).
#     """

#     def __init__(self, order: int = 10, lam: float = 0.97, delta: float = 1e4):
#         self.p     = order
#         self.lam   = lam
#         self.phi   = np.zeros(order)          # AR coefficients
#         self.P     = np.eye(order) * delta    # RLS covariance matrix
#         self.buf   = np.zeros(order)          # circular buffer of recent velocities

#     def _regressor(self) -> np.ndarray:
#         """Return x = [v[n-1], v[n-2], ..., v[n-p]] from the buffer."""
#         return self.buf.copy()

#     def update(self, v_new: float) -> float:
#         """
#         Ingest the latest true velocity measurement, update the AR coefficients
#         via RLS, and return the one-step-ahead prediction for the NEXT step.

#         Call order each time step:
#             v_pred_next = predictor.update(v_measured_now)
#         """
#         x = self._regressor()   # regressor built from previous observations

#         # RLS update (only meaningful once buffer has at least one real sample)
#         Px      = self.P @ x
#         denom   = self.lam + x @ Px
#         gain    = Px / denom if denom > 1e-12 else np.zeros(self.p)
#         innov   = v_new - x @ self.phi
#         self.phi = self.phi + gain * innov
#         #self.P  = (self.P - np.outer(gain, x) @ self.P) / self.lam
#         self.P = (self.P - np.outer(gain, x) @ self.P) / self.lam + np.eye(self.p) * 1e-6

#         # shift buffer: drop oldest, insert v_new at front
#         self.buf = np.roll(self.buf, 1)
#         self.buf[0] = v_new

#         # one-step-ahead prediction using the updated coefficients
#         x_next  = self.buf.copy()            # regressor for step n+1
#         v_pred  = self.phi @ x_next
#         return float(v_pred)


# # ── main solver ───────────────────────────────────────────────────────────────

# def calculate_variable_damping_power(history, cutoff=50):
#     """Identical signature to the Kalman version for drop-in compatibility."""
#     v = np.array(history["v"][cutoff:])
#     c = np.array(history["c_pto"][cutoff:])
#     p = c * v ** 2
#     return p, np.mean(p)


# def solve_cummins_stepwise_adaptive_ar(
#     body,
#     A_heave_inf,
#     t_kernel,
#     kernel,
#     K_heave,
#     F_ex_time,
#     C_pto_base,
#     K_pto,
#     t_span,
#     dt            = 0.05,
#     adaptive_gain = 5e4,
#     cpto_min      = 1e4,
#     cpto_max      = 5e5,
#     ar_order      = 10,
#     ar_lambda     = 0.97,
#     seed          = 42,
# ):
#     """
#     Solve the Cummins equation with AR-adaptive PTO damping.

#     The radiation memory kernel is omitted from the inner ODE (same
#     simplification as AdaptiveKalman.solve_cummins_stepwise_adaptive_kalman)
#     so that both adaptive methods are compared on equal footing.

#     Parameters
#     ----------
#     body, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, K_pto, t_span, dt
#         — identical to the Kalman runner; passed straight through.
#     C_pto_base    : float   baseline damping around which adaptive gain acts
#     adaptive_gain : float   scales |v_pred| → delta_c_pto  (same as Kalman)
#     cpto_min/max  : float   hard clamp on c_pto  (same as Kalman)
#     ar_order      : int     AR model order p  (10 is a good default for ocean waves)
#     ar_lambda     : float   RLS forgetting factor
#     seed          : int     kept for API symmetry (AR model is deterministic)
#     """

#     print("Running adaptive AR simulation")

#     M   = body.mass + A_heave_inf
#     ar  = ARPredictor(order=ar_order, lam=ar_lambda)

#     history = {
#         "t":       [0.0],
#         "x":       [0.0],
#         "v":       [0.0],
#         "v_pred":  [0.0],   # AR one-step-ahead prediction (diagnostic)
#         "c_pto":   [C_pto_base],
#     }

#     t_now  = 0.0
#     x_now  = 0.0
#     v_now  = 0.0
#     c_pto  = C_pto_base
#     v_pred = 0.0           # prediction from previous step

#     while t_now < t_span[1]:
#         t_next = t_now + dt

#         # ── integrate one dt step with current (fixed) c_pto ──────────────────
#         def rhs(t, state):
#             x, v = state
#             dvdt = (F_ex_time(t) - c_pto * v - K_heave * x) / M
#             return [v, dvdt]

#         sol   = solve_ivp(rhs, [t_now, t_next], [x_now, v_now])
#         x_now = sol.y[0, -1]
#         v_now = sol.y[1, -1]
#         t_now = t_next

#         # ── AR update: feed true velocity, get prediction for next step ────────
#         v_pred = ar.update(v_now)

#         # ── adaptive damping law (same form as Kalman version) ─────────────────
#         v_pred_clamped = np.clip(v_pred, -5.0, 5.0)   # adjust limit to your expected velocity range
#         c_pto = _clip(C_pto_base + adaptive_gain * abs(v_pred_clamped), cpto_min, cpto_max)

#         history["t"].append(t_now)
#         history["x"].append(x_now)
#         history["v"].append(v_now)
#         history["v_pred"].append(v_pred)
#         history["c_pto"].append(c_pto)

#     return history


# # ── plotting ──────────────────────────────────────────────────────────────────

# def plot_results_ar(history_const, history_kalman, history_ar,
#                     p_const, p_kalman, p_ar,
#                     output_dir="results", show=True):
#     """
#     Three-way comparison: constant / Kalman adaptive / AR adaptive.
#     Saves three figures to output_dir.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     cut = 50    # drop transient (matches cutoff in calculate_variable_damping_power)

#     t_c  = history_const["t"][cut:]
#     t_k  = history_kalman["t"][cut:]
#     t_ar = history_ar["t"][cut:]

#     # ── 1. Power comparison ───────────────────────────────────────────────────
#     fig, ax = plt.subplots(figsize=(11, 4))
#     ax.plot(t_c,  moving_average(p_const),  label="Constant damping",  linewidth=1.8)
#     ax.plot(t_k,  moving_average(p_kalman), label="Kalman adaptive",   linewidth=1.8)
#     ax.plot(t_ar, moving_average(p_ar),     label="AR adaptive",       linewidth=1.8)
#     ax.set_title("Instantaneous absorbed power — three-way comparison", fontsize=13)
#     ax.set_xlabel("Time (s)");  ax.set_ylabel("Power (W)")
#     ax.legend();  ax.grid(alpha=0.3)
#     fig.tight_layout()
#     fig.savefig(os.path.join(output_dir, "power_comparison.png"), dpi=300)

#     # ── 2. Velocity prediction quality (AR only) ──────────────────────────────
#     fig, ax = plt.subplots(figsize=(11, 4))
#     t_ar_full = history_ar["t"][cut:]
#     v_true    = history_ar["v"][cut:]
#     v_pred    = history_ar["v_pred"][cut:]
#     ax.plot(t_ar_full, v_true, label="True velocity",         linewidth=1.8)
#     ax.plot(t_ar_full, v_pred, label="AR predicted velocity", linewidth=1.4, ls="--")
#     ax.set_title("AR one-step-ahead velocity prediction", fontsize=13)
#     ax.set_xlabel("Time (s)");  ax.set_ylabel("Velocity (m/s)")
#     ax.legend();  ax.grid(alpha=0.3)
#     fig.tight_layout()
#     fig.savefig(os.path.join(output_dir, "ar_prediction.png"), dpi=300)

#     # ── 3. Adaptive damping coefficient ───────────────────────────────────────
#     fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=False)
#     axes[0].plot(history_kalman["t"][cut:], history_kalman["c_pto"][cut:],
#                  linewidth=1.6, color="tab:orange")
#     axes[0].set_title("Kalman adaptive c_pto", fontsize=11)
#     axes[0].set_ylabel("Damping (Ns/m)");  axes[0].grid(alpha=0.3)

#     axes[1].plot(history_ar["t"][cut:], history_ar["c_pto"][cut:],
#                  linewidth=1.6, color="tab:green")
#     axes[1].set_title("AR adaptive c_pto", fontsize=11)
#     axes[1].set_xlabel("Time (s)");  axes[1].set_ylabel("Damping (Ns/m)")
#     axes[1].grid(alpha=0.3)
#     fig.suptitle("Adaptive PTO damping coefficient over time", fontsize=13)
#     fig.tight_layout()
#     fig.savefig(os.path.join(output_dir, "damping_comparison.png"), dpi=300)

#     if show:
#         plt.show()

"""
AdaptiveAR.py
=============
Adaptive PTO damping using an AutoRegressive (AR) velocity predictor with
Recursive Least Squares (RLS) coefficient estimation.

x_est is a multi-step open-loop displacement forecast: the AR model is
propagated N_HORIZON steps forward using only its own outputs (no true
measurements), with per-step clamping to prevent exponential blow-up from
transient coefficient instability. The Euler-integrated result gives a
realistic predicted displacement that diverges from truth over the horizon.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp


def _clip(value, lower, upper):
    return max(lower, min(upper, value))


def moving_average(y, window=15):
    y = np.asarray(y, dtype=float)
    if window <= 1 or len(y) < window:
        return y.copy()
    kernel = np.ones(window) / window
    y_pad  = np.pad(y, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


class ARPredictor:
    """
    Online AR(p) predictor using Recursive Least Squares.

    Stability guard
    ---------------
    After each RLS update, the AR polynomial roots are checked. If any root
    lies on or outside the unit circle (unstable AR process), the coefficients
    are reset toward a stable prior. This prevents the open-loop horizon
    rollout from exploding exponentially when the sea is calm and RLS has
    drifted into an ill-conditioned region.
    """

    def __init__(self, order: int = 10, lam: float = 0.97, delta: float = 1e4,
                 v_max: float = 3.0):
        self.p     = order
        self.lam   = lam
        self.phi   = np.zeros(order)
        self.P     = np.eye(order) * delta
        self.buf   = np.zeros(order)
        self.v_max = v_max          # hard clamp on any single predicted velocity [m/s]

    def _is_stable(self) -> bool:
        """Return True if all roots of the AR polynomial are inside the unit circle."""
        # AR polynomial: 1 - phi_1*z^-1 - ... - phi_p*z^-p = 0
        # Rearranged as companion polynomial coefficients for numpy
        coeffs = np.concatenate([[1.0], -self.phi])
        roots  = np.roots(coeffs)
        return bool(np.all(np.abs(roots) < 1.0))

    def update(self, v_new: float) -> float:
        """Ingest true velocity, update RLS, return one-step prediction."""
        x     = self.buf.copy()
        Px    = self.P @ x
        denom = self.lam + x @ Px
        gain  = Px / denom if denom > 1e-12 else np.zeros(self.p)
        self.phi += gain * (v_new - x @ self.phi)
        self.P    = (self.P - np.outer(gain, x) @ self.P) / self.lam + np.eye(self.p) * 1e-6

        # Stability guard: if AR process is unstable, damp coefficients toward zero
        if not self._is_stable():
            self.phi *= 0.5     # shrink toward zero — preserves sign structure

        self.buf    = np.roll(self.buf, 1)
        self.buf[0] = v_new
        return float(np.clip(self.phi @ self.buf, -self.v_max, self.v_max))

    def predict_ahead(self, steps: int) -> np.ndarray:
        """
        Open-loop multi-step forecast. Each prediction feeds back as the next
        regressor with no true measurements. Per-step clamping prevents the
        exponential blow-up that occurs when RLS has produced transient
        coefficient instability (visible as the 4000 m spike in the plot).

        Returns velocity predictions for steps n+1 .. n+steps.
        """
        buf   = self.buf.copy()
        preds = np.zeros(steps)
        for i in range(steps):
            v_i      = float(self.phi @ buf)
            v_i      = np.clip(v_i, -self.v_max, self.v_max)   # per-step clamp
            preds[i] = v_i
            buf       = np.roll(buf, 1)
            buf[0]    = v_i
        return preds


# Horizon length for x_est. At dt=0.05 s, 40 steps = 2.0 s ahead.
N_HORIZON = 40

def calculate_variable_damping_power(history, cutoff=50):
    v = np.array(history["v"][cutoff:])
    c = np.array(history["c_pto"][cutoff:])
    return c * v**2, np.mean(c * v**2)


def solve_cummins_stepwise_adaptive_ar(
    body, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time,
    C_pto_base, K_pto, t_span,
    dt=0.05, adaptive_gain=5e4, cpto_min=1e4, cpto_max=5e5,
    ar_order=10, ar_lambda=0.97, seed=42,
):
    print("Running adaptive AR simulation")

    M  = body.mass + A_heave_inf
    ar = ARPredictor(order=ar_order, lam=ar_lambda)

    history = {
        "t": [0.0], "x": [0.0], "v": [0.0],
        "v_pred": [0.0], "x_est": [0.0], "c_pto": [C_pto_base],
    }

    t_now = 0.0; x_now = 0.0; v_now = 0.0; c_pto = C_pto_base

    while t_now < t_span[1]:
        t_next = t_now + dt

        def rhs(t, state):
            x, v = state
            return [v, (F_ex_time(t) - c_pto * v - K_heave * x) / M]

        sol   = solve_ivp(rhs, [t_now, t_next], [x_now, v_now])
        x_now = sol.y[0, -1]
        v_now = sol.y[1, -1]
        t_now = t_next

        # Update AR with true velocity — one-step prediction used for damping law
        v_pred_1    = ar.update(v_now)
        v_pred_safe = float(np.clip(v_pred_1,
                                    -3.0 * abs(v_now + 1e-6),
                                     3.0 * abs(v_now + 1e-6)))
        v_pred_safe = float(np.clip(v_pred_safe, -5.0, 5.0))
        c_pto       = _clip(C_pto_base + adaptive_gain * abs(v_pred_safe),
                            cpto_min, cpto_max)

        # Multi-step open-loop displacement estimate.
        # predict_ahead clamps each step internally, so no single bad phi
        # can cause exponential blow-up across the horizon.
        v_horizon = ar.predict_ahead(N_HORIZON)
        x_est     = x_now + np.sum(v_horizon) * dt

        # Final safety clamp on x_est — catches any residual numerical issues
        x_est = float(np.clip(x_est, -50.0, 50.0))

        history["t"].append(t_now)
        history["x"].append(x_now)
        history["v"].append(v_now)
        history["v_pred"].append(v_pred_1)
        history["x_est"].append(x_est)
        history["c_pto"].append(c_pto)

    return history


def plot_results_ar(history_const, history_kalman, history_ar,
                    p_const, p_kalman, p_ar, output_dir="results", show=True):
    os.makedirs(output_dir, exist_ok=True)
    cut = 50

    # ── 1. Power ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(history_const["t"][cut:],  moving_average(p_const),  label="Constant damping", lw=1.8)
    ax.plot(history_kalman["t"][cut:], moving_average(p_kalman), label="Kalman adaptive",  lw=1.8)
    ax.plot(history_ar["t"][cut:],     moving_average(p_ar),     label="AR adaptive",      lw=1.8)
    ax.set_title("Instantaneous absorbed power — three-way comparison", fontsize=13)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Power (W)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "power_comparison.png"), dpi=300)

    # ── 2. Displacement estimation ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(
        f"Displacement estimation — Kalman vs AR\n"
        f"(AR x_est = {N_HORIZON}-step open-loop projection, {N_HORIZON*0.05:.1f} s ahead)",
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
                 label=f"AR predicted displacement ({N_HORIZON*0.05:.1f} s ahead)",
                 lw=1.4, ls="--", color="tab:green")
    axes[1].set_title("AR predictor", fontsize=11)
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Displacement (m)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "displacement_estimation.png"), dpi=300)

    # ── 3. Damping coefficients ───────────────────────────────────────────────
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


# def plot_ar_coefficients(history_ar, ar_order=10, output_dir="results", show=True):
#     os.makedirs(output_dir, exist_ok=True)
#     ar = ARPredictor(order=ar_order)
#     coeff_history = []
#     for v in history_ar["v"]:
#         ar.update(v)
#         coeff_history.append(ar.phi.copy())
#     coeff_history = np.array(coeff_history)

#     fig, ax = plt.subplots(figsize=(11, 4))
#     for i in range(ar_order):
#         ax.plot(history_ar["t"], coeff_history[:, i], lw=1.0, alpha=0.7, label=f"phi_{i+1}")
#     ax.set_title(f"AR({ar_order}) coefficient evolution (RLS)", fontsize=13)
#     ax.set_xlabel("Time (s)"); ax.set_ylabel("Coefficient value")
#     ax.legend(fontsize=7, ncol=5); ax.grid(alpha=0.3)
#     fig.tight_layout()
#     fig.savefig(os.path.join(output_dir, "ar_coefficients.png"), dpi=300)
#     if show:
#         plt.show()