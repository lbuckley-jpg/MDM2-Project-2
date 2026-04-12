import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize


def compute_K(t_grid, omega, radiation_damping):
    """Approximate the retardation function K(t) with a Riemann sum."""
    K = np.zeros_like(t_grid, dtype=float)
    for idx in range(len(omega) - 1):
        dw = omega[idx + 1] - omega[idx]
        K += radiation_damping[idx] * np.cos(omega[idx] * t_grid) * dw
    return (2.0 / np.pi) * K


def prony_model(t_grid, *params):
    """Evaluate a real-valued Prony series."""
    params = np.asarray(params, dtype=float)
    n_terms = len(params) // 4
    result = np.zeros_like(t_grid, dtype=float)

    for idx in range(n_terms):
        alpha_r = params[4 * idx]
        beta_r = params[4 * idx + 1]
        alpha_i = params[4 * idx + 2]
        beta_i = params[4 * idx + 3]
        exp_term = np.exp(beta_r * t_grid)
        result += alpha_r * exp_term * np.cos(beta_i * t_grid)
        result -= alpha_i * exp_term * np.sin(beta_i * t_grid)

    return result


def init_prony_guess(omega_n, zeta, K_max, n_terms):
    """Build a stable initial Prony guess from physical scales."""
    omega_d = omega_n * np.sqrt(max(1e-8, 1.0 - zeta**2))
    decay_rate = -abs(zeta * omega_n)

    guess = np.zeros((n_terms, 4), dtype=float)
    for idx in range(n_terms):
        freq_factor = 0.5 + 0.5 * idx
        guess[idx, 0] = K_max / (idx + 1.0)
        guess[idx, 1] = decay_rate * (0.5 + 0.3 * idx)
        guess[idx, 2] = K_max / (2.0 * (idx + 1.0))
        guess[idx, 3] = omega_d * freq_factor
    return guess


def fit_prony_coefficients(t_grid, omega, radiation_damping, mass, added_mass_inf, pto_damping, hydrostatic_stiffness, n_terms):
    """Fit stable Prony coefficients for the radiation memory model."""
    print("Initialising function: fit_prony_coefficients")
    K_data = compute_K(t_grid, omega, radiation_damping)
    omega_n = np.sqrt(hydrostatic_stiffness / (mass + added_mass_inf))
    zeta = pto_damping / (2.0 * np.sqrt(hydrostatic_stiffness * (mass + added_mass_inf)))
    K_max = np.max(np.abs(K_data))
    initial = init_prony_guess(omega_n, zeta, K_max, n_terms)

    def objective(flat_params):
        fitted = prony_model(t_grid, *flat_params)
        return np.sum((K_data - fitted) ** 2)

    bounds = []
    for _ in range(n_terms):
        bounds.extend(
            [
                (-np.inf, np.inf),
                (-10.0, -1e-3),
                (-np.inf, np.inf),
                (-10.0, 10.0),
            ]
        )

    print("finding stable prony initial fit")
    stable_fit = minimize(
        objective,
        initial.flatten(),
        method="L-BFGS-B",
        bounds=bounds,
    )
    lower_bounds = np.array([b[0] for b in bounds], dtype=float)
    upper_bounds = np.array([b[1] for b in bounds], dtype=float)
    start = stable_fit.x if stable_fit.success else initial.flatten()

    # L-BFGS-B already minimizes sum((K_data - prony_model)^2) with the same bounds.
    # A second bounded curve_fit -> least_squares(TRF) on ~1000 points repeats that
    # at huge cost (many Jacobian SVDs). With bounds, pass max_nfev (not maxfev) to least_squares.
    if stable_fit.success:
        print("L-BFGS-B converged; using result (skipping curve_fit)")
        coeffs = start.reshape(-1, 4)
    else:
        print("L-BFGS-B did not converge; bounded least_squares on subsampled grid")
        n_fit = min(250, len(t_grid))
        idx = np.unique(np.linspace(0, len(t_grid) - 1, n_fit, dtype=int))
        t_fit = t_grid[idx]
        K_fit = K_data[idx]
        try:
            popt, _ = curve_fit(
                prony_model,
                t_fit,
                K_fit,
                p0=start,
                bounds=(lower_bounds, upper_bounds),
                method="dogbox",
                max_nfev=2000,
                ftol=1e-8,
                xtol=1e-8,
            )
            coeffs = popt.reshape(-1, 4)
        except Exception:
            print("curve_fit failed, using L-BFGS-B / initial vector")
            coeffs = start.reshape(-1, 4)

    print("finished: fit_prony_coefficients")
    return coeffs, K_data


def interleave(a, b):
    """Interleave two vectors [a0,b0,a1,b1,...]."""
    out = np.empty(a.size + b.size, dtype=float)
    out[0::2] = a
    out[1::2] = b
    return out


def wec_state_rhs(time, state, params, prony_coeffs, forcing, latch_state):
    """State dynamics for [z, z_dot, I_R1, I_I1, ...]."""
    mass, added_mass_inf, pto_damping, latch_gain, hydrostatic_stiffness = params
    alpha_r = prony_coeffs[:, 0]
    beta_r = prony_coeffs[:, 1]
    alpha_i = prony_coeffs[:, 2]
    beta_i = prony_coeffs[:, 3]

    z = state[0]
    z_dot = state[1]
    I_r = state[2::2]
    I_i = state[3::2]

    u = float(latch_state(time))
    f_ex = float(forcing(time))

    z_ddot = (
        f_ex
        - hydrostatic_stiffness * z
        - (pto_damping + latch_gain * u) * z_dot
        - np.sum(I_r)
    ) / (mass + added_mass_inf)

    dI_r = beta_r * I_r - beta_i * I_i + alpha_r * z_dot
    dI_i = beta_i * I_r + beta_r * I_i + alpha_i * z_dot

    return [z_dot, z_ddot, *interleave(dI_r, dI_i)]


def state_adjoint_rhs(time, lam, params, prony_coeffs, state_sol, latch_state):
    """Adjoint dynamics solved backwards in time."""
    mass, added_mass_inf, pto_damping, latch_gain, hydrostatic_stiffness = params
    alpha_r = prony_coeffs[:, 0]
    beta_r = prony_coeffs[:, 1]
    alpha_i = prony_coeffs[:, 2]
    beta_i = prony_coeffs[:, 3]

    state = state_sol.sol(time)
    z_dot = state[1]
    u = float(latch_state(time))

    lam_1 = lam[0]
    lam_2 = lam[1]
    lam_r = lam[2::2]
    lam_i = lam[3::2]

    dlam_1 = lam_2 * hydrostatic_stiffness / (mass + added_mass_inf)
    dlam_2 = (
        -2.0 * pto_damping * z_dot
        - lam_1
        + lam_2 * (pto_damping + latch_gain * u) / (mass + added_mass_inf)
        - np.sum(lam_r * alpha_r + lam_i * alpha_i)
    )
    dlam_r = -lam_r * beta_r - lam_i * beta_i + lam_2 / (mass + added_mass_inf)
    dlam_i = lam_r * beta_i - lam_i * beta_r

    return [dlam_1, dlam_2, *interleave(dlam_r, dlam_i)]


def build_excitation_time_series(forcing, t_grid):
    """Sample a forcing callback and return sampled values + interpolator."""
    print("Initialising function: build_excitation_time_series")
    values = np.array([float(forcing(t)) for t in t_grid], dtype=float)
    forcing_interp = interp1d(t_grid, values, kind="cubic", fill_value="extrapolate")
    print("finished: build_excitation_time_series")
    return values, forcing_interp


def solve_pontryagin_latching(omega, radiation_damping, t_grid, params, forcing, max_iter=100, n_terms=7, tol_changes=10):
    """Iterate state-adjoint equations until latch profile converges."""
    print("Initialising function: solve_pontryagin_latching")
    mass, added_mass_inf, pto_damping, latch_gain, hydrostatic_stiffness = params
    prony_coeffs, K_data = fit_prony_coefficients(
        t_grid=t_grid,
        omega=omega,
        radiation_damping=radiation_damping,
        mass=mass,
        added_mass_inf=added_mass_inf,
        pto_damping=pto_damping,
        hydrostatic_stiffness=hydrostatic_stiffness,
        n_terms=n_terms,
    )

    x0 = np.zeros(2 + 2 * n_terms, dtype=float)
    u_vals = np.ones_like(t_grid, dtype=float)
    u_new = np.zeros_like(u_vals)

    print("running pontryagin iteration loop")
    for iter_idx in range(max_iter):
        n_changes = int(np.sum(u_new != u_vals))
        print(f"iteration {iter_idx + 1}/{max_iter}, latch changes={n_changes}")
        if n_changes <= tol_changes:
            print("convergence reached for latch profile")
            break

        u_vals = u_new.copy()
        u_func = interp1d(t_grid, u_vals, kind="previous", fill_value="extrapolate")

        state_sol = solve_ivp(
            wec_state_rhs,
            [t_grid[0], t_grid[-1]],
            x0,
            args=(params, prony_coeffs, forcing, u_func),
            dense_output=True,
            method="LSODA",
            max_step=0.5,
            t_eval=t_grid,
        )

        adjoint_sol = solve_ivp(
            state_adjoint_rhs,
            [t_grid[-1], t_grid[0]],
            np.zeros_like(x0),
            args=(params, prony_coeffs, state_sol, u_func),
            dense_output=True,
            method="LSODA",
            max_step=0.5,
            t_eval=t_grid[::-1],
        )

        for idx, t_now in enumerate(t_grid):
            lam_2 = float(adjoint_sol.sol(t_now)[1])
            z_dot = float(state_sol.sol(t_now)[1])
            u_new[idx] = 1.0 if (-lam_2 * latch_gain * z_dot) > 0.0 else 0.0

    print("running final state solve with converged latch sequence")
    # Final solve with converged latch sequence.
    latch_interp = interp1d(t_grid, u_new, kind="previous", fill_value="extrapolate")
    state_sol = solve_ivp(
        wec_state_rhs,
        [t_grid[0], t_grid[-1]],
        x0,
        args=(params, prony_coeffs, forcing, latch_interp),
        dense_output=True,
        method="LSODA",
        max_step=0.5,
        t_eval=t_grid,
    )

    history = {
        "t": t_grid,
        "x": state_sol.y[0],
        "v": state_sol.y[1],
        "u": u_new,
    }
    print("finished: solve_pontryagin_latching")
    return history, prony_coeffs, K_data


def solve_no_latch(omega, radiation_damping, t_grid, params, forcing, n_terms=7):
    """Baseline simulation with no latching."""
    print("Initialising function: solve_no_latch")
    mass, added_mass_inf, pto_damping, _, hydrostatic_stiffness = params
    prony_coeffs, _ = fit_prony_coefficients(
        t_grid=t_grid,
        omega=omega,
        radiation_damping=radiation_damping,
        mass=mass,
        added_mass_inf=added_mass_inf,
        pto_damping=pto_damping,
        hydrostatic_stiffness=hydrostatic_stiffness,
        n_terms=n_terms,
    )

    x0 = np.zeros(2 + 2 * n_terms, dtype=float)
    u_func = lambda _: 0.0
    print("running no-latch state solve")
    state_sol = solve_ivp(
        wec_state_rhs,
        [t_grid[0], t_grid[-1]],
        x0,
        args=(params, prony_coeffs, forcing, u_func),
        dense_output=True,
        method="LSODA",
        max_step=0.5,
        t_eval=t_grid,
    )

    print("finished: solve_no_latch")
    return {"t": t_grid, "x": state_sol.y[0], "v": state_sol.y[1]}


def calc_power(v_history, pto_damping, trim_steps=50):
    """Instantaneous and mean absorbed PTO power."""
    print("Initialising function: calc_power")
    v_used = np.asarray(v_history[trim_steps:], dtype=float)
    p_inst = pto_damping * v_used**2
    p_mean = float(np.mean(p_inst)) if p_inst.size > 0 else 0.0
    print("finished: calc_power")
    return p_inst, p_mean


def plot_pontryagin_results(history_opt, history_no_latch, forcing_values, p_inst_opt, p_inst_no):
    """Plot latch state, motion and power comparison."""
    print("Initialising function: plot_pontryagin_results")
    t = history_opt["t"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].step(t, history_opt["u"], where="post", label="Latch state u(t)")
    axes[0].set_ylabel("u")
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(t, history_opt["x"], label="Displacement with Pontryagin latch")
    axes[1].plot(t, history_no_latch["x"], label="Displacement without latch", alpha=0.8)
    scale = np.max(np.abs(history_opt["x"])) / max(np.max(np.abs(forcing_values)), 1e-12)
    axes[1].plot(t, forcing_values * scale, "--", label="Scaled excitation force")
    axes[1].set_ylabel("z (m)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    axes[2].plot(t[50:], p_inst_opt, label="Power with Pontryagin latch")
    axes[2].plot(t[50:], p_inst_no, label="Power without latch", alpha=0.8)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Power (W)")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    plt.tight_layout()
    plt.show()
    print("finished: plot_pontryagin_results")
