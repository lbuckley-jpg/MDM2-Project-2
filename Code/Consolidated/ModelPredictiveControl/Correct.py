import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, curve_fit
from scipy.linalg import expm
from scipy.stats import linregress
import cvxpy as cp
import capytaine as cpt

cpt.set_logging("WARNING")


def generate_frequencies(N=40, Tp=8.0):
    wp = 2 * np.pi / Tp
    w0 = 0.5 * wp
    wmax = 4.0 * wp
    omega, delta_omega = np.linspace(w0, wmax, N, retstep=True)
    return omega, delta_omega


def jonswap_frequency_amplitudes(omega, delta_omega, Hs=2.0, Tp=12.0, gamma=3.3):
    omega_p = 2 * np.pi / Tp
    alpha = 0.0624 / (0.230 + 0.0336 * gamma - 0.185 / (1.9 + gamma))
    S_pm = (alpha * 9.81**2 / omega**5) * np.exp(-1.25 * (omega_p / omega)**4)
    sigma = np.where(omega <= omega_p, 0.07, 0.09)
    r = np.exp(-((omega - omega_p)**2) / (2 * sigma**2 * omega_p**2))
    S = S_pm * gamma**r
    m0 = np.trapezoid(S, omega)
    S *= (Hs / 4)**2 / m0
    return np.sqrt(2 * S * delta_omega)


def generate_buoy(radius=5.0, mass=5000.0):
    buoy_mesh = cpt.mesh_sphere(radius=radius, center=(0.0, 0.0, 0.0), resolution=(30, 30))
    rotation_center = (0.0, 0.0, 0.0)
    lid_mesh = buoy_mesh.generate_lid(z=0.0)
    buoy = cpt.FloatingBody(
        mesh=buoy_mesh,
        lid_mesh=lid_mesh,
        dofs=cpt.rigid_body_dofs(rotation_center=rotation_center),
        center_of_mass=rotation_center,
        mass=mass,
        name="Point Absorber",
    )
    buoy.radius = radius
    buoy.inertia_matrix = buoy.compute_rigid_body_inertia()
    buoy.hydrostatic_stiffness = buoy.immersed_part().compute_hydrostatic_stiffness()
    return buoy


def solve_with_capytaine(body, omegas, wave_direction=np.pi, water_depth=np.inf, water_density=1000):
    bem_solver = cpt.BEMSolver()
    radiation_problems = [
        cpt.RadiationProblem(
            omega=w,
            body=body.immersed_part(),
            radiating_dof=dof,
            water_depth=water_depth,
            rho=water_density,
        )
        for w in omegas
        for dof in body.dofs
    ]
    diffraction_problems = [
        cpt.DiffractionProblem(
            omega=w,
            body=body.immersed_part(),
            wave_direction=wave_direction,
            water_depth=water_depth,
            rho=water_density,
        )
        for w in omegas
    ]
    radiation_results = bem_solver.solve_all(radiation_problems)
    diffraction_results = [bem_solver.solve(p) for p in diffraction_problems]
    return cpt.assemble_dataset(radiation_results + diffraction_results)


def get_cummins_components(body, capytaine_dataset, wave_direction, wave_amplitudes, omegas, seed):
    A_heave = capytaine_dataset["added_mass"].sel(radiating_dof="Heave", influenced_dof="Heave").values
    B_heave = capytaine_dataset["radiation_damping"].sel(radiating_dof="Heave", influenced_dof="Heave").values

    tail_idx = max(int(len(omegas) * 0.8), 1)
    w_tail = omegas[tail_idx:]
    A_tail = A_heave[tail_idx:]
    valid = w_tail > 0
    if np.sum(valid) >= 2:
        inv_w_sq = 1.0 / (w_tail[valid] ** 2)
        slope, A_heave_inf, *_ = linregress(inv_w_sq, A_tail[valid])
    else:
        A_heave_inf = float(A_heave[-1])

    F_ex_complex = (
        capytaine_dataset["Froude_Krylov_force"] + capytaine_dataset["diffraction_force"]
    ).sel(influenced_dof="Heave", wave_direction=wave_direction).values

    rng = np.random.default_rng(seed)
    epsilon = rng.uniform(0, 2 * np.pi, size=omegas.shape)
    hydro_phase = np.angle(F_ex_complex)
    total_phase = hydro_phase + epsilon
    amplitudes = np.abs(F_ex_complex) * wave_amplitudes

    def F_ex_time(t):
        return np.sum(amplitudes * np.cos(omegas * t + total_phase))

    def F_ex_time_dot(t):
        return np.sum(-amplitudes * omegas * np.sin(omegas * t + total_phase))

    K_heave = float(body.hydrostatic_stiffness.sel(influenced_dof="Heave", radiating_dof="Heave"))

    t_kernel = np.linspace(0, 60, 1000)
    cos_matrix = np.cos(np.outer(t_kernel, omegas))
    kernel = (2 / np.pi) * np.trapezoid(B_heave * cos_matrix, omegas, axis=1)

    return A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, B_heave


def kernel_from_prony(t, coeffs):
    y = np.zeros_like(t, dtype=float)
    for alpha, beta in coeffs:
        y += alpha * np.exp(-beta * t)
    return y


def fit_prony_coefficients(t_grid, kernel_data, n_terms=4):
    y = np.asarray(kernel_data, dtype=float)
    t = np.asarray(t_grid, dtype=float)

    alpha0 = np.linspace(y[0] / max(n_terms, 1), y[0] / max(n_terms, 1), n_terms)
    beta0 = np.linspace(0.05, 1.0, n_terms)
    p0 = np.column_stack([alpha0, beta0]).ravel()

    bounds_low = []
    bounds_high = []
    for _ in range(n_terms):
        bounds_low += [-np.inf, 1e-4]
        bounds_high += [np.inf, 50.0]

    def objective(p):
        coeffs = p.reshape(-1, 2)
        yhat = kernel_from_prony(t, coeffs)
        return np.sum((y - yhat) ** 2)

    res = minimize(objective, p0, method="L-BFGS-B", bounds=list(zip(bounds_low, bounds_high)))
    start = res.x if res.success else p0

    popt, _ = curve_fit(
        lambda tt, *pp: kernel_from_prony(tt, np.array(pp).reshape(-1, 2)),
        t,
        y,
        p0=start,
        bounds=(bounds_low, bounds_high),
        maxfev=50000,
    )
    coeffs = np.array(popt).reshape(-1, 2)
    return coeffs, kernel_from_prony(t, coeffs)


def prony_to_continuous_state_space(prony_coeffs):
    alpha = np.asarray(prony_coeffs[:, 0], dtype=float)
    beta = np.asarray(prony_coeffs[:, 1], dtype=float)
    n = len(alpha)
    Ar = -np.diag(beta)
    Br = np.ones((n, 1))
    Cr = alpha.reshape(1, n)
    Dr = np.zeros((1, 1))
    return Ar, Br, Cr, Dr


def discretize_radiation_model(Ar, Br, Cr, Dr, dt):
    Ad = expm(Ar * dt)
    Bd = np.linalg.solve(Ar, (Ad - np.eye(Ar.shape[0]))) @ Br
    Cd = Cr.copy()
    Dd = Dr.copy()
    return Ad, Bd, Cd, Dd


def build_full_discrete_model(mass, added_mass_inf, K_heave, C_pto, prony_coeffs, dt):
    Ar, Br, Cr, Dr = prony_to_continuous_state_space(prony_coeffs)
    Ard, Brd, Crd, Drd = discretize_radiation_model(Ar, Br, Cr, Dr, dt)

    M = mass + added_mass_inf
    n_r = Ard.shape[0]
    n_state = 2 + n_r

    Ad = np.zeros((n_state, n_state))
    Bd_u = np.zeros((n_state, 1))
    Bd_f = np.zeros((n_state, 1))

    Ad[0, 0] = 1.0
    Ad[0, 1] = dt

    Ad[1, 0] = -dt * K_heave / M
    Ad[1, 1] = 1.0 - dt * C_pto / M

    if n_r > 0:
        Ad[1, 2:] = -dt * Crd.reshape(-1) / M
        Ad[2:, 1] = (Brd * dt).reshape(-1)
        Ad[2:, 2:] = Ard

    Bd_u[1, 0] = -dt / M
    Bd_f[1, 0] = dt / M
    if n_r > 0:
        Bd_f[2:, 0] = np.zeros(n_r)

    return Ad, Bd_u, Bd_f, (Ard, Brd, Crd, Drd)


def simulate_full_cummins_no_control(body, A_inf, t_kernel, kernel, K_heave, F_ex_time, C_pto, K_pto, t_span, dt=0.05):
    M_eff = body.mass + A_inf
    history = {"t": [0.0], "x": [0.0], "v": [0.0], "F_ex": [F_ex_time(0.0)]}

    def rhs(t, state):
        x, v = state
        t_arr = np.array(history["t"])
        v_arr = np.array(history["v"])
        if len(t_arr) < 2:
            memory = 0.0
        else:
            tau = t - t_arr
            k_vals = np.interp(tau, t_kernel, kernel, left=kernel[0], right=0.0)
            memory = np.trapezoid(k_vals * v_arr, t_arr)
        dvdt = (F_ex_time(t) - memory - C_pto * v - (K_heave + K_pto) * x) / M_eff
        return [v, dvdt]

    t_now = 0.0
    x_now = 0.0
    v_now = 0.0
    t_final = t_span[1]

    while t_now < t_final - 1e-12:
        t_next = min(t_now + dt, t_final)
        sol = solve_ivp(rhs, [t_now, t_next], [x_now, v_now], max_step=dt)
        t_now = sol.t[-1]
        x_now = sol.y[0, -1]
        v_now = sol.y[1, -1]
        history["t"].append(t_now)
        history["x"].append(x_now)
        history["v"].append(v_now)
        history["F_ex"].append(F_ex_time(t_now))

    return history


# def solve_mpc(x0, Ad, Bd_u, Bd_f, wave_force_pred, N_horizon, u_limit):
#     n_state = Ad.shape[0]
#     x = cp.Variable((n_state, N_horizon + 1))
#     u = cp.Variable((1, N_horizon))

#     cost = 0
#     constraints = [x[:, 0] == x0]

#     for k in range(N_horizon):
#         constraints += [x[:, k + 1] == Ad @ x[:, k] + Bd_u.flatten() * u[0, k] + Bd_f.flatten() * wave_force_pred[k]]
#         constraints += [cp.abs(u[0, k]) <= u_limit]
#         cost += 1e-3 * cp.square(u[0, k]) - 1.0 * u[0, k] * x[1, k]
#         cost += 1e-2 * cp.square(x[0, k]) + 1e-3 * cp.square(x[1, k])

#     problem = cp.Problem(cp.Minimize(cost), constraints)
#     problem.solve(solver=cp.CLARABEL, verbose=False)

#     if u.value is None:
#         return 0.0, np.zeros(N_horizon), np.zeros(N_horizon + 1)

#     return float(u.value[0, 0]), u.value.flatten(), x.value[1, :-1]

def solve_mpc(x0, Ad, Bd_u, Bd_f, wave_force_pred, N_horizon, u_limit):
    n_state = Ad.shape[0]
    x = cp.Variable((n_state, N_horizon + 1))
    u = cp.Variable((1, N_horizon))

    cost = 0
    constraints = [x[:, 0] == x0]

    for k in range(N_horizon):
        constraints += [
            x[:, k + 1] == Ad @ x[:, k] + Bd_u.flatten() * u[0, k] + Bd_f.flatten() * wave_force_pred[k]
        ]
        constraints += [cp.abs(u[0, k]) <= u_limit]

        cost += 1e-3 * cp.square(u[0, k])
        cost += 1e-2 * cp.square(x[0, k])
        cost += 1e-3 * cp.square(x[1, k])

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP, verbose=False)

    if u.value is None:
        return 0.0, np.zeros(N_horizon), np.zeros(N_horizon + 1)

    return float(u.value[0, 0]), u.value.flatten(), x.value[1, :-1]


def simulate_mpc(body, A_inf, prony_coeffs, K_heave, F_ex_time, t_span, dt, N_horizon, C_pto, u_limit):
    Ad, Bd_u, Bd_f, _ = build_full_discrete_model(body.mass, A_inf, K_heave, C_pto, prony_coeffs, dt)
    x = np.zeros(Ad.shape[0])
    history = {"t": [t_span[0]], "x": [0.0], "v": [0.0], "u": [0.0], "F_ex": [F_ex_time(t_span[0])]}

    for t in np.arange(t_span[0], t_span[1], dt):
        wave_pred = np.array([F_ex_time(t + (k + 1) * dt) for k in range(N_horizon)])
        u_opt, _, _ = solve_mpc(x, Ad, Bd_u, Bd_f, wave_pred, N_horizon, u_limit)
        x = Ad @ x + Bd_u.flatten() * u_opt + Bd_f.flatten() * F_ex_time(t)
        history["t"].append(t + dt)
        history["x"].append(float(x[0]))
        history["v"].append(float(x[1]))
        history["u"].append(float(u_opt))
        history["F_ex"].append(float(F_ex_time(t + dt)))

    return history

    
def calc_power_absorbed_no_control(history, c_pto):
    v = np.array(history['v'][50:])
    p_inst = c_pto * v**2
    p_mean = np.mean(p_inst)
    return p_inst, p_mean

def calc_power_absorbed_mpc(history):
    v = np.array(history['v'][50:])
    u = np.array(history['u'][50:])
    p_inst = u * v
    return p_inst, float(np.mean(p_inst))


def plot_history(history_mpc, history_no_control):
    plt.figure()
    plt.title("Displacement Time Graph for Point Absorber")
    plt.plot(history_mpc["t"], history_mpc["x"], label="with MPC")
    plt.plot(history_no_control["t"], history_no_control["x"], label="no control")
    plt.plot(history_no_control["t"], history_no_control["F_ex"], ls="--", label="excitation force")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement / Force")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_power(history_mpc, history_no_control, p_inst_mpc, p_inst_no_control):
    plt.figure()
    plt.title("Instantaneous Power")
    plt.plot(history_mpc["t"][50:], p_inst_mpc, label="with MPC")
    plt.plot(history_no_control["t"][50:], p_inst_no_control, label="no control")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Wave Simulation")
    parser.add_argument("--save", type=int, default=0)
    parser.add_argument("--tspan", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)

    parser.add_argument("--nfreqcomponents", type=int, default=40)
    parser.add_argument("--peakperiod", type=float, default=8.0)
    parser.add_argument("--significantwaveheight", type=float, default=2.0)

    parser.add_argument("--buoymass", type=float, default=5000.0)
    parser.add_argument("--buoyradius", type=float, default=5.0)

    parser.add_argument("--waterdensity", type=float, default=1000.0)
    parser.add_argument("--waterdepth", type=float, default=np.inf)

    parser.add_argument("--wavedirection", type=float, default=np.pi)

    parser.add_argument("--cpto", type=float, default=20000.0)
    parser.add_argument("--kpto", type=float, default=0.0)

    parser.add_argument("--nprony", type=int, default=4)
    parser.add_argument("--mpc_dt", type=float, default=0.1)
    parser.add_argument("--mpc_N", type=int, default=20)
    parser.add_argument("--u_limit", type=float, default=20000.0)

    args = parser.parse_args()

    buoy = generate_buoy(radius=args.buoyradius, mass=args.buoymass)
    omegas, delta_omega = generate_frequencies(N=args.nfreqcomponents, Tp=args.peakperiod)
    wave_amplitudes = jonswap_frequency_amplitudes(
        omegas, delta_omega, Hs=args.significantwaveheight, Tp=args.peakperiod
    )

    print(f'solve capytaine')
    capytaine_dataset = solve_with_capytaine(
        body=buoy,
        omegas=omegas,
        wave_direction=args.wavedirection,
        water_depth=args.waterdepth,
        water_density=args.waterdensity,
    )

    A_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, B_heave = get_cummins_components(
        body=buoy,
        capytaine_dataset=capytaine_dataset,
        wave_direction=args.wavedirection,
        wave_amplitudes=wave_amplitudes,
        omegas=omegas,
        seed=args.seed,
    )


    print('fit prony coeffs')
    prony_coeffs, k_fit = fit_prony_coefficients(t_kernel, kernel, n_terms=args.nprony)

    print('simulate mpc')
    history_mpc = simulate_mpc(
        body=buoy,
        A_inf=A_inf,
        prony_coeffs=prony_coeffs,
        K_heave=K_heave,
        F_ex_time=F_ex_time,
        t_span=[0.0, args.tspan],
        dt=args.mpc_dt,
        N_horizon=args.mpc_N,
        C_pto=args.cpto,
        u_limit=args.u_limit,
    )
    print('simulate no control')
    history_no_control = simulate_full_cummins_no_control(
        body=buoy,
        A_inf=A_inf,
        t_kernel=t_kernel,
        kernel=kernel,
        K_heave=K_heave,
        F_ex_time=F_ex_time,
        C_pto=args.cpto,
        K_pto=args.kpto,
        t_span=[0.0, args.tspan],
        dt=args.mpc_dt,
    )

    print('calc powers')
    p_inst_mpc, p_mean_mpc = calc_power_absorbed_mpc(history_mpc)
    p_inst_no_control, p_mean_no_control = calc_power_absorbed_no_control(history_no_control, args.cpto)

    print("Mean absorbed power with MPC:", p_mean_mpc)
    print("Mean absorbed power without control:", p_mean_no_control)

    plot_history(history_mpc, history_no_control)
    plot_power(history_mpc, history_no_control, p_inst_mpc, p_inst_no_control)



