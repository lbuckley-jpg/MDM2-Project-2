'''
BEM solve for multiple wave frequencies, then convert to
Cummins time-domain equation and integrate with solve_ivp.
Irregular sea state described by JONSWAP spectrum.
'''

from numpy import pi
from numpy import inf
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import capytaine as cpt

import os
import time

cpt.set_logging('WARNING')

bem_solver = cpt.BEMSolver()


# ─────────────────────────────────────────────────────────────────────────────
def jonswap_frequency_amplitudes(omega, delta_omega, Hs, Tp, gamma=3.3):
    """
    JONSWAP spectral density S(omega) [m^2 s / rad]
    Returns wave amplitude for each frequency component.

    Parameters
    ----------
    omega       : np.ndarray  angular frequencies [rad/s]
    delta_omega : float       frequency resolution [rad/s]
    Hs          : float       significant wave height [m]
    Tp          : float       peak period [s]
    gamma       : float       peak enhancement factor (default 3.3)
    """
    omega_p = 2 * pi / Tp
    alpha   = 0.0624 / (0.230 + 0.0336 * gamma - 0.185 / (1.9 + gamma))

    # Pierson-Moskowitz base spectrum
    S_pm = (alpha * 9.81**2 / omega**5) * np.exp(-1.25 * (omega_p / omega)**4)

    # JONSWAP peak enhancement
    sigma = np.where(omega <= omega_p, 0.07, 0.09)
    r     = np.exp(-((omega - omega_p)**2) / (2 * sigma**2 * omega_p**2))
    S     = S_pm * gamma**r

    # Scale to match desired Hs  (Hs = 4 * sqrt(m0))
    m0 = np.trapezoid(S, omega)
    S *= (Hs / 4)**2 / m0

    return np.sqrt(2 * S * delta_omega)  # wave amplitude per component [m]


# ─────────────────────────────────────────────────────────────────────────────
def generate_frequencies(N, Tp):
    """
    Generate N evenly spaced frequencies centred around the peak frequency.

    Parameters
    ----------
    N  : int    number of frequency components
    Tp : float  peak wave period [s]
    """
    wp   = 2 * np.pi / Tp
    w0   = 0.5 * wp
    wmax = 4.0 * wp

    omega, delta_omega = np.linspace(w0, wmax, N, retstep=True)
    return omega, delta_omega


# ─────────────────────────────────────────────────────────────────────────────
def generate_buoy(radius=5, mass=500):
    """Create a spherical buoy with all rigid-body DOFs."""

    buoy_mesh = cpt.mesh_sphere(radius=radius, center=(0.0, 0.0, 0.0),
                                      resolution=(30, 30))
    rotation_center = (0.0, 0.0, 0.0)

    buoy = cpt.FloatingBody(
        mesh=buoy_mesh,
        dofs=cpt.rigid_body_dofs(rotation_center=rotation_center),
        center_of_mass=rotation_center,
        mass=mass,
        name="Point Absorber",
    )

    buoy.radius = radius  # store for logging

    buoy.inertia_matrix      = buoy.compute_rigid_body_inertia()
    buoy.hydrostatic_stiffness = buoy.immersed_part().compute_hydrostatic_stiffness()

    return buoy


# ─────────────────────────────────────────────────────────────────────────────
def simulate(body, omegas: np.ndarray, delta_omega: float,
             Hs=2.0, Tp=8.0,
             wave_direction=pi,
             water_depth=inf, water_density=1000,
             c_pto=0.0, k_pto=0.0,
             save=True):
    """
    Full Cummins time-domain simulation driven by a JONSWAP irregular sea state.

    Parameters
    ----------
    body          : capytaine FloatingBody
    omegas        : np.ndarray   angular frequencies from generate_frequencies()
    delta_omega   : float        frequency spacing [rad/s]
    Hs            : float        significant wave height [m]
    Tp            : float        peak wave period [s]
    wave_direction: float        wave heading [rad]
    water_depth   : float        water depth [m]  (inf = deep water)
    water_density : float        water density [kg/m^3]
    c_pto         : float        PTO damping coefficient [Ns/m]
    k_pto         : float        PTO stiffness coefficient [N/m]
    save          : bool         write results to CSV
    """

    # ── 1. BEM frequency sweep ────────────────────────────────────────────
    print("Running BEM frequency sweep...")

    radiation_problems = [
        cpt.RadiationProblem(omega=w, body=body.immersed_part(),
                             radiating_dof=dof,
                             water_depth=water_depth, rho=water_density)
        for w in omegas for dof in body.dofs
    ]
    diffraction_problems = [
        cpt.DiffractionProblem(omega=w, body=body.immersed_part(),
                               wave_direction=wave_direction,
                               water_depth=water_depth, rho=water_density)
        for w in omegas
    ]

    radiation_results   = bem_solver.solve_all(radiation_problems)
    diffraction_results = [bem_solver.solve(p) for p in diffraction_problems]

    dataset = cpt.assemble_dataset(radiation_results + diffraction_results)

    # ── 2. Extract Cummins components from dataset ────────────────────────
    print("Extracting Cummins components...")

    # Added mass A33(omega)  [N s^2/m]
    A33 = dataset['added_mass'].sel(
              radiating_dof='Heave', influenced_dof='Heave').values

    # Approximate A33 at infinite frequency using highest frequency value
    A33_inf = float(A33[-1])

    # Radiation damping B33(omega)  [N s/m]
    B33 = dataset['radiation_damping'].sel(
              radiating_dof='Heave', influenced_dof='Heave').values

    # Complex excitation force  F_ex(omega) = F_FK + F_diffraction  [N/m]
    F_ex_complex = (
        dataset['Froude_Krylov_force'] + dataset['diffraction_force']
    ).sel(influenced_dof='Heave', wave_direction=wave_direction).values

    # Hydrostatic stiffness  [N/m]
    K33 = float(body.hydrostatic_stiffness.sel(
                    influenced_dof='Heave', radiating_dof='Heave'))

    # ── 3. Build retardation kernel Kr(t) ────────────────────────────────
    # Kr decays to zero; 60 s is typically sufficient
    t_Kr = np.linspace(0, 60, 1000)

    Kr = np.array([
        (2 / np.pi) * np.trapezoid(B33 * np.cos(omegas * ti), omegas)
        for ti in t_Kr
    ])

    # ── 4. JONSWAP wave amplitudes and F_ex(t) ───────────────────────────
    wave_amps = jonswap_frequency_amplitudes(omegas, delta_omega, Hs, Tp)

    F_amps   = np.abs(F_ex_complex) * wave_amps   # [N]
    F_phases = np.angle(F_ex_complex)              # [rad]

    def F_ex_time(t):
        """Superpose all frequency components into a single time-domain force."""
        return np.sum(F_amps * np.cos(omegas * t + F_phases))

    # ── 5. Build Cummins RHS for solve_ivp ───────────────────────────────
    history = {"t": [], "v": []}

    def rhs(t, state):
        x, v = state

        # Accumulate velocity history
        history['t'].append(t)
        history['v'].append(v)

        t_hist = np.array(history['t'])
        v_hist = np.array(history['v'])

        # Numerical convolution integral  ∫ Kr(t-τ) v(τ) dτ
        if len(t_hist) > 1:
            t_shifted = t - t_hist                          # t - tau >= 0
            Kr_vals   = np.interp(t_shifted, t_Kr, Kr,
                                  left=Kr[0], right=0.0)
            memory = np.trapezoid(Kr_vals * v_hist, t_hist)
        else:
            memory = 0.0

        # Cummins equation solved for acceleration
        v_dot = (
              F_ex_time(t)
            - memory
            - c_pto * v
            - (K33 + k_pto) * x
        ) / (body.mass + A33_inf)

        return [v, v_dot]   # [dx/dt, dv/dt]

    # ── 6. Time integration ───────────────────────────────────────────────
    print("Integrating Cummins equation...")

    t_span = (0.0, 400.0)
    t_eval = np.linspace(0.0, 400.0, 8000)

    solution = solve_ivp(
        rhs,
        t_span=t_span,
        y0=[0.0, 0.0],       # start at rest
        t_eval=t_eval,
        method='RK45',
        max_step=0.05,       # keep history ordered; prevents RK step rejection issues
        rtol=1e-4,
        atol=1e-6,
    )

    if not solution.success:
        print(f"WARNING: solve_ivp did not converge: {solution.message}")

    x_t = solution.y[0]     # heave displacement [m]
    v_t = solution.y[1]     # heave velocity     [m/s]

    # ── 7. Power calculation ──────────────────────────────────────────────
    steady = t_eval > 100.0  # discard transient startup

    P_inst = c_pto * v_t**2
    P_mean = np.mean(P_inst[steady])

    print(f"Mean absorbed power: {P_mean:.1f} W")

    # ── 8. Save results ───────────────────────────────────────────────────
    if save:
        os.makedirs('results', exist_ok=True)
        logfile    = 'results/cummins_results.csv'
        file_exists = os.path.isfile(logfile)

        with open(logfile, 'a') as f:
            if not file_exists:
                f.write("timestamp,Hs_m,Tp_s,c_pto_Ns_m,k_pto_N_m,"
                        "buoy_mass_kg,buoy_radius_m,"
                        "water_depth_m,water_density_kg_m3,"
                        "P_mean_W\n")

            f.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')},"
                f"{Hs},{Tp},{c_pto},{k_pto},"
                f"{body.mass},{body.radius},"
                f"{water_depth},{water_density},"
                f"{P_mean}\n"
            )
        print("Results logged successfully.")

    # ── 9. Plot ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(t_eval, x_t, color='steelblue', linewidth=0.8)
    axes[0].axvline(x=100, color='red', linestyle='--', linewidth=0.8,
                    label='transient cutoff')
    axes[0].set_ylabel('Displacement (m)')
    axes[0].set_title('Heave Displacement')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_eval, v_t, color='darkorange', linewidth=0.8)
    axes[1].axvline(x=100, color='red', linestyle='--', linewidth=0.8)
    axes[1].set_ylabel('Velocity (m/s)')
    axes[1].set_title('Heave Velocity')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_eval, P_inst / 1e3, color='green', linewidth=0.8)
    axes[2].axvline(x=100, color='red', linestyle='--', linewidth=0.8)
    axes[2].axhline(y=P_mean / 1e3, color='black', linestyle='--',
                    linewidth=1.0, label=f'mean = {P_mean / 1e3:.2f} kW')
    axes[2].set_ylabel('Power (kW)')
    axes[2].set_title('Instantaneous Absorbed Power')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/cummins_time_response.png', dpi=150)
    plt.show()

    return solution, P_mean


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # Sea state
    Hs = 2.0   # significant wave height [m]
    Tp = 8.0   # peak period [s]
    N  = 40    # number of frequency components

    # Generate frequencies
    omegas, delta_omega = generate_frequencies(N, Tp)

    # Build buoy
    body = generate_buoy(radius=5, mass=500)

    # Run simulation
    solution, P_mean = simulate(
        body        = body,
        omegas      = omegas,
        delta_omega = delta_omega,
        Hs          = Hs,
        Tp          = Tp,
        wave_direction = pi,
        water_depth    = inf,
        water_density  = 1000,
        c_pto          = 1e4,
        k_pto          = 0.0,
        save           = True,
    )
