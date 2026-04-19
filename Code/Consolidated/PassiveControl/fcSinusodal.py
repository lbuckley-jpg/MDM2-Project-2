"""
Linear damping  vs.  Coulomb damping
====================================

Compares the mean absorbed power of two fixed-control PTO strategies on a
half-submerged spherical buoy (r = 2 m) oscillating in monochromatic heave
waves of frequency f = 0.2 Hz (omega = 2*pi*0.2 rad/s).

PTO laws:
    linear    :   F_u(t) = - B_pto * x_dot(t)
    coulomb   :   F_u(t) = - C_pto * sign(x_dot(t))

For each law the coefficient is swept across a wide range so that an
*optimal fixed* value can be identified and the two peak powers compared.
The Cummins equation is used for the buoy dynamics, with added mass /
radiation damping / excitation force obtained from Capytaine.

Run:     python buoy_linear_vs_coulomb.py
Needs:   capytaine >= 2.0, numpy, scipy, matplotlib
         pip install capytaine numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import capytaine as cpt

# -----------------------------------------------------------------------------
# 0. Physical constants, buoy geometry, target wave
# -----------------------------------------------------------------------------
# Set FAST_MODE = True for a quick (< ~1 min) run with coarser sweeps and a
# shorter simulation window. Set to False for the full-resolution version.
FAST_MODE = True

RADIUS   = 2.0
RHO      = 1025.0
G        = 9.81

F_WAVE   = 0.2                    # wave frequency [Hz]
OMEGA    = 2.0 * np.pi * F_WAVE   # [rad/s]
WAVE_AMP = 1.0                    # incident wave amplitude [m]

# Tunables that change between fast and full mode
if FAST_MODE:
    MESH_RES   = (14, 14)         # sphere mesh resolution (ntheta, nphi)
    N_OMEGA_LO = 18               # freqs in main band
    N_OMEGA_HI = 6                # extra high freqs for A_inf / kernel
    N_SWEEP    = 8                # number of coefficients in each sweep
    T_END      = 80.0             # simulation length [s]
    T_SETTLE   = 40.0             # discard first T_SETTLE s for mean power
    DT         = 0.05             # coarse step
else:
    MESH_RES   = (20, 20)
    N_OMEGA_LO = 40
    N_OMEGA_HI = 15
    N_SWEEP    = 18
    T_END      = 160.0
    T_SETTLE   = 80.0
    DT         = 0.02

# Heave hydrostatic stiffness of the sphere at the waterline
C = RHO * G * np.pi * RADIUS**2
# Reference (equilibrium / half-submerged) mass, kept for comparison only
m_eq = (2.0 / 3.0) * np.pi * RADIUS**3 * RHO

print(f"Wave:   f = {F_WAVE} Hz   ->  omega = {OMEGA:.4f} rad/s, "
      f"T = {1/F_WAVE:.2f} s")
print(f"Buoy:   C = {C:,.1f} N/m,   m_equilibrium = {m_eq:,.1f} kg "
      f"(for reference)")

# ---m is set to resonant mass after A(omega) is calculated---

# -----------------------------------------------------------------------------
# 1. Capytaine BEM for the hydrodynamic coefficients
# -----------------------------------------------------------------------------
mesh = cpt.mesh_sphere(radius=RADIUS, center=(0, 0, 0),
                       resolution=MESH_RES).immersed_part()
body = cpt.FloatingBody(mesh=mesh, name="sphere_r2")
body.add_translation_dof(name="Heave")
body.center_of_mass = np.array([0.0, 0.0, -3.0 * RADIUS / 8.0])

# Dense sweep + a few high frequencies so A_inf is well approximated and
# the kernel K(t) = (2/pi) int B(w) cos(w t) dw is resolved.
omega_grid = np.concatenate([np.linspace(0.1, 3.0, N_OMEGA_LO),
                             np.linspace(3.2, 8.0, N_OMEGA_HI)])

problems  = [cpt.RadiationProblem(body=body, radiating_dof="Heave",
                                  omega=w, rho=RHO, g=G) for w in omega_grid]
problems += [cpt.DiffractionProblem(body=body, wave_direction=0.0,
                                    omega=w, rho=RHO, g=G) for w in omega_grid]

solver  = cpt.BEMSolver()
results = solver.solve_all(problems, progress_bar=True)
ds      = cpt.assemble_dataset(results)

A   = ds.added_mass.sel(radiating_dof="Heave",
                        influenced_dof="Heave").values.flatten()
B   = ds.radiation_damping.sel(radiating_dof="Heave",
                               influenced_dof="Heave").values.flatten()
Fex = ds.excitation_force.sel(influenced_dof="Heave").values.flatten()

A_inf = A[-1]                                 # A at the highest solved freq
A_w   = float(np.interp(OMEGA, omega_grid, A))
B_w   = float(np.interp(OMEGA, omega_grid, B))
Fex_mag = float(np.interp(OMEGA, omega_grid, np.abs(Fex)))
Fex_phs = float(np.interp(OMEGA, omega_grid, np.angle(Fex)))

print(f"\nA(omega) = {A_w:,.1f} kg,  B(omega) = {B_w:,.1f} Ns/m")
print(f"|Fex(omega)| = {Fex_mag:,.1f} N   (per unit wave amp)")

# -----------------------------------------------------------------------------
# 1b. Tune the buoy mass for resonance at the wave frequency
#     => m = C / omega^2  -  A(omega)
# -----------------------------------------------------------------------------
M_r = C / OMEGA**2                 # required total (m + A) for resonance
m   = M_r - A_w                    # buoy mass that makes the system resonate
if m <= 0:
    raise RuntimeError(
        f"Resonant mass would be non-physical (m = {m:.1f} kg). "
        "Wave frequency is too high for this geometry/stiffness.")

print(f"\n--- Resonance tuning ---")
print(f"Required M_r = C/omega^2        = {M_r:,.1f} kg")
print(f"Added mass A(omega)            = {A_w:,.1f} kg")
print(f"=> Buoy mass m = M_r - A(omega) = {m:,.1f} kg")
print(f"   (m_equilibrium was           {m_eq:,.1f} kg "
      f"-- ratio m / m_eq = {m / m_eq:.2f})")

# -----------------------------------------------------------------------------
# 2. Radiation impulse-response kernel  K(t)
# -----------------------------------------------------------------------------
'''AI check'''

dt   = DT
t_K  = np.arange(0.0, 40.0, dt)
K_t  = np.array([(2.0/np.pi) * np.trapz(B * np.cos(omega_grid * tt),
                                        omega_grid)
                 for tt in t_K])
K_interp = interp1d(t_K, K_t, bounds_error=False, fill_value=0.0)

# -----------------------------------------------------------------------------
# 3. Cummins equation with a user-supplied PTO law
# -----------------------------------------------------------------------------
def excitation(tt):
    return WAVE_AMP * Fex_mag * np.cos(OMEGA * tt + Fex_phs)

# Truncate the radiation-memory convolution to the last MEMORY_LEN seconds.
# K(t) decays quickly for this buoy so ~15 s captures essentially all of it,
# while keeping the per-step cost bounded (otherwise it's O(N^2) total).
MEMORY_LEN = 15.0

def make_rhs(pto_force):
    """pto_force(v) -> instantaneous PTO force on the buoy."""
    from collections import deque
    times = deque()
    vels  = deque()

    def rhs(tt, y):
        x, v = y
        # Drop history older than MEMORY_LEN so the convolution cost is bounded
        while times and times[0] < tt - MEMORY_LEN:
            times.popleft(); vels.popleft()
        times.append(tt); vels.append(v)
        if len(times) > 1:
            th  = np.fromiter(times, dtype=float, count=len(times))
            vh  = np.fromiter(vels,  dtype=float, count=len(vels))
            mem = np.trapz(K_interp(tt - th) * vh, th)
        else:
            mem = 0.0
        F  = excitation(tt)
        Fu = pto_force(v)
        a  = (F - mem + Fu - C * x) / (m + A_inf)
        return [v, a]

    return rhs

def simulate(pto_force, t_end=T_END, settle=T_SETTLE, method="LSODA"):
    # LSODA auto-switches between Adams (non-stiff) and BDF (stiff), which
    # is ideal for Coulomb's near-discontinuity around v = 0.
    sol = solve_ivp(make_rhs(pto_force), (0.0, t_end), [0.0, 0.0],
                    max_step=dt, dense_output=True, method=method,
                    rtol=1e-4, atol=1e-5)
    tt = np.arange(0.0, t_end, dt)
    x  = sol.sol(tt)[0]
    v  = sol.sol(tt)[1]
    # Use the steady-state tail to compute mean absorbed power:
    #     P = < -F_u(v) * v >   (power flowing OUT of the fluid into the PTO)
    mask = tt > settle
    Fu   = np.array([pto_force(vi) for vi in v[mask]])
    P    = float(np.mean(-Fu * v[mask]))
    return tt, x, v, P

# -----------------------------------------------------------------------------
# 4. Sweep both control laws
# -----------------------------------------------------------------------------
# Linear  :  F_u = -B_pto * v
B_scale = B_w                                        # natural unit
B_vals  = np.geomspace(0.05 * B_scale, 40.0 * B_scale, N_SWEEP)

# Coulomb :  F_u = -C_pto * sign(v)
# A sensible scale is the excitation force amplitude (per unit wave amp).
C_scale = Fex_mag * WAVE_AMP
C_vals  = np.geomspace(0.02 * C_scale, 2.0  * C_scale, N_SWEEP)

print("\nSweeping linear damping ...")
P_lin = []
for Bp in B_vals:
    _, _, _, P = simulate(lambda v, Bp=Bp: -Bp * v)
    P_lin.append(P)
P_lin = np.array(P_lin)

print("Sweeping Coulomb damping ...")
P_cou = []
# Wider eps => the tanh transition zone is ~2 cm/s; still a faithful
# approximation of sign(v) for heave velocities ~ O(0.5-1.5 m/s), but avoids
# the extreme stiffness that forced tiny time steps with eps = 1e-3.
COULOMB_EPS = 0.02
for Cp in C_vals:
    _, _, _, P = simulate(
        lambda v, Cp=Cp, eps=COULOMB_EPS: -Cp * np.tanh(v / eps))
    P_cou.append(P)
P_cou = np.array(P_cou)

i_lin, i_cou = int(np.argmax(P_lin)), int(np.argmax(P_cou))
print("\n--- Optimal fixed coefficients ---")
print(f"Linear  : B_pto* = {B_vals[i_lin]:>10,.0f} Ns/m       "
      f"P_max = {P_lin[i_lin]/1e3:6.2f} kW")
print(f"Coulomb : C_pto* = {C_vals[i_cou]:>10,.0f} N          "
      f"P_max = {P_cou[i_cou]/1e3:6.2f} kW")
print(f"Ratio   : P_lin / P_coulomb = {P_lin[i_lin] / P_cou[i_cou]:.3f}")

# -----------------------------------------------------------------------------
# 5. Time-trace comparison at the optimal coefficients
# -----------------------------------------------------------------------------
tL, xL, vL, _ = simulate(lambda v, Bp=B_vals[i_lin]: -Bp * v)
tC, xC, vC, _ = simulate(
    lambda v, Cp=C_vals[i_cou], eps=COULOMB_EPS: -Cp * np.tanh(v / eps))

# -----------------------------------------------------------------------------
# 6. Plots
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

ax[0, 0].semilogx(B_vals, P_lin / 1e3, 'o-')
ax[0, 0].axvline(B_vals[i_lin], color='k', ls='--', lw=0.8)
ax[0, 0].set_xlabel(r"$B_{PTO}$ [Ns/m]")
ax[0, 0].set_ylabel("Mean absorbed power [kW]")
ax[0, 0].set_title("Linear damping sweep")
ax[0, 0].grid(True, which='both', alpha=0.3)

ax[0, 1].semilogx(C_vals, P_cou / 1e3, 's-', color='C1')
ax[0, 1].axvline(C_vals[i_cou], color='k', ls='--', lw=0.8)
ax[0, 1].set_xlabel(r"$C_{PTO}$ [N]")
ax[0, 1].set_ylabel("Mean absorbed power [kW]")
ax[0, 1].set_title("Coulomb damping sweep")
ax[0, 1].grid(True, which='both', alpha=0.3)

# Show a 30 s window of the steady-state tail (scales with T_END)
win_start = max(T_SETTLE, T_END - 30.0)
win_end   = T_END
window    = (tL > win_start) & (tL < win_end)
ax[1, 0].plot(tL[window], xL[window],
              label=f"Linear, B*={B_vals[i_lin]:.0f}")
ax[1, 0].plot(tC[window], xC[window], '--',
              label=f"Coulomb, C*={C_vals[i_cou]:.0f}")
ax[1, 0].set_xlabel("t [s]")
ax[1, 0].set_ylabel("Heave x(t) [m]")
ax[1, 0].set_title(f"Steady-state response at f = {F_WAVE} Hz")
ax[1, 0].legend(fontsize=8)
ax[1, 0].grid(True, alpha=0.3)

p_lin_inst = np.array([B_vals[i_lin] * vi**2 for vi in vL[window]])
p_cou_inst = np.array([C_vals[i_cou] * abs(vi) for vi in vC[window]])
ax[1, 1].plot(tL[window], p_lin_inst/1e3, label="Linear")
ax[1, 1].plot(tC[window], p_cou_inst/1e3, '--', label="Coulomb")
ax[1, 1].set_xlabel("t [s]")
ax[1, 1].set_ylabel("Instantaneous PTO power [kW]")
ax[1, 1].set_title("Power dissipated into the PTO")
ax[1, 1].legend(fontsize=8)
ax[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/linear_vs_coulomb.png", dpi=150)
plt.show()