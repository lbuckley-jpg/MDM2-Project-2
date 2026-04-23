"""
Comparing coulomb and linear damping in JONSWAP and sinusodal sea states
using capytaine solver
"""

import os
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import capytaine as cpt

# -----------------------------------------------------------------------------
# pParameters
# -----------------------------------------------------------------------------
RADIUS = 5.0
RHO    = 1025.0
G      = 9.81

# Sea state (jonswap)
HS     = 1.5                    # significant wave height [m]
TP     = 6.01                   # peak period [s]
GAMMA  = 3.3                    # jonswaop peak enhancement factor
SEED   = 42                     

OMEGA_P = 2.0 * np.pi / TP

# capytaine simulation resolution
MESH_RES   = (60, 60)
N_OMEGA_LO = 40                 # |no. of low/high freq.s to solve for 
N_OMEGA_HI = 15                 # |A(ω), B(ω), Fex(ω)
N_COMPS    = 60                 # jonswap no. of harmonics
N_SWEEP    = 10                 # no. of coefficients in each sweep
T_END      = 300.0              # simulation length [s]
T_SETTLE   = 120.0              # discard first time for mean power
DT         = 0.02               

# reference equilibrium mass (half-submerged)
C    = RHO * G * np.pi * RADIUS**2
MASS = 427561.1     #   [kg] calculated from resonant mass at sea state 2
m = float(MASS)

print(f"Sea state: Hs = {HS:.2f} m,  Tp = {TP:.2f} s (omega_p = {OMEGA_P:.3f} rad/s),  gamma = {GAMMA}")
print(f"Buoy: r  = {RADIUS} m,  C = {C:,.1f} N/m")
print(f"Using FIXED mass:  m = {m:,.1f} kg")

# -----------------------------------------------------------------------------
# Capytaine BEM for A(w), B(w), Fex(w)
# same method as in ResonantMass.py for Bound Element Meth.
# -----------------------------------------------------------------------------
mesh = cpt.mesh_sphere(radius=RADIUS, center=(0, 0, 0),
                       resolution=MESH_RES).immersed_part()
body = cpt.FloatingBody(mesh=mesh, name="sphere_r5")
body.add_translation_dof(name="Heave")
body.center_of_mass = np.array([0.0, 0.0, -3.0 * RADIUS / 8.0])

# Frequency grid covers the jonswap + high-frequency vals.
# so that K(t) and A(inf) are well approximated.
omega_grid = np.concatenate([np.linspace(0.15, 3.2, N_OMEGA_LO),
                             np.linspace(3.5, 9.0, N_OMEGA_HI)])

problems  = [cpt.RadiationProblem(body=body, radiating_dof="Heave",
                                  omega=w, rho=RHO, g=G) for w in omega_grid]
problems += [cpt.DiffractionProblem(body=body, wave_direction=0.0,
                                    omega=w, rho=RHO, g=G) for w in omega_grid]

# BEM
solver  = cpt.BEMSolver()
results = solver.solve_all(problems, progress_bar=True)
ds      = cpt.assemble_dataset(results)

A   = ds.added_mass.sel(radiating_dof="Heave",
                        influenced_dof="Heave").values.flatten()
B   = ds.radiation_damping.sel(radiating_dof="Heave",
                               influenced_dof="Heave").values.flatten()
Fex = ds.excitation_force.sel(influenced_dof="Heave").values.flatten()
A_inf = A[-1]

A_of    = interp1d(omega_grid, A,           bounds_error=False,
                   fill_value=(A[0],   A[-1]))
B_of    = interp1d(omega_grid, B,           bounds_error=False,
                   fill_value=(B[0],   B[-1]))
Fmag_of = interp1d(omega_grid, np.abs(Fex), bounds_error=False,
                   fill_value=(np.abs(Fex)[0], np.abs(Fex)[-1]))
Fphs_of = interp1d(omega_grid, np.unwrap(np.angle(Fex)),
                   bounds_error=False, fill_value=0.0)

# coefficients at the peak frequency
A_wp = float(A_of(OMEGA_P))
B_wp = float(B_of(OMEGA_P))

# Report how far the fixed mass is from the resonant mass for diagnostics
M_r_required = C / OMEGA_P**2 - A_wp    # the m that *would* resonate
print(f"\nAt omega_p: A(wp) = {A_wp:,.1f} kg,  B(wp) = {B_wp:,.1f} Ns/m")
print(f"Resonance would require m = C/wp^2 - A(wp) = "
      f"{M_r_required:,.1f} kg  (ratio m / m_res = "
      f"{m / M_r_required:.2f})")

# -----------------------------------------------------------------------------
# Radiation impulse-response kernel calc.
# -----------------------------------------------------------------------------
t_K  = np.arange(0.0, 40.0, DT)
K_t  = np.array([(2.0/np.pi) * np.trapz(B * np.cos(omega_grid * tt),
                                        omega_grid) for tt in t_K])
K_interp = interp1d(t_K, K_t, bounds_error=False, fill_value=0.0)

# -----------------------------------------------------------------------------
# jonswap spectrum
# -----------------------------------------------------------------------------
def jonswap(omega, Hs, Tp, gamma=3.3):
    """jonswap spectrum calculater"""
    wp = 2.0 * np.pi / Tp
    sigma = np.where(omega <= wp, 0.07, 0.09)
    r = np.exp(-0.5 * ((omega - wp) / (sigma * wp))**2)
    S = (omega**(-5)) * np.exp(-1.25 * (wp / omega)**4) * gamma**r
    S[omega <= 0] = 0.0
    m0 = np.trapz(S, omega)
    target_m0 = (Hs / 4.0)**2
    S *= target_m0 / m0
    return S

# freq. sweep
w_lo = max(omega_grid[0], 0.3 * OMEGA_P)
w_hi = min(omega_grid[-1], 3.5 * OMEGA_P)
w_comp = np.linspace(w_lo, w_hi, N_COMPS)
dw = w_comp[1] - w_comp[0]
S_comp = jonswap(w_comp, HS, TP, GAMMA)

eta_amp = np.sqrt(2.0 * S_comp * dw)
m0_num  = np.trapz(S_comp, w_comp)
Hs_num  = 4.0 * np.sqrt(m0_num)
sigma_eta_jsw = np.sqrt(m0_num)

rng = np.random.default_rng(SEED)
phases = rng.uniform(0.0, 2.0 * np.pi, N_COMPS)

def excitation_force(tt):
    Fm = Fmag_of(w_comp)
    phi = Fphs_of(w_comp)
    return np.sum(eta_amp * Fm * np.cos(w_comp * tt + phases + phi))

# Match monochromatic to JONSWAP in mean wave-elevation energy
a_mono = np.sqrt(2.0) * sigma_eta_jsw
Fmag_p = float(Fmag_of(OMEGA_P))
Fphs_p = float(Fphs_of(OMEGA_P))

def F_ex_mono(tt):
    return a_mono * Fmag_p * np.cos(OMEGA_P * tt + Fphs_p)

print(f"\nJONSWAP realisation:  Hs(numeric) = {Hs_num:.3f} m  "
      f"(target {HS:.3f}),  sigma_eta = {sigma_eta_jsw:.3f} m")
print(f"Monochromatic equivalent amplitude (matched energy): "
      f"a_mono = {a_mono:.3f} m  (= Hs / (2 sqrt(2)))")

# -----------------------------------------------------------------------------
# Cummins simulation with bounded memory
# (ai used to assist writing here)
# -----------------------------------------------------------------------------
MEMORY_LEN = 15.0

def make_rhs(F_ex_func, pto_force):
    times, vels = deque(), deque()

    def rhs(tt, y):
        x, v = y
        while times and times[0] < tt - MEMORY_LEN:
            times.popleft(); vels.popleft()
        times.append(tt); vels.append(v)
        if len(times) > 1:
            th = np.fromiter(times, float, count=len(times))
            vh = np.fromiter(vels,  float, count=len(vels))
            mem = np.trapz(K_interp(tt - th) * vh, th)
        else:
            mem = 0.0
        a = (F_ex_func(tt) - mem + pto_force(v) - C * x) / (m + A_inf)
        return [v, a]

    return rhs

def simulate(F_ex_func, pto_force, t_end=T_END, settle=T_SETTLE):
    sol = solve_ivp(make_rhs(F_ex_func, pto_force), (0.0, t_end), [0.0, 0.0],
                    max_step=DT, dense_output=True, method="LSODA",
                    rtol=1e-4, atol=1e-5)
    tt = np.arange(0.0, t_end, DT)
    x  = sol.sol(tt)[0]
    v  = sol.sol(tt)[1]
    mask = tt > settle
    Fu   = np.array([pto_force(vi) for vi in v[mask]])
    P    = float(np.mean(-Fu * v[mask]))
    return tt, x, v, P

# -----------------------------------------------------------------------------
# sweep both control laws for each sea state
# -----------------------------------------------------------------------------
B_scale = B_wp
B_vals  = np.geomspace(0.05 * B_scale, 40.0 * B_scale, N_SWEEP)

C_scale_mono = Fmag_p * a_mono
C_scale_jsw  = Fmag_p * sigma_eta_jsw * np.sqrt(2)
C_vals_mono  = np.geomspace(0.02 * C_scale_mono, 2.0 * C_scale_mono, N_SWEEP)
C_vals_jsw   = np.geomspace(0.02 * C_scale_jsw,  2.0 * C_scale_jsw,  N_SWEEP)

COULOMB_EPS = 0.02

def coul_f(Cp):
    return lambda v, Cp=Cp, eps=COULOMB_EPS: -Cp * np.tanh(v / eps)

def lin_f(Bp):
    return lambda v, Bp=Bp: -Bp * v

sweeps = {
    ("mono", "linear")  : (B_vals,      F_ex_mono, lin_f),
    ("mono", "coulomb") : (C_vals_mono, F_ex_mono, coul_f),
    ("jsw",  "linear")  : (B_vals,      excitation_force, lin_f),
    ("jsw",  "coulomb") : (C_vals_jsw,  excitation_force, coul_f),
}

powers = {}
for (sea, law), (coeffs, Fex_fn, pto_fn) in sweeps.items():
    print(f"\nSweeping {law:7s} under {sea:4s} sea state ...")
    P = []
    for c in coeffs:
        _, _, _, p = simulate(Fex_fn, pto_fn(c))
        P.append(p)
    powers[(sea, law)] = (np.asarray(coeffs), np.asarray(P))

# -----------------------------------------------------------------------------
# find optimum and run best val case for plotting
# -----------------------------------------------------------------------------
print("\n================  Optimal fixed PTO coefficients  ================")
traces = {}
for (sea, law), (coeffs, P) in powers.items():
    i = int(np.argmax(P))
    label = f"{sea:>5s} / {law:>7s}"
    print(f"{label}:  opt = {coeffs[i]:>11,.0f}  =>  "
          f"P_max = {P[i]/1e3:6.2f} kW")
    Fex_fn = sweeps[(sea, law)][1]
    pto_fn = sweeps[(sea, law)][2]
    tt, x, v, P_opt = simulate(Fex_fn, pto_fn(coeffs[i]))
    traces[(sea, law)] = dict(coef=coeffs[i], t=tt, x=x, v=v, P=P_opt)

P_mono_lin = powers[("mono", "linear")][1].max()
P_mono_cou = powers[("mono", "coulomb")][1].max()
P_jsw_lin  = powers[("jsw",  "linear")][1].max()
P_jsw_cou  = powers[("jsw",  "coulomb")][1].max()
print("\n----------------  Cross-comparison  ----------------")
print(f"  Linear:   P_mono = {P_mono_lin/1e3:6.2f} kW   "
      f"P_JONSWAP = {P_jsw_lin/1e3:6.2f} kW   "
      f"ratio = {P_jsw_lin/P_mono_lin:.3f}")
print(f"  Coulomb:  P_mono = {P_mono_cou/1e3:6.2f} kW   "
      f"P_JONSWAP = {P_jsw_cou/1e3:6.2f} kW   "
      f"ratio = {P_jsw_cou/P_mono_cou:.3f}")
print(f"  P_coulomb / P_linear  (mono)    = {P_mono_cou/P_mono_lin:.3f}  "
      f"(describing-fn says 8/pi^2 = 0.811)")
print(f"  P_coulomb / P_linear  (JONSWAP) = {P_jsw_cou/P_jsw_lin:.3f}")

# -----------------------------------------------------------------------------
# plot
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(15, 8))

ax[0, 0].plot(w_comp, S_comp)
ax[0, 0].axvline(OMEGA_P, color='k', ls='--', lw=0.8)
ax[0, 0].set_xlabel(r"$\omega$ [rad/s]"); ax[0, 0].set_ylabel("S(ω) [m²s/rad]")
ax[0, 0].set_title(f"JONSWAP:  Hs={HS} m, Tp={TP} s, γ={GAMMA}")

t_show = np.arange(0.0, min(60.0, T_END), DT)
Fex_m  = np.array([F_ex_mono(tt)      for tt in t_show])
Fex_j  = np.array([excitation_force(tt) for tt in t_show])
ax[0, 1].plot(t_show, Fex_j/1e3, label="JONSWAP", color='C1')
ax[0, 1].plot(t_show, Fex_m/1e3, label="monochromatic", color='C0', lw=1)
ax[0, 1].set_xlabel("t [s]"); ax[0, 1].set_ylabel(r"$F_{ex}$ [kN]")
ax[0, 1].set_title("Wave excitation force"); ax[0, 1].legend(fontsize=8)

for ((sea, law), (coeffs, P)) in powers.items():
    c0 = 'C0' if sea == "mono" else 'C1'
    ls = '-' if law == "linear" else '--'
    ax[0, 2].semilogx(coeffs, P/1e3, marker='o', color=c0, ls=ls,
                      label=f"{sea} / {law}")
ax[0, 2].set_xlabel("PTO coefficient"); ax[0, 2].set_ylabel("Mean power [kW]")
ax[0, 2].set_title("Parameter sweeps"); ax[0, 2].grid(alpha=0.3)
ax[0, 2].legend(fontsize=8)

win = lambda t: (t > max(T_SETTLE, T_END - 40)) & (t < T_END)
for (sea, law), tr in traces.items():
    m_ = win(tr['t'])
    ls = '-' if law == "linear" else '--'
    c0 = 'C0' if sea == "mono" else 'C1'
    ax[1, 0].plot(tr['t'][m_], tr['x'][m_], ls=ls, color=c0,
                  label=f"{sea} / {law}")
ax[1, 0].set_xlabel("t [s]"); ax[1, 0].set_ylabel("heave x(t) [m]")
ax[1, 0].set_title("Steady-state heave response"); ax[1, 0].legend(fontsize=8)

labels = ["mono / linear", "mono / coulomb", "JONSWAP / linear",
          "JONSWAP / coulomb"]
vals   = [P_mono_lin, P_mono_cou, P_jsw_lin, P_jsw_cou]
colors = [(0.12, 0.47, 0.71, 0.9),
          (0.12, 0.47, 0.71, 0.5),
          (1.00, 0.50, 0.05, 0.9),
          (1.00, 0.50, 0.05, 0.5)]
bars   = ax[1, 1].bar(labels, np.array(vals)/1e3, color=colors)
ax[1, 1].set_ylabel("Peak mean absorbed power [kW]")
ax[1, 1].set_title("Best fixed-PTO performance")
ax[1, 1].tick_params(axis='x', rotation=20)
for b, v in zip(bars, vals):
    ax[1, 1].text(b.get_x() + b.get_width()/2, b.get_height()*1.01,
                  f"{v/1e3:.2f}", ha='center', fontsize=8)

jl = traces[("jsw", "linear")]
jc = traces[("jsw", "coulomb")]
m_l = win(jl['t']); m_c = win(jc['t'])
ax[1, 2].plot(jl['t'][m_l], jl['coef']*jl['v'][m_l]**2 / 1e3,
              label=f"linear, B*={jl['coef']:.0f}")
ax[1, 2].plot(jc['t'][m_c], jc['coef']*np.abs(jc['v'][m_c]) / 1e3,
              '--', label=f"coulomb, C*={jc['coef']:.0f}")
ax[1, 2].set_xlabel("t [s]")
ax[1, 2].set_ylabel("Instantaneous PTO power [kW]")
ax[1, 2].set_title("JONSWAP: instantaneous absorbed power")
ax[1, 2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("jonswap_vs_sinusoidal.png", dpi=150)
plt.show()