# Note: this draft stil contains ai for controls 2 and 3
#     - in process of rewriting method more cleanly onto new doc

import numpy as np
import matplotlib.pyplot as plt

rho = 1025.0
g   = 9.81
a   = 0.5           # wave amplitude [m]

b = 0.3             # intrinsic dampening - FIND REAL VALUE?
                        # dimensionless radiation + drag damping

# target wave freq
f_target = 0.12
omega_t  = 2 * np.pi * f_target

# buoy dimensions
R = 2.0
A = np.pi * R**2

# resonant mass when m = 1
K = rho * g * A         # hydrostatic stiffness [N/m]
M = K / omega_t**2      # resonant mass [kg]

# find m0 given target freq
def m_param(M_val, omega):
    return (M_val * omega**2) / K

m0 = m_param(M, omega_t)

print(f"Hydrostatic stiffness K      : {K:.2f} N/m")
print(f"Resonant buoy mass M         : {M:.2f} kg")
print(f"Normalised parameter m       : {m0:.6f}\n")

# formula in linear and fixed dampening
def P_linear(b_pto, M_val, omega, m):
    """
    Average PTO power for velocity-proportional (linear) damping.
    Eq. C7 from Coastal Wiki with b_pto as the only free parameter.
    """
    num   = a**2 * omega**3 * M_val * b_pto
    denom = 2 * ((1 - m)**2 + m**2 * (b + b_pto)**2)
    return num / denom

# find optimal b_pto from iranian study ~ !go over maths here!
def b_pto_unconstrained_opt(m):
    output = np.sqrt(((1 - m) / m)**2 + b**2)
    return output

# ═══════════════════════════════════════════════════════════════════════════════
# CONTROL 1 – LINEAR OPTIMAL PTO
#   F_PTO = b_pto · M · ω · dy /dt   (velocity-proportional)
#   Optimal b_pto = b at resonance → P_max = M·a²·ω³/(8b)   [Eq. C8]
# ═══════════════════════════════════════════════════════════════════════════════

b_pto_arr     = np.linspace(0.001, 1.5, 3000)
P_lin_arr     = P_linear(b_pto_arr, M, omega_t, m0)

b_pto_opt_num = b_pto_arr[np.argmax(P_lin_arr)]
P_lin_max_num = P_lin_arr.max()
b_pto_opt_ana = float(b_pto_unconstrained_opt(m0))     # ≈ b
P_lin_max_ana = M * a**2 * omega_t**3 / (8 * b)        # Eq. C8

print("── Control 1: Linear Optimal PTO ──────────────────────────────────")
print(f"  Optimal b_pto (numerical)  : {b_pto_opt_num:.5f}")
print(f"  Optimal b_pto (analytical) : {b_pto_opt_ana:.5f}  (≈ b = {b})")
print(f"  P_max  (numerical)         : {P_lin_max_num:.5f} W")
print(f"  P_max  (Eq. C8 analytical) : {P_lin_max_ana:.5f} W\n")

fig1, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(b_pto_arr, P_lin_arr, color="steelblue", lw=2, label="PTO power")
ax1.axvline(b_pto_opt_num, ls="--", color="crimson", lw=1.8,
            label=f"Optimal b_pto = {b_pto_opt_num:.4f}  (b = {b})")
ax1.scatter([b_pto_opt_num], [P_lin_max_num], color="crimson", s=60, zorder=5,
            label=f"P_max = {P_lin_max_num:.4f} W")
ax1.set_xlabel("PTO damping coefficient  b_pto  [-]")
ax1.set_ylabel("Average PTO power  [W]")
ax1.set_title(
    f"Control 1 – Linear Optimal PTO\n"
    f"M = {M:.1f} kg,  f = {f_target} Hz,  b = {b},  a = {a} m"
)
ax1.legend()
ax1.grid(True, alpha=0.4)
plt.tight_layout()

# ═══════════════════════════════════════════════════════════════════════════════
# CONTROL 2 – FIXED (UNIFORM) PTO DAMPING COEFFICIENT
#   b_pto is locked to the resonance-optimal value and applied unchanged as
#   wave frequency varies.  Mirrors Ma et al. §4.2: the coefficient chosen at
#   the design condition degrades off-peak because m ≠ 1 at other frequencies.
# ═══════════════════════════════════════════════════════════════════════════════

b_pto_fixed = b_pto_opt_num               # locked at resonance optimum

f_arr     = np.linspace(0.04, 0.30, 600)
omega_arr = 2 * np.pi * f_arr
m_arr     = m_param(M, omega_arr)         # m varies with ω (M fixed)

# (a) Fixed b_pto across all frequencies
P_fixed_arr = P_linear(b_pto_fixed, M, omega_arr, m_arr)

# (b) Truly adaptive b_pto*(ω) — re-optimised at every frequency
b_pto_adaptive = b_pto_unconstrained_opt(m_arr)
P_adaptive_arr = P_linear(b_pto_adaptive, M, omega_arr, m_arr)

P_fixed_at_target    = float(np.interp(f_target, f_arr, P_fixed_arr))
P_adaptive_at_target = float(np.interp(f_target, f_arr, P_adaptive_arr))

print("── Control 2: Fixed PTO Damping ────────────────────────────────────")
print(f"  Fixed b_pto                : {b_pto_fixed:.5f}")
print(f"  P at design freq (fixed)   : {P_fixed_at_target:.5f} W")
print(f"  P at design freq (adaptive): {P_adaptive_at_target:.5f} W")
print(f"  Peak P adaptive            : {P_adaptive_arr.max():.5f} W\n")

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))

ax2a = axes2[0]
ax2a.plot(f_arr, P_adaptive_arr, color="steelblue", lw=2,
          label="Adaptive b_pto*(ω)")
ax2a.plot(f_arr, P_fixed_arr,    color="darkorange", lw=2, ls="--",
          label=f"Fixed b_pto = {b_pto_fixed:.3f}  (tuned at {f_target} Hz)")
ax2a.axvline(f_target, ls=":", color="gray", lw=1.3,
             label=f"Design freq = {f_target} Hz")
ax2a.set_xlabel("Wave frequency  f  [Hz]")
ax2a.set_ylabel("Average PTO power  [W]")
ax2a.set_title("Power vs frequency")
ax2a.legend()
ax2a.grid(True, alpha=0.4)

ax2b = axes2[1]
ax2b.plot(f_arr, b_pto_adaptive, color="steelblue", lw=2,
          label="Adaptive b_pto*(ω)")
ax2b.axhline(b_pto_fixed, color="darkorange", lw=2, ls="--",
             label=f"Fixed b_pto = {b_pto_fixed:.3f}")
ax2b.axvline(f_target, ls=":", color="gray", lw=1.3,
             label=f"Design freq = {f_target} Hz")
ax2b.set_xlabel("Wave frequency  f  [Hz]")
ax2b.set_ylabel("b_pto  [-]")
ax2b.set_title("Required b_pto vs frequency")
ax2b.legend()
ax2b.grid(True, alpha=0.4)

fig2.suptitle(
    f"Control 2 – Fixed vs Adaptive PTO Damping\n"
    f"M = {M:.1f} kg,  b = {b},  a = {a} m"
)
plt.tight_layout()

# ═══════════════════════════════════════════════════════════════════════════════
# CONTROL 3 – COULOMB (CONSTANT-FORCE) PTO
#   F_PTO = C_pto · sign(dy/dt)
#
#   Describing-function linearisation (harmonic balance) gives:
#       b_pto_eq = 4·C_pto / (π · M · ω² · y_0)        [amplitude-dependent]
#   Then y_0 and b_pto_eq are mutually dependent, solved iteratively:
#       y_0 = K·a / sqrt((K - Mω²)² + (b + b_eq)²·M²ω⁴)
#   Average power (mean of C·|dy/dt| over cycle):
#       P = C_pto · (2/π) · y_0 · ω
# ═══════════════════════════════════════════════════════════════════════════════

def coulomb_iterate(C_pto, M_val, omega, b, tol=1e-9, maxiter=800):
    """
    Self-consistent describing-function solution for Coulomb PTO.
    Returns (zeta0, P_pto) or (nan, nan) if not converged.
    """
    zeta0 = a                             # seed: unloaded wave amplitude

    for _ in range(maxiter):
        zeta0 = max(zeta0, 1e-12)        # guard against collapse to zero
        b_eq  = 4 * C_pto / (np.pi * M_val * omega**2 * zeta0)
        denom = np.sqrt((K - M_val * omega**2)**2
                        + (b + b_eq)**2 * (M_val * omega**2)**2)
        zeta0_new = K * a / denom

        if abs(zeta0_new - zeta0) < tol:
            P = C_pto * (2 / np.pi) * zeta0_new * omega
            return zeta0_new, P

        zeta0 = 0.5 * (zeta0 + zeta0_new)   # damped update for stability

    return np.nan, np.nan

C_pto_arr  = np.linspace(1, 12000, 800)
zeta0_coul = np.full_like(C_pto_arr, np.nan)
P_coul_arr = np.full_like(C_pto_arr, np.nan)

for i, Cp in enumerate(C_pto_arr):
    z0, P = coulomb_iterate(Cp, M, omega_t, b)
    zeta0_coul[i] = z0
    P_coul_arr[i] = P

valid      = ~np.isnan(P_coul_arr)
C_opt_idx  = int(np.nanargmax(P_coul_arr))
C_pto_opt  = C_pto_arr[C_opt_idx]
P_coul_max = P_coul_arr[C_opt_idx]
zeta0_opt  = zeta0_coul[C_opt_idx]

# Equivalent linear b_pto at optimum — expected to be close to b
b_eq_at_opt = 4 * C_pto_opt / (np.pi * M * omega_t**2 * zeta0_opt)

print("── Control 3: Coulomb (Constant-Force) PTO ─────────────────────────")
print(f"  Optimal C_pto              : {C_pto_opt:.2f} N")
print(f"  Converged ζ₀               : {zeta0_opt:.5f} m")
print(f"  Equivalent b_pto at optimum: {b_eq_at_opt:.5f}  (cf. b = {b})")
print(f"  P_max  (Coulomb)           : {P_coul_max:.5f} W")
print(f"  P_max  (linear Eq.C8)      : {P_lin_max_ana:.5f} W")
print(f"  Ratio Coulomb / Linear     : {P_coul_max / P_lin_max_ana:.4f}")
print(f"  (theory: Coulomb/Linear = 8/(π²) ≈ {8 / np.pi**2:.4f})\n")

fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))

ax3a = axes3[0]
ax3a.plot(C_pto_arr[valid], P_coul_arr[valid], color="steelblue", lw=2,
          label="PTO power")
ax3a.axvline(C_pto_opt, ls="--", color="crimson", lw=1.8,
             label=f"Optimal C_pto = {C_pto_opt:.1f} N")
ax3a.scatter([C_pto_opt], [P_coul_max], color="crimson", s=60, zorder=5,
             label=f"P_max = {P_coul_max:.4f} W")
ax3a.set_xlabel("Coulomb force magnitude  C_pto  [N]")
ax3a.set_ylabel("Average PTO power  [W]")
ax3a.set_title("Power vs C_pto")
ax3a.legend()
ax3a.grid(True, alpha=0.4)

ax3b = axes3[1]
ax3b.plot(C_pto_arr[valid], zeta0_coul[valid], color="darkorange", lw=2,
          label="Converged heave amplitude  ζ₀")
ax3b.axvline(C_pto_opt, ls="--", color="crimson", lw=1.8,
             label=f"Optimal C_pto = {C_pto_opt:.1f} N")
ax3b.scatter([C_pto_opt], [zeta0_opt], color="crimson", s=60, zorder=5,
             label=f"ζ₀ = {zeta0_opt:.4f} m")
ax3b.set_xlabel("Coulomb force magnitude  C_pto  [N]")
ax3b.set_ylabel("Heave amplitude  ζ₀  [m]")
ax3b.set_title("Heave amplitude vs C_pto")
ax3b.legend()
ax3b.grid(True, alpha=0.4)

fig3.suptitle(
    f"Control 3 – Coulomb PTO\n"
    f"M = {M:.1f} kg,  f = {f_target} Hz,  b = {b},  a = {a} m"
)
plt.tight_layout()

# ── SUMMARY BAR CHART ─────────────────────────────────────────────────────────
labels  = ["Linear optimal\n(b_pto = b)", 
           f"Fixed PTO\n(b_pto = {b_pto_fixed:.3f}, at design freq)",
           "Coulomb optimal\n(C_pto iterated)"]
powers  = [P_lin_max_ana, P_fixed_at_target, P_coul_max]
colors  = ["steelblue", "darkorange", "mediumseagreen"]

fig4, ax4 = plt.subplots(figsize=(9, 5))
bars = ax4.bar(labels, powers, color=colors, width=0.5, edgecolor="black", lw=0.8)
for bar, P in zip(bars, powers):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
             f"{P:.4f} W", ha="center", va="bottom", fontsize=10)
ax4.set_ylabel("Average PTO power  [W]")
ax4.set_title(
    f"Summary – Peak power for all three controls\n"
    f"M = {M:.1f} kg,  f = {f_target} Hz,  b = {b},  a = {a} m"
)
ax4.grid(True, axis="y", alpha=0.4)
ax4.set_ylim(0, max(powers) * 1.2)
plt.tight_layout()

plt.show()