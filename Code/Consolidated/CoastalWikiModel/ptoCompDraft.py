"""
pto_comparison.py
=================
Comparison of two Power Take-Off (PTO) control strategies for a heaving
wave energy converter (WEC):

  1. LINEAR (viscous) PTO  — F_pto = b_pto_dim · ζ̇
     Solved analytically; closed-form optimal b_pto derived by calculus.

  2. COULOMB (dry-friction) PTO  — F_pto = C_pto · sign(ζ̇)
     Solved by direct time-domain ODE integration of the full equation of
     motion — no harmonic-balance / linearisation assumptions.

Physical model
--------------
  (M)ζ̈ + B_rad·ζ̇ + ρgA·ζ = ρgA·a·cos(ωt) − F_pto

  M      : buoy mass (chosen for resonance)
  B_rad  : radiation (intrinsic) damping
  ρgA    : hydrostatic stiffness  (K)
  a      : wave amplitude
  ω      : wave angular frequency

Normalised damping b = B_rad·ω / (m·K),  m = Mω²/K
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Physical parameters
# ─────────────────────────────────────────────────────────────────────────────

RHO   = 1025.0          # seawater density           [kg/m³]
G     = 9.81            # gravitational acceleration  [m/s²]
A_WL  = np.pi * 2**2   # waterplane area  (R = 2 m)  [m²]
a     = 0.5             # wave amplitude              [m]
b     = 0.3             # normalised intrinsic damping (dimensionless)
freq  = 0.2             # wave frequency              [Hz]
omega = 2 * np.pi * freq


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Derived / resonance quantities
# ─────────────────────────────────────────────────────────────────────────────

K = RHO * G * A_WL                         # hydrostatic stiffness  [N/m]

# Mass that maximises linear-PTO power at this frequency
#   d/dm [ power(m) ] = 0  →  m* = 1 / (1 + b²)
m_peak = 1.0 / (1.0 + b**2)
M      = (K * m_peak) / omega**2            # buoy mass              [kg]

# Dimensional radiation damping recovered from normalised b
#   b_norm = B_rad·ω / (m·K)
B_rad  = b * m_peak * K / omega             # [N·s/m]

# Wave excitation force amplitude  (Froude–Krylov, long-wave limit)
F0 = K * a                                  # [N]

print("=" * 60)
print("SYSTEM PARAMETERS")
print("=" * 60)
print(f"  Wave frequency   ω  = {omega:.4f} rad/s  ({freq} Hz)")
print(f"  Hydrostatic K       = {K:.4f} N/m")
print(f"  Resonant mass  M    = {M:.2f} kg")
print(f"  Radiation damp B    = {B_rad:.4f} N·s/m")
print(f"  Norm. parameter m   = {m_peak:.6f}")
print(f"  Excitation amp  F0  = {F0:.2f} N")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — LINEAR PTO  (analytical)
#
#   F_pto = b_pto_dim · ζ̇    (viscous damper)
#
#   Amplitude:   ζ₀ = a / √[(1−m)² + m²(b + b_pto)²]
#   Power:       P  = ½ · b_pto_dim · ω² · ζ₀²
#                   = M a² ω³ b_pto / [ 2((1−m)² + m²(b + b_pto)²) ]
#   Optimum:     b_pto* = √[b² + ((1−m)/m)²]   (from dP/d(b_pto) = 0)
# ─────────────────────────────────────────────────────────────────────────────

def linear_amplitude(a, m, b, b_pto):
    """Heave amplitude for linear (viscous) PTO."""
    return a / np.sqrt((1.0 - m)**2 + m**2 * (b + b_pto)**2)


def linear_power(a, M, omega, m, b, b_pto):
    """Extracted power for linear PTO."""
    return (M * a**2 * omega**3 * b_pto /
            (2.0 * ((1.0 - m)**2 + m**2 * (b + b_pto)**2)))


def linear_optimal_b_pto(m, b):
    """Closed-form optimal normalised b_pto."""
    return np.sqrt(b**2 + ((1.0 - m) / m)**2)


# Sweep b_pto for plotting
b_pto_sweep = np.linspace(0.0, 3.0, 2000)
linear_power_sweep = np.array([
    linear_power(a, M, omega, m_peak, b, bp) for bp in b_pto_sweep
])

# Analytical optimum
b_pto_star    = linear_optimal_b_pto(m_peak, b)
zeta_lin_star = linear_amplitude(a, m_peak, b, b_pto_star)
power_lin_star = linear_power(a, M, omega, m_peak, b, b_pto_star)

# Dimensional b_pto for reporting
b_pto_dim_star = b_pto_star * m_peak * K / omega    # [N·s/m]

print("\n" + "=" * 60)
print("LINEAR PTO  —  analytical optimum")
print("=" * 60)
print(f"  b_pto*  (norm.)    = {b_pto_star:.6g}")
print(f"  b_pto*  (dim.)     = {b_pto_dim_star:.6g} N·s/m")
print(f"  |ζ₀|   at optimum = {zeta_lin_star:.6g} m")
print(f"  Power* at optimum = {power_lin_star:.6g} W")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — COULOMB PTO  (time-domain)
#
#   F_pto = C_pto · sign(ζ̇)
#
#   Full EOM integrated directly:
#     M·ζ̈ + B_rad·ζ̇ + K·ζ = F0·cos(ωt) − C_pto·sign(ζ̇)
#
#   Power extracted: P = ⟨C_pto·|ζ̇|⟩  (true time average, no assumptions)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_coulomb(C_pto,
                     n_cycles=60, n_settle=30,
                     pts_per_cycle=300,
                     eps_vel=0.01,
                     return_trace_cycles=4):   # ← new
    T = 2 * np.pi / omega

    def eom(t, y):
        zeta, zdot = y
        F_ex  = F0 * np.cos(omega * t)
        F_pto = C_pto * np.tanh(zdot / eps_vel)
        zddot = (F_ex - B_rad * zdot - K * zeta - F_pto) / M
        return [zdot, zddot]

    t_end  = n_cycles * T
    t_eval = np.linspace(0.0, t_end, n_cycles * pts_per_cycle)

    sol = solve_ivp(eom, (0.0, t_end), [0.0, 0.0],
                    t_eval=t_eval, method='RK45',
                    rtol=1e-6, atol=1e-8,
                    max_step=T / pts_per_cycle)

    if not sol.success:
        print(f"  WARNING: solver failed for C_pto={C_pto:.3g} — {sol.message}")
        return np.nan, np.nan, None, None

    # steady-state mask (for power / amplitude)
    ss_mask  = sol.t >= n_settle * T
    zdot_ss  = sol.y[1, ss_mask]
    zeta_ss  = sol.y[0, ss_mask]

    power = float(np.mean(C_pto * np.abs(zdot_ss)))
    amp   = float(np.max(np.abs(zeta_ss)))

    # trace: last `return_trace_cycles` periods — already computed, free to return
    trace_start = (n_cycles - return_trace_cycles) * T
    tr_mask  = sol.t >= trace_start
    t_trace  = sol.t[tr_mask] - sol.t[tr_mask][0]   # shift to start at 0
    z_trace  = sol.y[0, tr_mask]

    return power, amp, t_trace, z_trace


def grid_search_coulomb(C_min=0.0, C_max=2e6, n_grid=150, verbose=True):
    C_vals  = np.linspace(C_min, C_max, n_grid)
    powers  = np.full(n_grid, np.nan)
    amps    = np.full(n_grid, np.nan)
    traces  = [None] * n_grid          # ← store traces

    for i, C in enumerate(C_vals):
        powers[i], amps[i], t_tr, z_tr = simulate_coulomb(C)
        traces[i] = (t_tr, z_tr)
        if verbose and (i + 1) % 10 == 0:
            p_str = f"{powers[i]:.4g} W" if not np.isnan(powers[i]) else "failed"
            print(f"    {i+1:>4}/{n_grid}  C = {C:.3g} N  →  P = {p_str}")

    idx = int(np.nanargmax(powers))
    return {
        "C_vals":     C_vals,
        "powers":     powers,
        "amps":       amps,
        "idx_opt":    idx,
        "C_star":     C_vals[idx],
        "power_star": powers[idx],
        "amp_star":   amps[idx],
        "trace":      traces[idx],     # ← (t, zeta) at the optimum, ready to plot
    }


print("\n" + "=" * 60)
print("COULOMB PTO  —  time-domain grid search")
print("=" * 60)

res = grid_search_coulomb(C_min=0.0, C_max=2e6, n_grid=50, verbose=True)

print(f"\n  C_pto*  (optimal)  = {res['C_star']:.6g} N")
print(f"  |ζ₀|   at optimum = {res['amp_star']:.6g} m")
print(f"  Power* at optimum = {res['power_star']:.6g} W")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

power_diff    = res["power_star"] - power_lin_star
power_diff_pc = 100.0 * power_diff / power_lin_star

print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"  {'Quantity':<28}  {'Linear PTO':>14}  {'Coulomb PTO':>14}")
print(f"  {'-'*28}  {'-'*14}  {'-'*14}")
print(f"  {'Control parameter':<28}  "
      f"  {f'b_pto* = {b_pto_star:.4f}':>14}  "
      f"  {f'C* = {res['C_star']:.4g} N':>14}")
print(f"  {'Peak amplitude |ζ₀|  [m]':<28}  "
      f"  {zeta_lin_star:>14.6f}  "
      f"  {res['amp_star']:>14.6f}")
print(f"  {'Extracted power  [W]':<28}  "
      f"  {power_lin_star:>14.4f}  "
      f"  {res['power_star']:>14.4f}")
print(f"  {'Power difference  [W]':<28}  "
      f"  {'—':>14}  "
      f"  {power_diff:>+14.4f}")
print(f"  {'Power difference  [%]':<28}  "
      f"  {'—':>14}  "
      f"  {power_diff_pc:>+14.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — PLOTS
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("PTO Method Comparison — Heaving WEC", fontsize=13, fontweight='bold')

# ── Plot 1: Linear PTO power vs b_pto ────────────────────────────────────────
ax = axes[0]
ax.plot(b_pto_sweep, linear_power_sweep / 1e3, color='steelblue', lw=2)
ax.axvline(b_pto_star, color='steelblue', ls='--', lw=1.2, label=f'b_pto* = {b_pto_star:.3f}')
ax.axhline(power_lin_star / 1e3, color='steelblue', ls=':', lw=1.2,
           label=f'P* = {power_lin_star/1e3:.2f} kW')
ax.set_xlabel('Normalised damping parameter b_pto')
ax.set_ylabel('Extracted power  [kW]')
ax.set_title('Linear PTO\nPower vs b_pto  (analytical)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.35)

# ── Plot 2: Coulomb PTO power vs C_pto ───────────────────────────────────────
ax = axes[1]
ax.plot(res['C_vals'] / 1e3, res['powers'] / 1e3, color='firebrick', lw=2)
ax.axvline(res['C_star'] / 1e3, color='firebrick', ls='--', lw=1.2,
           label=f"C* = {res['C_star']/1e3:.1f} kN")
ax.axhline(res['power_star'] / 1e3, color='firebrick', ls=':', lw=1.2,
           label=f"P* = {res['power_star']/1e3:.2f} kW")
ax.set_xlabel('Coulomb force C_PTO  [kN]')
ax.set_ylabel('Extracted power  [kW]')
ax.set_title('Coulomb PTO\nPower vs C_PTO  (time-domain)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.35)

# ── Plot 3: Steady-state time traces at respective optima ────────────────────
ax = axes[2]
T  = 2 * np.pi / omega

# Linear PTO — reconstruct steady-state sinusoid
t_trace = np.linspace(0, 4 * T, 800)
# phase lag from linear EOM: φ = atan2(m²(b+b_pto), 1−m)
phi_lin = np.arctan2(m_peak**2 * (b + b_pto_star), 1 - m_peak)
zeta_lin_trace = zeta_lin_star * np.cos(omega * t_trace - phi_lin)

ax.plot(t_trace, zeta_lin_trace, color='steelblue', lw=1.8,
        label=f'Linear  (|ζ₀| = {zeta_lin_star:.3f} m)')

# Coulomb trace — already computed during grid search, no extra integration needed
t_plot, z_plot = res["trace"]

ax.plot(t_plot, z_plot, color='firebrick', lw=1.8, ls='--',
        label=f"Coulomb  (|ζ₀| = {res['amp_star']:.3f} m)")

ax.set_xlabel('Time  [s]')
ax.set_ylabel('Heave displacement ζ  [m]')
ax.set_title('Steady-state heave\nat respective optima')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.35)
ax.set_xlim(0, t_plot[-1])

plt.tight_layout()
plt.savefig('pto_comparison.png')
print("\n  Figure saved → pto_comparison.png")
plt.close()