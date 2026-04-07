import numpy as np
import matplotlib.pyplot as plt

# physical constants 
rho = 1025.0
g   = 9.81
a   = 0.5

# damping parameters
b     = 0.3
b_pto = 0.3
beta  = b + b_pto

# target wave frequency
f_target  = 0.12
omega_t   = 2 * np.pi * f_target

# buoy radius
R = 2.0
A = np.pi * R**2

# analytical inversion using coastal wiki
m_peak       = 1.0 / (1.0 + beta**2)
M_analytical = (rho * g * A * m_peak) / omega_t**2

# mass array sweep
M_array = np.linspace(100, M_analytical * 3, 5000)
m_array = (M_array * (omega_t**2)) / (rho * g * A)
y_amp = a / (np.sqrt((1 - m_array)**2 + (m_array**2) * beta**2))

peak_idx = np.argmax(y_amp)
M_peak   = M_array[peak_idx]
y_peak   = y_amp[peak_idx]

print(f"Optimal M (analytical) : {M_analytical:.2f} kg")
print(f"Optimal M (numerical)  : {M_peak:.2f} kg")

# ── PLOT ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(M_array, y_amp, color="steelblue", lw=2, label="Heave amplitude")
ax.axvline(M_analytical, ls="--", color="crimson", lw=1.8,
           label=f"Optimal M = {M_analytical:.1f} kg")
ax.scatter([M_peak], [y_peak], color="crimson", zorder=5, s=60,
           label=f"Peak y response = {y_peak:.3f} m  at  M = {M_peak:.1f} kg")

ax.set_xlabel("Buoy mass  M  [kg]")
ax.set_ylabel("Heave amplitude [m]")
ax.set_title(
    f"Heave amplitude at f = {f_target} Hz vs buoy mass\n"
    f"R = {R} m,  beta = {beta},  m_peak = {m_peak:.3f}"
)
ax.legend()
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()