# version rewritten, checked and editted to contain no AI

import numpy as np
import matplotlib.pyplot as plt

# physical contants
g = 9.81
rho = 1025  # seawater density 
a = 0.5     # wave amp. (avg)  [m]

b = 0.1     # intrinsic dampening
            # - can be estimated using hydrodynamic models

# buoy dimensions
R = 2
A = np.pi * R**2

# target wave freq
freq = 0.12     # [hz]
omega = 2 * np.pi * freq

# find resonant mass (m=1)
K = rho * g * A         # hydrostatic stiffness [N/m]
M = K / omega**2      # resonant mass [kg]

# find m0 given target freq
def m_param(M_val, omega):
    return (M_val * omega**2) / K

m0 = m_param(M, omega)

print(f"Hydrostatic stiffness K      : {K:.2f} N/m")
print(f"Resonant buoy mass M         : {M:.2f} kg")
print(f"Normalised parameter m       : {m0:.6f}\n")

# find b_pto for max power Eq.(C10)
def b_pto_unconstrained_opt(m):
    output = np.sqrt(((1 - m) / m)**2 + b**2)
    return output

# =======================
# FIXED CONTROL 1: linear pto
# vary b_pto with velocity of buoy
# F_pto = b_pto * Mω * ∂ζ/∂t
#
# Find mass such that power maximised
# =======================

def P_linear(b_pto, M_val, omega, m):
    """
    Average PTO power for velocity-proportional (linear) damping
    Eq. C7 from Coastal Wiki
    """
    num   = a**2 * omega**3 * M_val * b_pto
    denom = 2 * ((1 - m)**2 + m**2 * (b + b_pto)**2)
    return num / denom


b_pto_arr     = np.linspace(0.001, 1.5, 3000)
P_lin_arr     = P_linear(b_pto_arr, M, omega, m0)

b_pto_opt_num = b_pto_arr[np.argmax(P_lin_arr)]
P_lin_max_num = P_lin_arr.max()
b_pto_opt_ana = float(b_pto_unconstrained_opt(m0))     # ≈ b
P_lin_max_ana = M * a**2 * omega**3 / (8 * b)        # Eq. C8

print("Control 1: Linear Optimal PTO")
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
    f"M = {M:.1f} kg,  f = {freq} Hz,  b = {b},  a = {a} m"
)
ax1.legend()
ax1.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()