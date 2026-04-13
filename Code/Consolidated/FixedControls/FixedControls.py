# version rewritten, checked and editted to contain no AI

import numpy as np
import matplotlib.pyplot as plt

# physical contants
g = 9.81
rho = 1025  # seawater density 
a = 0.5     # wave amp. (avg)  [m]

b = 0.3     # intrinsic dampening
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


# =====================================
# FIXED CONTROL 2: linear pto
# vary b_pto with velocity of buoy
# 
# This is calculated analytically
# F_pto = b_pto * Mω * ∂ζ/∂t
# P_pto = (M a^2 ω^3)/8b
#
# Find mass such that power maximised
# =====================================

def P_linear_resonant(M, a, omega, b, b_pto):
    return (M * a**2 * omega**3 * b_pto) / (2 * (b + b_pto)**2)

b_pto_arr = np.linspace(0.001, 1.5, 3000)
P_lin_arr = P_linear_resonant(M, a, omega, b, b_pto_arr)

b_pto_opt_num = b_pto_arr[np.argmax(P_lin_arr)]
P_lin_max_num = P_lin_arr.max()
P_lin_max_ana = M * a**2 * omega**3 / (8 * b)   # eq. C8

# printed and plotted with AI
print("Control 1: Linear Optimal PTO")
print(f"  Optimal b_pto (numerical)  : {b_pto_opt_num:.5f}")
print(f"  Optimal b_pto (analytical) : {b:.5f}  (≈ b = {b})")
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


# =====================================
# FIXED CONTROL 3: fixed damping coefficient
# 
# 
# This is calculated analytically
# F_pto = b_pto * Mω * ∂ζ/∂t
# P_pto = (M a^2 ω^3)/8b
#
# Find mass such that power maximised
# =====================================



plt.show()