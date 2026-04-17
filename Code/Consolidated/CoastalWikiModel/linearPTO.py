import numpy as np
import matplotlib.pyplot as plt

# physical contants
g = 9.81
rho = 1025  # seawater density
a = 0.5     # wave amp. (avg)  [m]

b = 0.3     # intrinsic dampening
            # - can be estimated using hydrodynamic models
b_pto = ...

# buoy dimensions
R = 2
A = np.pi * R**2

# target wave freq
freq = 0.2    # [hz]
omega = 2 * np.pi * freq

# calculate resonant Mass
# Resonant m from d/dm[(1-m)^2 + b^2*m^2] = 0  →  m = 1/(1+b^2)
m_peak       = 1 / (1 + b**2)

# find resonant masses
M = (rho * g * A * m_peak) / omega**2

print(f'\nresonant mass {M:.2f} kg')



def optimal_b_pto_linear(M, omega, A, b, rho=1025, g=9.81):
    m = M * omega**2 / (rho * g * A)
    return np.sqrt(b**2 + ((1- m) / m)**2)


def heave_amplitude_linear(a, M, omega, A, b, b_pto, rho=1025, g=9.81):
    m = M * omega**2 / (rho * g * A)
    return a / np.sqrt((1 - m)**2 + m**2 * (b + b_pto)**2)


def extracted_power_linear(a, M, omega, A, b, b_pto, rho=1025, g=9.81):
    m = M * omega**2 / (rho * g * A)
    return M * a**2 * omega**3 * b_pto / (
        2 * ((1 - m)**2 + m**2 * (b + b_pto)**2)
    )



b_pto_star = optimal_b_pto_linear(M, omega, A, b)
zeta_star = heave_amplitude_linear(a, M, omega, A, b, b_pto_star)
power_star = extracted_power_linear(a, M, omega, A, b, b_pto_star)

print("Analytical optimum for linear PTO:")
print(f"  b_pto*         = {b_pto_star:.6g}")
print(f"  Power*         = {power_star:.6g} W")
print(f"  |zeta0| at opt = {zeta_star:.6g} m")
