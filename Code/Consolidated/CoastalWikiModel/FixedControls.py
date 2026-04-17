# version rewritten, checked and editted to contain no AI

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
m_peak       = 1.0 / (1.0 + b**2)

# find resonant masses
M = (rho * g * A * m_peak) / omega**2

print(f'\nresonant mass {M:.2f} kg')

# =====================================
# FIXED CONTROL 1: Coulomb pto
#
# F_PTO = C_PTO sign(∂ζ/∂t)
#
# =====================================

def solve_zeta_amplitude(a, M, omega, A, b, C_PTO, rho=1025.0, g=9.81,
                         tol=1e-10, max_iter=200):

    '''
    Solve for the amplitude of the buoys motion
    '''

    if C_PTO < 0:
        raise ValueError("C_PTO must be non-negative.")

    m = M * omega**2 / (rho * g * A)

    # Undamped / no-PTO starting guess
    z = a / np.sqrt((1 - m)**2 + (m * b)**2)

    # Fixed-point iteration
    for i in range(max_iter):
        z_old = z
        z = a / (np.sqrt((1 - m)**2 + (m * b + (4 * C_PTO) / (np.pi * rho * g * A * z_old))**2))

        if abs(z-z_old) < tol:
            return z

    return z  # return last iteration if not fully converged


def extracted_power(zeta_abs, omega, C_PTO):
    return (2.0 / np.pi) * C_PTO * omega * zeta_abs


def grid_search(a, M, omega, A, b,
                      rho=1025.0, g=9.81,
                      C_min=0.0, C_max=2e6, n_grid=5000):
    """
    Brute-force grid search over C_PTO to maximise extracted power only.
    """
    C_grid = np.linspace(C_min, C_max, n_grid)
    zeta_grid = np.empty_like(C_grid)
    power_grid = np.empty_like(C_grid)

    for i, C in enumerate(C_grid):
        zeta_abs = solve_zeta_amplitude(a, M, omega, A, b, C, rho, g)
        power = extracted_power(zeta_abs, omega, C)

        zeta_grid[i] = zeta_abs
        power_grid[i] = power

    idx_power = np.argmax(power_grid)

    return {
        "C_grid": C_grid,
        "zeta_grid": zeta_grid,
        "power_grid": power_grid,
        "power_opt": {
            "C_PTO_star": C_grid[idx_power],
            "power_star": power_grid[idx_power],
            "zeta_abs_at_power_opt": zeta_grid[idx_power]
        }
    }

# -----------------------
# Example usage
# -----------------------


results = grid_search(
    a=a,
    M=M,
    omega=omega,
    A=A,
    b=b,
    C_min=0.0,
    C_max=2e6,
    n_grid=10000
)

print("\nOptimal C_PTO for maximum extracted power:")
print(f"  C_PTO*         = {results['power_opt']['C_PTO_star']:.6g} N")
print(f"  Power*         = {results['power_opt']['power_star']:.6g} W")
print(f"  |zeta0| at opt = {results['power_opt']['zeta_abs_at_power_opt']:.6g} m")


# =====================================
# FIXED CONTROL 2: Spring pto
# 
# Find mass such that power maximised
# =====================================


def optimal_b_pto_linear(M, omega, A, b, rho=1025.0, g=9.81):
    m = M * omega**2 / (rho * g * A)
    return np.sqrt(b**2 + ((1.0 - m) / m)**2)


def heave_amplitude_linear(a, M, omega, A, b, b_pto, rho=1025.0, g=9.81):
    m = M * omega**2 / (rho * g * A)
    return a / np.sqrt((1.0 - m)**2 + m**2 * (b + b_pto)**2)


def extracted_power_linear(a, M, omega, A, b, b_pto, rho=1025.0, g=9.81):
    m = M * omega**2 / (rho * g * A)
    return M * a**2 * omega**3 * b_pto / (
        2.0 * ((1.0 - m)**2 + m**2 * (b + b_pto)**2)
    )




b_pto_star = optimal_b_pto_linear(M, omega, A, b)
zeta_star = heave_amplitude_linear(a, M, omega, A, b, b_pto_star)
power_star = extracted_power_linear(a, M, omega, A, b, b_pto_star)

print("Analytical optimum for linear PTO:")
print(f"  b_pto*         = {b_pto_star:.6g}")
print(f"  Power*         = {power_star:.6g} W")
print(f"  |zeta0| at opt = {zeta_star:.6g} m")

