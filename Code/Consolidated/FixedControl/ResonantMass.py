'''
RESONANT MASS for most common sea state
'''


import numpy as np
from scipy.interpolate import interp1d
import capytaine as cpt

# -----------------------------------------------------------------------------
# Constnt parameters
# -----------------------------------------------------------------------------

RADIUS = 2.0
RHO    = 1025.0
G      = 9.81

# sea state params for jonswap
HS     = 1.5                    # significant wave height [m]
TP     = 6.01                   # peak period [s] ~ SLOWER WAVE FREQ TO BE TUNED 
GAMMA  = 3.3                    # jonswap peak enhancement factor
SEED   = 42                     

OMEGA_P = 2.0 * np.pi / TP

# capytaine mesh resolution
MESH_RES   = (60, 60)
N_OMEGA_LO = 24                 # no. of low/high freq.s to solve for 
N_OMEGA_HI = 6                  # A(ω), B(ω), Fex(ω)

# start with buoy half submerged
C    = RHO * G * np.pi * RADIUS**2
m_eq = (2.0 / 3.0) * np.pi * RADIUS**3 * RHO

print(f"Sea state: Hs = {HS:.2f} m,  Tp = {TP:.2f} s (omega_p = "
      f"{OMEGA_P:.3f} rad/s),  gamma = {GAMMA}")
print(f"Buoy:      r  = {RADIUS} m,  C = {C:,.1f} N/m,  "
      f"m_equilibrium = {m_eq:,.1f} kg")

# -----------------------------------------------------------------------------
# Capytaine BEM for A(w), B(w), Fex(w)
# -----------------------------------------------------------------------------
mesh = cpt.mesh_sphere(radius=RADIUS, center=(0, 0, 0),
                       resolution=MESH_RES).immersed_part()
body = cpt.FloatingBody(mesh=mesh, name="sphere_r2")
body.add_translation_dof(name="Heave")


'''AI here to solve cpt: MUST CHECK!!!'''
# USES SAME METHOD AS OTHER ADAPTIVE CONTROL IMPLEMENTATION OF CPT

# Frequency grid covers the entire JONSWAP support + some high-frequency
# bandwidth so that K(t) and A(inf) are well approximated.
omega_grid = np.concatenate([np.linspace(0.15, 3.2, N_OMEGA_LO),
                             np.linspace(3.5, 9.0, N_OMEGA_HI)])

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
A_inf = A[-1]

# Interpolators over the solved grid
A_of   = interp1d(omega_grid, A,               bounds_error=False,
                  fill_value=(A[0],   A[-1]))
B_of   = interp1d(omega_grid, B,               bounds_error=False,
                  fill_value=(B[0],   B[-1]))
Fmag_of = interp1d(omega_grid, np.abs(Fex),    bounds_error=False,
                   fill_value=(np.abs(Fex)[0], np.abs(Fex)[-1]))
Fphs_of = interp1d(omega_grid, np.unwrap(np.angle(Fex)),
                   bounds_error=False, fill_value=0.0)

# -----------------------------------------------------------------------------
# Tune buoy mass to resonate at JONSWAP peak frequency
#     omega_p^2 (m + A(omega_p)) = C   =>   m = C/omega_p^2 - A(omega_p)
# -----------------------------------------------------------------------------
A_wp = float(A_of(OMEGA_P))
B_wp = float(B_of(OMEGA_P))
M_r  = C / OMEGA_P**2
m    = M_r - A_wp
if m <= 0:
    raise RuntimeError("Resonant mass non-physical; choose a longer Tp.")
print(f"\nResonance tuning at omega_p: A(wp) = {A_wp:,.1f} kg, "
      f"m = {m:,.1f} kg ({m/m_eq:.2f} * m_eq)")