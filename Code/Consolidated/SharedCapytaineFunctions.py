'''imports'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import os
import time
import capytaine as cpt

'''-----------generate frequencies---------------'''

def generate_frequencies(N=40, Tp=8.0):
    """
    Generate N evenly spaced frequencies centred around the peak frequency.
    Parameters
    N  : int    number of frequency components
    Tp : float  peak wave period [s]
    """
    wp   = 2 * np.pi / Tp
    w0   = 0.5 * wp
    wmax = 4.0 * wp

    omega, delta_omega = np.linspace(w0, wmax, N, retstep=True)
    return omega, delta_omega


'''------------generate jonswap amplitudes-----------'''

def jonswap_frequency_amplitudes(omega, delta_omega, Hs = 2.0, Tp=12.0 , gamma=3.3):
    """
    JONSWAP spectral density S(omega) [m^2 s / rad]
    Returns wave amplitude for each frequency component.

    Parameters
    ----------
    omega       : np.ndarray  angular frequencies [rad/s]
    delta_omega : float       frequency resolution [rad/s]
    Hs          : float       significant wave height [m]
    Tp          : float       peak period [s]
    gamma       : float       peak enhancement factor (default 3.3)
    """
    omega_p = 2 * np.pi / Tp
    alpha   = 0.0624 / (0.230 + 0.0336 * gamma - 0.185 / (1.9 + gamma))

    # Pierson-Moskowitz base spectrum
    S_pm = (alpha * 9.81**2 / omega**5) * np.exp(-1.25 * (omega_p / omega)**4)

    # JONSWAP peak enhancement
    sigma = np.where(omega <= omega_p, 0.07, 0.09)
    r     = np.exp(-((omega - omega_p)**2) / (2 * sigma**2 * omega_p**2))
    S     = S_pm * gamma**r

    # Scale to match desired Hs  (Hs = 4 * sqrt(m0))
    m0 = np.trapezoid(S, omega)
    S *= (Hs / 4)**2 / m0

    return np.sqrt(2 * S * delta_omega)  # wave amplitude per component [m]


'''-----------generate buoy----------'''

def generate_buoy(radius=5, mass=5000):
    print('Initialising function: generate_buoy')

    # create buoy mesh
    buoy_mesh = cpt.mesh_sphere(radius=radius, center=(0.0,0.0,0.0), resolution=(30,30))

    # set center of mass
    rotation_center = (0.0,0.0,0.0)

    # add a internal lid at the water level this is needed when using the solve over immersed part method
    lid_mesh = buoy_mesh.generate_lid(z=0.0)

    # create buoy object
    buoy = cpt.FloatingBody(mesh = buoy_mesh, lid_mesh=lid_mesh, dofs=cpt.rigid_body_dofs(rotation_center=rotation_center), center_of_mass = rotation_center, mass=mass, name='Point Absorber')

    # store the buoy radius for logging data

    buoy.radius = radius

    # create the intertia mass matrix
    buoy.intertia_matrix = buoy.compute_rigid_body_inertia()

    # calculate the hydrostatic stiffness of the bouy
    buoy.hydrostatic_stiffness = buoy.immersed_part().compute_hydrostatic_stiffness()

    return buoy


'''----------solve with capytaine------'''

def solve_with_capytaine(body, omegas, wave_direction=np.pi, water_depth=np.inf, water_density=1000):
    print('Initialising function: solve_with_capytaine')

    # instantiate the solver
    bem_solver = cpt.BEMSolver()
    cpt.set_logging('WARNING')

    # creat the radiation problems
    radiation_problems = [cpt.RadiationProblem(omega=w, body=body.immersed_part(), radiating_dof=dof, water_depth=water_depth, rho=water_density) for w in omegas for dof in body.dofs]

    # create diffraction problems
    diffraction_problems = [cpt.DiffractionProblem(omega=w, body=body.immersed_part(), wave_direction=wave_direction, water_depth=water_depth, rho=water_density) for w in omegas]

    # solve the problems
    radiation_results   = bem_solver.solve_all(radiation_problems)
    diffraction_results = [bem_solver.solve(p) for p in diffraction_problems]

    # add results to a data set
    capytaine_dataset = cpt.assemble_dataset(radiation_results + diffraction_results)

    return capytaine_dataset 



'''---------get cummins components from capytaine data----------'''

def get_cummins_components(body, capytaine_dataset, wave_direction, wave_amplitudes, omegas, seed):
    print('Initialising function: get_cummins_components')

    # get the added masses for heave motion
    A_heave = capytaine_dataset['added_mass'].sel(radiating_dof='Heave', influenced_dof='Heave').values

    # approximate the added mass at infinite frequency as the added mass at the highest frequency
    A_heave_inf = float(A_heave[-1])

    # get the radiation damping coefficient for heave motion
    B_heave = capytaine_dataset['radiation_damping'].sel(radiating_dof='Heave', influenced_dof='Heave').values

    # get the complex (frequency domain) excitation force
    F_ex_complex = (capytaine_dataset['Froude_Krylov_force'] + capytaine_dataset['diffraction_force']).sel(influenced_dof='Heave', wave_direction=wave_direction).values

    # add random phases to wave components
    rng = np.random.default_rng(seed)
    epsilon = rng.uniform(0, 2 * np.pi, size=omegas.shape)
    hydro_phase = np.angle(F_ex_complex)
    total_phase = hydro_phase + epsilon

    # convert to a function of time
    def F_ex_time(t):
        return np.sum(np.abs(F_ex_complex) * wave_amplitudes * np.cos(omegas * t + total_phase))

    # derivative for latching controls

    def F_ex_time_dot(t):
        amplitudes = np.abs(F_ex_complex) * wave_amplitudes
        return np.sum(-amplitudes * omegas * np.sin(omegas * t + total_phase))

    # get the hydrostatic stiffness coefficient for heave
    K_heave = float(body.hydrostatic_stiffness.sel(influenced_dof='Heave', radiating_dof='Heave'))

    # build the memory kernel for 0 to 60 seconds after which effects are negligible
    t_kernel = np.linspace(0, 60, 1000)

    kernel = np.array([(2 / np.pi) * np.trapezoid(B_heave * np.cos(omegas * ti), omegas) for ti in t_kernel]) # solve for each time and frequence

    return A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, B_heave


import numpy as np
from scipy.stats import linregress

# def get_cummins_components(body, capytaine_dataset, wave_direction, wave_amplitudes, omegas, seed):
#     print('Initialising function: get_cummins_components')

#     # get frequency-dependent added mass and damping
#     A_heave = capytaine_dataset['added_mass'].sel(radiating_dof='Heave', influenced_dof='Heave').values
#     B_heave = capytaine_dataset['radiation_damping'].sel(radiating_dof='Heave', influenced_dof='Heave').values

#     # xcALCULATE A_inf via Linear Regression
#     # We use the high-frequency tail where A(w) is linear relative to 1/w^2
#     # Typically, the last 10-20% of the frequency range is used.
#     tail_idx = int(len(omegas) * 0.8) 
#     w_tail = omegas[tail_idx:]
#     A_tail = A_heave[tail_idx:]
    
#     # Filter out any zero frequencies to avoid division errors
#     valid = w_tail > 0
#     inv_w_sq = 1.0 / (w_tail[valid]**2)
    
#     # Perform linear regression: A(w) = A_inf + C * (1/w^2)
#     # slope = C, intercept = A_inf
#     slope, A_heave_inf, r_value, p_value, std_err = linregress(inv_w_sq, A_tail[valid])
    
#     print(f"Calculated A_inf: {A_heave_inf:.4f} (R-squared: {r_value**2:.4f})")

#     # Excitation Force setup
#     F_ex_complex = (capytaine_dataset['Froude_Krylov_force'] + capytaine_dataset['diffraction_force']).sel(influenced_dof='Heave', wave_direction=wave_direction).values

#     rng = np.random.default_rng(seed)
#     epsilon = rng.uniform(0, 2 * np.pi, size=omegas.shape)
#     hydro_phase = np.angle(F_ex_complex)
#     total_phase = hydro_phase + epsilon
#     amplitudes = np.abs(F_ex_complex) * wave_amplitudes

#     def F_ex_time(t):
#         return np.sum(amplitudes * np.cos(omegas * t + total_phase))

#     def F_ex_time_dot(t):
#         return np.sum(-amplitudes * omegas * np.sin(omegas * t + total_phase))

#     # 4. Hydrostatic Stiffness
#     K_heave = float(body.hydrostatic_stiffness.sel(influenced_dof='Heave', radiating_dof='Heave'))

#     # 5. Memory Kernel Calculation (Optimized with Matrix Multiply)
#     t_kernel = np.linspace(0, 60, 1000)
#     # Pre-compute the cosine matrix for speed: shape (len(t_kernel), len(omegas))
#     cos_matrix = np.cos(np.outer(t_kernel, omegas))
#     # Integrate using trapezoidal rule across the frequency axis
#     kernel = (2 / np.pi) * np.trapezoid(B_heave * cos_matrix, omegas, axis=1)

#     return A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot

def solve_cummins_stepwise_no_control(body, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, C_pto, K_pto, t_span, dt=0.05):

    print('Initialising function: solve_cummins_stepwise_no_control')

    t_final = t_span[1]
    M_eff = body.mass + A_heave_inf

    history = {
        't': [0.0],
        'x': [0.0],
        'v': [0.0],
        'F_ex': [F_ex_time(0.0)],
        'c_pto': [C_pto]
    }

    t_final = t_span[1]
    t_now = 0.0
    x_now = 0.0
    v_now = 0.0

    def rhs(t, state):
        x, v = state
        t_arr = np.array(history['t'])
        v_arr = np.array(history['v'])
        if len(t_arr) < 2:
            memory = 0.0
        else:
            tau = t - t_arr
            k_vals = np.interp(tau, t_kernel, kernel, left=kernel[0], right=0.0)
            memory = np.trapezoid(k_vals * v_arr, t_arr)
            
            # Correct for current step to prevent lagging damping in high-accel events
            if t > t_arr[-1]:
                k_t = kernel[0]
                k_prev = k_vals[-1]
                memory += 0.5 * (k_prev * v_arr[-1] + k_t * v) * (t - t_arr[-1])
        dvdt = (F_ex_time(t) - memory - C_pto * v - (K_heave + K_pto) * x) / M_eff
        return [v, dvdt]

    while t_now < t_final - 1e-10:
        t_next = min(t_now + dt, t_final)
        sol = solve_ivp(rhs, [t_now, t_next], [x_now, v_now], max_step=dt)

        t_now = sol.t[-1]
        x_now = sol.y[0, -1]
        v_now = sol.y[1, -1]

        history['t'].append(t_now)
        history['x'].append(x_now)
        history['v'].append(v_now)
        history['F_ex'].append(F_ex_time(t_now))
        history['c_pto'].append(C_pto)
    return history


def solve_cummins_stepwise_no_control_limited(body, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, C_pto, K_pto, t_span, dt=0.05):

    print('Initialising function: solve_cummins_stepwise_no_control_limited')

    t_final = t_span[1]
    M_eff = body.mass + A_heave_inf
    
    # physical limits
    max_displacement = body.radius
    rho = 1000.0
    Cd = 1.0
    A_cross = np.pi * max_displacement**2

    history = {
        't': [0.0],
        'x': [0.0],
        'v': [0.0],
        'F_ex': [F_ex_time(0.0)],
        'c_pto': [C_pto]
    }

    t_final = t_span[1]
    t_now = 0.0
    x_now = 0.0
    v_now = 0.0

    def rhs(t, state):
        x, v = state
        t_arr = np.array(history['t'])
        v_arr = np.array(history['v'])
        if len(t_arr) < 2:
            memory = 0.0
        else:
            tau = t - t_arr
            k_vals = np.interp(tau, t_kernel, kernel, left=kernel[0], right=0.0)
            memory = np.trapezoid(k_vals * v_arr, t_arr)
            
            # Correct for current step to prevent lagging damping in high-accel events
            if t > t_arr[-1]:
                k_t = kernel[0]
                k_prev = k_vals[-1]
                memory += 0.5 * (k_prev * v_arr[-1] + k_t * v) * (t - t_arr[-1])
            
        hydrostatic_force = K_heave * np.clip(x, -max_displacement, max_displacement)
        viscous_drag = 0.5 * rho * Cd * A_cross * abs(v) * v
        
        dvdt = (F_ex_time(t) - memory - C_pto * v - viscous_drag - hydrostatic_force - K_pto * x) / M_eff
        return [v, dvdt]

    while t_now < t_final - 1e-10:
        t_next = min(t_now + dt, t_final)
        sol = solve_ivp(rhs, [t_now, t_next], [x_now, v_now], max_step=dt)

        t_now = sol.t[-1]
        x_now = sol.y[0, -1]
        v_now = sol.y[1, -1]

        history['t'].append(t_now)
        history['x'].append(x_now)
        history['v'].append(v_now)
        history['F_ex'].append(F_ex_time(t_now))
        history['c_pto'].append(C_pto)
    return history


def calc_power_absorbed(history):

    print('Initialising function: calc_power_absorbed')

    t = np.array(history['t'][50:])
    v = np.array(history['v'][50:]) # remove transients
    c_pto = np.array(history['c_pto'][50:])
    # calculate the mean absorved power. not damping was added a force proportional to velcotiy
    # we know that power  = force x velocity 
    # this gives power = constant x velcoity x velocity
    p_inst =  c_pto * v ** 2 # power at each time step
    
    # average power must be time-weighted via integral to account for variable-step dense ivp solving during unlatching
    duration = t[-1] - t[0]
    p_mean = np.trapezoid(p_inst, t) / duration if duration > 0 else 0.0

    return p_inst, p_mean


