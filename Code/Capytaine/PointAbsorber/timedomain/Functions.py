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

def jonswap_frequency_amplitudes(omega, delta_omega, Hs = 2.0, Tp=8.0 , gamma=3.3):
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

def get_cummins_components(body, capytaine_dataset, wave_direction, wave_amplitudes, omegas):
    print('Initialising function: get_cummins_components')

    # get the added masses for heave motion
    A_heave = capytaine_dataset['added_mass'].sel(radiating_dof='Heave', influenced_dof='Heave').values

    # approximate the added mass at infinite frequency as the added mass at the highest frequency
    A_heave_inf = float(A_heave[-1])

    # get the radiation damping coefficient for heave motion
    B_heave = capytaine_dataset['radiation_damping'].sel(radiating_dof='Heave', influenced_dof='Heave').values

    # get the complex (frequency domain) excitation force
    F_ex_complex = (capytaine_dataset['Froude_Krylov_force'] + capytaine_dataset['diffraction_force']).sel(influenced_dof='Heave', wave_direction=wave_direction).values

    # convert to a function of time
    def F_ex_time(t):
        return np.sum(np.abs(F_ex_complex) * wave_amplitudes * np.cos(omegas * t + np.angle(F_ex_complex)))

    # derivative for latching controls

    def F_ex_time_dot(t):
        amplitudes = np.abs(F_ex_complex) * wave_amplitudes
        phases = np.angle(F_ex_complex)
        return np.sum(-amplitudes * omegas * np.sin(omegas * t + phases))

    # get the hydrostatic stiffness coefficient for heave
    K_heave = float(body.hydrostatic_stiffness.sel(influenced_dof='Heave', radiating_dof='Heave'))

    # build the memory kernel for 0 to 60 seconds after which effects are negligible
    t_kernel = np.linspace(0, 60, 1000)

    kernel = np.array([(2 / np.pi) * np.trapezoid(B_heave * np.cos(omegas * ti), omegas) for ti in t_kernel]) # solve for each time and frequence

    return A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot



'''--------solve cummins equation---------'''

def solve_cummins_equation_latch(body, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, C_pto, K_pto, t_span):

    # create a dictionary's and list to store data in
    solution_history = {'t': [],'x': [], 'v': []}

    memory_history = {'t':[], 'v':[]}

    t_final = t_span[1]
    t_last = 0.0
    v_last = 0.0
    x_last = 0.0

    # create a function to store the data after each solution of solve_ivp
    def append_segment(sol):
        nonlocal t_last, x_last, v_last
        start_idx = 0 if not solution_history['t'] else 1 # if first segment add all of sol else add after first t as that is already added from previous segment
        solution_history['t'].extend(sol.t[start_idx:].tolist())
        solution_history['x'].extend(sol.y[0][start_idx:].tolist())
        solution_history['v'].extend(sol.y[1][start_idx:].tolist())

        t_last = sol.t[-1]
        x_last = sol.y[0][-1]
        v_last = sol.y[1][-1]

    # define the rhs function for solve_ivp
    def rhs_unlatched(t, state):

        # unpack the state
        x, v = state

        # store the state history in during each function run so that integral can by calculated
        memory_history['t'].append(t)
        memory_history['x'].append(x)
        memory_history['v'].append(v)

        # get the history
        t_hist = np.array(memory_history['t'])
        v_hist = np.array(memory_history['v'])

        # calculate the convolution integral
        if len(t_hist) > 1:
            t_shifted = t - t_hist
            kernel_values = np.interp(t_shifted, t_kernel, kernel, left=kernel[0], right = 0.0) # interpolate our pre calculate kernel values
            memory = np.trapezoid(kernel_values * v_hist, t_hist) # numerically integrate 
        else:
            memory = 0.0

        # define the differential equation

        dvdt = (F_ex_time(t) - memory - C_pto*v - (K_heave + K_pto) * x) / (body.mass + A_heave_inf)

        return [v, dvdt]

    # define the rhs function for latched phase
    def rhs_latched(t, state):
        return [0,0]

    # define the latch event
    def latch_event(t, state):
        x, v = state
        return v

    # define the unlatch event
    def unlatch_event(t, state):
        return F_ex_time_dot(t)

    # simulate 
    print('simulating')

    '''------------ Initial unlatched phase to move off v = 0 -----------'''

    sol = solve_ivp(rhs_unlatched,[t_last, 1e-2], [x_last, v_last], max_step=0.5)
    append_segment(sol)

    while t_last < t_final:
        

        '''------------------------------unlatched phase ---------------------------'''

        # set the event detection
        latch_event.terminal = True
        latch_event.direction = -1
        sol = solve_ivp(rhs_unlatched,[t_last, t_final], [x_last, v_last], events = latch_event, max_step=0.5)
        append_segment(sol)

        print(f'Latch event detected at t = {t_last}')

        '''-----------------------------latched phase---------------------------------'''

        # set the event detection
        unlatch_event.terminal = True
        unlatch_event.direction = -1
        sol = solve_ivp(rhs_latched,[t_last, t_final], [x_last, v_last], events = unlatch_event, max_step=0.5)
        append_segment(sol)

        print(f'unlatch event detected at t = {t_last}')

        '''------------------------------unlatched phase ---------------------------'''

        # set the event detection
        latch_event.terminal = True
        latch_event.direction = 1
        sol = solve_ivp(rhs_unlatched,[t_last, t_final], [x_last, v_last], events = latch_event, max_step=0.5)
        append_segment(sol)

        print(f'latch event detected at t = {t_last}')

        '''-----------------------------latched phase---------------------------------'''

        # set the event detection
        unlatch_event.terminal = True
        unlatch_event.direction = -1
        sol = solve_ivp(rhs_latched,[t_last, t_final], [x_last, v_last], events = unlatch_event, max_step=0.5)
        append_segment(sol)
        print(f'unlatch event detected at t = {t_last}')       

    return solution_history

'''------------------------solve cummins equation no latch--------------------'''

def solve_cummins_equation_no_latch(body, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, C_pto, K_pto, t_span):

    # create a dictionary's and list to store data in
    solution_history = {'t': [],'x': [], 'v': []}

    memory_history = {'t':[], 'v':[]}

    t_final = t_span[1]
    t_last = 0.0
    v_last = 0.0
    x_last = 0.0

    # create a function to store the data after each solution of solve_ivp
    def append_segment(sol):
        nonlocal t_last, x_last, v_last
        start_idx = 0 if not solution_history['t'] else 1 # if first segment add all of sol else add after first t as that is already added from previous segment
        solution_history['t'].extend(sol.t[start_idx:].tolist())
        solution_history['x'].extend(sol.y[0][start_idx:].tolist())
        solution_history['v'].extend(sol.y[1][start_idx:].tolist())

        t_last = sol.t[-1]
        x_last = sol.y[0][-1]
        v_last = sol.y[1][-1]

    # define the rhs function for solve_ivp
    def rhs_unlatched(t, state):

        # unpack the state
        x, v = state

        # store the state history in during each function run so that integral can by calculated
        memory_history['t'].append(t)
        memory_history['x'].append(x)
        memory_history['v'].append(v)

        # get the history
        t_hist = np.array(memory_history['t'])
        v_hist = np.array(memory_history['v'])

        # calculate the convolution integral
        if len(t_hist) > 1:
            t_shifted = t - t_hist
            kernel_values = np.interp(t_shifted, t_kernel, kernel, left=kernel[0], right = 0.0) # interpolate our pre calculate kernel values
            memory = np.trapezoid(kernel_values * v_hist, t_hist) # numerically integrate 
        else:
            memory = 0.0

        # define the differential equation

        dvdt = (F_ex_time(t) - memory - C_pto*v - (K_heave + K_pto) * x) / (body.mass + A_heave_inf)

        return [v, dvdt]

    # simulate 
    print('simulating')

    '''------------ Initial unlatched phase to move off v = 0 -----------'''

    solve_ivp(rhs_unlatched,[t_last, t_final], [x_last, v_last], max_step=0.5)

    return solution_history


'''--------solve cummins equation stepwise (latch)---------'''

def solve_cummins_stepwise_latch(body, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, C_pto, K_pto, t_span, dt=0.05):

    print('Initialising function: solve_cummins_stepwise_latch')

    M_eff = body.mass + A_heave_inf

    # full solution output stored in dictionary
    history = {'t': [0.0], 'x': [0.0], 'v': [0.0], 'F_ex': [F_ex_time(0.0)]}

    t_final = t_span[1]
    t_now = 0.0
    x_now = 0.0
    v_now = 0.0
    latched = False # is sytem latched or not
    latch_armed = False
    latch_velocity_tol = 1e-3  # |v| must exceed this before latch detection activates used for the start of simulation to avoid immediate latch

    # a rhs function that is used for solve ivp
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
        dvdt = (F_ex_time(t) - memory - C_pto * v - (K_heave + K_pto) * x) / M_eff
        return [v, dvdt]

    # detects zero velocity of the buoy ie the latch event
    def latch_event(t, state):
        return state[1]
    
    latch_event.terminal = True
    latch_event.direction = 0 # detects both maxima and minima

    # function that records the values
    def record(t_val, x_val, v_val):
        history['t'].append(t_val)
        history['x'].append(x_val)
        history['v'].append(v_val)
        history['F_ex'].append(F_ex_time(t_val))

    # function that sweeps a time interval looking for the unlatch criteria and then returns the root where F = 0
    def find_unlatch_time(t_start, t_end, x_latched): # this function calculates the time when the force is maximal and opposite to the latched position
        t_a = t_start # t_a is lower bound of time interval

        while t_a < t_end - 1e-10: # whilst interval is larger than 1e-10 (tolerance)
            t_b = min(t_a + dt, t_end) # t_b is the upper bound of time interval

            fa, fb = F_ex_time_dot(t_a), F_ex_time_dot(t_b) # calculate the force at end of interval

            # logic to find only the root we are interested in
            if x_latched > 0:
                sign_change = (fa <= 0 and fb > 0) # if x is positive then we want force minima
            else:
                sign_change = (fa >= 0 and fb < 0) # if x is negative then we want force maxima

            if sign_change: # when we find the interval return 
                return brentq(F_ex_time_dot, t_a, t_b) # find the root in the time interval
            t_a = t_b # update the lower interval as the upper interval 


    print('simulating (stepwise with latching)')

    # loop thorugh the times until t_final minus tolerance
    while t_now < t_final - 1e-10:

        # ---------------------------unlatched phase-------------------------------
        if not latched:

            t_next = min(t_now + dt, t_final) # calc how long to run sim for

            events = latch_event if latch_armed else None 

            sol = solve_ivp(rhs, [t_now, t_next], [x_now, v_now], events=events, max_step=dt) # solve ivp for the macro time step

            for i in range(1, len(sol.t)): # save all the data
                record(sol.t[i], sol.y[0, i], sol.y[1, i])

            # update the current time
            t_now = sol.t[-1]
            x_now = sol.y[0, -1]
            v_now = sol.y[1, -1]

            if not latch_armed and abs(v_now) > latch_velocity_tol: 
                latch_armed = True

            event_fired = (events is not None and sol.t_events is not None and len(sol.t_events[0]) > 0)

            if event_fired:
                latched = True 
                latch_armed = False 
                v_now = 0.0
                # print(f'Latched at t = {t_now:.3f}, x = {x_now:.4f}')

        # -----------------------------latched phase--------------------------------------
        else:

            # find the first unlatch time from current time to end of sim
            t_unlatch = find_unlatch_time(t_now, t_final, x_now)

            # check that there is a latch between now and the end of the sim
            t_end_latch = t_unlatch if t_unlatch is not None else t_final

            while t_now < t_end_latch - 1e-10:
                t_step = min(t_now + dt, t_end_latch) # step
                record(t_step, x_now, 0.0)
                t_now = t_step

            if t_unlatch is not None:
                latched = False
                v_now = 0.0
                # print(f'Unlatched at t = {t_now:.3f}')

    return history


'''--------solve cummins equation stepwise (no latch)---------'''

def solve_cummins_stepwise_no_latch(body, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, C_pto, K_pto, t_span, dt=0.05):

    print('Initialising function: solve_cummins_stepwise_no_latch')

    t_final = t_span[1]
    M_eff = body.mass + A_heave_inf

    history = {'t': [0.0], 'x': [0.0], 'v': [0.0], 'F_ex': [F_ex_time(0.0)]}

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
        dvdt = (F_ex_time(t) - memory - C_pto * v - (K_heave + K_pto) * x) / M_eff
        return [v, dvdt]

    print('simulating (stepwise without latching)')

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

    return history


'''--------------analyse data--------------'''

def calc_power_absorbed(history, c_pto):

    print('Initialising function: calc_power_absorbed')

    v = np.array(history['v'][50:]) # remove transients

    # calculate the mean absorved power. not damping was added a force proportional to velcotiy
    # we know that power  = force x velocity 
    # this gives power = constant x velcoity x velocity
    p_inst =  c_pto * v ** 2 # power at each time step
    p_mean = np.mean(p_inst) # average power
    return p_inst, p_mean


'''-------------plot history-----------'''

def plot_history(history_latch, history_no_latch, f):
    print('Initialising function: plot_history')
    plt.title('Displacement Time graph for point absorber')
    plt.plot(history_latch['t'], history_latch['x'], label='with latching')
    plt.plot(history_no_latch['t'], history_no_latch['x'], label='without latching')
    plt.plot(history_no_latch['t'], np.array(history_no_latch['F_ex']) / 1e5 , ls='--', label='excitation force')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m) / Force (N)/100000')
    plt.grid(True, alpha=0.25)
    plt.show()


'''------------plot power----------'''

def plot_power(history_latch, history_no_latch, p_inst_latch, p_inst_no_latch):
    print('Initialising function: plot_power')
    plt.title('Instaneous Power Plot')
    plt.plot(history_latch['t'][50:], p_inst_latch, label='with latch')
    plt.plot(history_no_latch['t'][50:], p_inst_no_latch, label='without latch')
    plt.xlabel('Time (s)')
    plt.ylabel('Power (w)')
    plt.legend()
    plt.show()


'''----------------save------------'''




