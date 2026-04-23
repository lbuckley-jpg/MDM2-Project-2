import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

'''--------solve cummins equation stepwise (latch)---------'''

def solve_cummins_stepwise_latch(body, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, B_pto, K_pto, t_span, dt=0.05):

    print('Initialising function: solve_cummins_stepwise_latch')

    M_eff = body.mass + A_heave_inf

    # full solution output stored in dictionary
    history = {
        't': [0.0],
        'x': [0.0],
        'v': [0.0],
        'F_ex': [F_ex_time(0.0)],
        'b_pto': [B_pto]
    }

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
            
            if t > t_arr[-1]:
                k_t = kernel[0]
                k_prev = k_vals[-1]
                memory += 0.5 * (k_prev * v_arr[-1] + k_t * v) * (t - t_arr[-1])
        dvdt = (F_ex_time(t) - memory - B_pto * v - (K_heave + K_pto) * x) / M_eff
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
        history['b_pto'].append(B_pto)

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


'''-------------------rk4 as solve_ivp is not great for cummins equation as the rhs is not self contained needs past states---------------------------'''

def rk4_step(rhs,t, x, v, h):
    dx1, dv1 = rhs(t, x, v)
    dx2, dv2 = rhs(t + 0.5*h, x + 0.5*h*dx1, v + 0.5*h*dv1)
    dx3, dv3 = rhs(t + 0.5*h, x + 0.5*h*dx2, v + 0.5*h*dv2)
    dx4, dv4 = rhs(t + h, x + h*dx3, v + h*dv3)
    x_new = x + (h/6.0) * (dx1 + 2*dx2 + 2*dx3 + dx4)
    v_new = v + (h/6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)
    return x_new, v_new


def solve_cummins_stepwise_no_latch_rl(rk4_solver, body, A_heave_inf, t_kernel, kernel,
                                        K_heave, F_ex_time, C_pto, K_pto, step_size,
                                        dt=0.05, history=None):

    hist_t = history['t']
    hist_v = history['v']
    hist_x = history['x']

    t_now  = hist_t[-1]
    t_final = t_now + step_size
    x_now  = float(hist_x[-1])
    v_now  = float(hist_v[-1])

    effective_mass = body.mass + A_heave_inf
    inv_mass = 1.0 / effective_mass
    t_kernel = np.asarray(t_kernel)
    kernel   = np.asarray(kernel)
    kernel_max_tau = t_kernel[-1]
    K_total = K_heave + K_pto

    def compute_memory(t):
        idx0 = np.searchsorted(hist_t, t - kernel_max_tau)
        t_memory = hist_t[idx0:]
        v_memory = hist_v[idx0:]
        tau    = t - t_memory
        k_vals = np.interp(tau, t_kernel, kernel, left=kernel[0], right=0.0)
        return float(np.trapezoid(k_vals * v_memory, t_memory))

    def rhs(t, x, v):
        dvdt = (F_ex_time(t) - compute_memory(t) - C_pto * v - K_total * x) * inv_mass
        dxdt = v
        return dxdt, dvdt

    # FIX 1: accumulate into lists, convert to arrays once at end

    # NEW - fixed
    new_t, new_v, new_x = [], [], []

    while t_now < t_final - 1e-10:
        h = min(dt, t_final - t_now)
        x_now, v_now = rk4_solver(rhs, t_now, x_now, v_now, h)
        t_now += h
        new_t.append(t_now)
        new_v.append(v_now)
        new_x.append(x_now)

    hist_t = np.concatenate([hist_t, new_t])
    hist_v = np.concatenate([hist_v, new_v])
    hist_x = np.concatenate([hist_x, new_x])

    # FIX 4: append F_ex for each new timestep
    history_out = {'t': hist_t, 'x': hist_x, 'v': hist_v}
    if history is not None and 'F_ex' in history:
        f_ex_list = list(history['F_ex'])
        for t in new_t[1:]:
            f_ex_list.append(float(F_ex_time(t)))
        history_out['F_ex'] = f_ex_list
    if history is not None and 'c_pto' in history:
        history_out['c_pto'] = list(history['c_pto'])

    return history_out

'''--------------analyse data--------------'''

def calc_power_absorbed(history, b_pto):

    print('Initialising function: calc_power_absorbed')

    v = np.array(history['v'][50:]) # remove transients

    # calculate the mean absorved power. not damping was added a force proportional to velcotiy
    # we know that power  = force x velocity 
    # this gives power = constant x velcoity x velocity
    p_inst =  b_pto * v ** 2 # power at each time step
    p_mean = np.mean(p_inst) # average power
    return p_inst, p_mean


'''-------------plot history-----------'''
def plot_history(history_latch, history_no_latch, f):
    print('Initialising function: plot_history')
    plt.title('Displacement Time graph for point absorber', fontsize=16)
    plt.plot(history_latch['t'], history_latch['x'], label='with latching')
    plt.plot(history_no_latch['t'], history_no_latch['x'], label='without latching')
    plt.plot(history_no_latch['t'], np.array(history_no_latch['F_ex']) / (np.max(np.array(history_no_latch['F_ex'])) / (max(history_no_latch['x']))), ls='--', label='scaled excitation force')
    plt.legend()
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Displacement (m) / Force (N)/100000', fontsize=14)
    plt.grid(True, alpha=0.25)
    plt.show()


'''------------plot power----------'''
def plot_power(history_latch, history_no_latch, p_inst_latch, p_inst_no_latch):
    print('Initialising function: plot_power')
    plt.title('Instaneous Power Plot', fontsize=16)
    plt.plot(history_latch['t'][50:], p_inst_latch, label='with latch')
    plt.plot(history_no_latch['t'][50:], p_inst_no_latch, label='without latch')
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Power (w)', fontsize=14)
    plt.legend()
    plt.show()



'''----------------save------------'''

def solve_cummins_stepwise_latch_limited(body, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, B_pto, K_pto, t_span, dt=0.05, pto_force_max=np.inf):

    print('Initialising function: solve_cummins_stepwise_latch_limited')

    M_eff = body.mass + A_heave_inf
    
    # physical limits
    max_displacement = body.radius
    rho = 1000.0
    Cd = 1.0
    A_cross = np.pi * max_displacement**2

    # full solution output stored in dictionary
    history = {
        't': [0.0],
        'x': [0.0],
        'v': [0.0],
        'F_ex': [F_ex_time(0.0)],
        'b_pto': [B_pto]
    }

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
            
            if t > t_arr[-1]:
                k_t = kernel[0]
                k_prev = k_vals[-1]
                memory += 0.5 * (k_prev * v_arr[-1] + k_t * v) * (t - t_arr[-1])
            
        hydrostatic_force = K_heave * np.clip(x, -max_displacement, max_displacement)
        viscous_drag = 0 #invicid
        
        # Linear PTO damping with force saturation (clipping)
        f_pto = np.clip(B_pto * v, -pto_force_max, pto_force_max)
            
        dvdt = (F_ex_time(t) - memory - f_pto - viscous_drag - hydrostatic_force - K_pto * x) / M_eff
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
        history['b_pto'].append(B_pto)

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

    print('simulating (stepwise with latching and physical limits)')

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

    return history




