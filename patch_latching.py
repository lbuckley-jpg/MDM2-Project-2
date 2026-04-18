import os

file_path = r"C:\Users\edwar\OneDrive - University of Bristol\Year 2\Project 2\MDM2-Project-2\Code\Consolidated\Latching\LatchingFunctions.py"

latch_limited_code = """

def solve_cummins_stepwise_latch_limited(body, A_heave_inf, t_kernel, kernel, K_heave, F_ex_time, F_ex_time_dot, C_pto, K_pto, t_span, dt=0.05):

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
        'c_pto': [C_pto]
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
            
        hydrostatic_force = K_heave * np.clip(x, -max_displacement, max_displacement)
        viscous_drag = 0.5 * rho * Cd * A_cross * abs(v) * v
            
        dvdt = (F_ex_time(t) - memory - C_pto * v - viscous_drag - hydrostatic_force - K_pto * x) / M_eff
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
        history['c_pto'].append(C_pto)

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
"""

with open(file_path, "a") as f:
    f.write(latch_limited_code)

print("Patch applied to LatchingFunctions.py")
