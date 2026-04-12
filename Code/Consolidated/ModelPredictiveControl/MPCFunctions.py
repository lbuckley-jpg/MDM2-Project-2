# Real time model predictive control framework
# These functions are designed to be augmented with the output of a suitable
# Kalman filter and AutoRegressive Excitation force predicting model
# However known excitation forces could be used for demonstration sake.

# The energy absorption maximisation probelem is framed as a
# convex optimisation of a convex optimisation function
# I researched a convex function and it basically means that between to points
# the straight line (line segment) that you draw between lies entirely above the graph between the two points.

# To solve the optimisation problem I have used a python library to do the heavy lifting for us

# whats happening at a high level:
# (1) the state of the buoy is measured
# (2) the full state is estimate with kalmans filter if some measurements missing
# (3) predict excitation forces using AR
# (4) MPC solves optimisation of control over time horizon
# (5) apply optimised control to cummins model and evaluate

import numpy as np
import scipy.linalg as la # for matrix solving etx
from scipy.integrate import solve_ivp
import cvxpy as cp # module for optimisation
import matplotlib.pyplot as plt
dt = 0.1
N = 20
u_max = 1500


def discrete_matrices_for_mpc_from_prony(prony_coeffs, mass, added_mass_inf, C_pto, K_heave, dt_sample):
    """
    Build discrete-time (Ad, Bd) for the same linear heave + Prony radiation model as
    prony_coeffs: (n_terms, 4) columns [alpha_r, beta_r, alpha_i, beta_i].
    """
    M = float(mass + added_mass_inf) # total mass buoy moves
    n_terms = int(prony_coeffs.shape[0]) # number of pront coefficients 
    n_state = 2 + 2 * n_terms 
    Ac = np.zeros((n_state, n_state)) # matrix derived from cummins equation. This is in the continous time space. contains natural hydrodynamics of the buoy
    Bc = np.zeros((n_state, 1)) # second matrix that contain terms for the external forces ie excitation and control

    # add the entries to the matrix
    Ac[0, 1] = 1.0
    Ac[1, 0] = -float(K_heave) / M
    Ac[1, 1] = -float(C_pto) / M
    Bc[1, 0] = 1.0 / M

    for r in range(n_terms): # add the memory kernel prony approximation terms 
        # subscript r for real part and subscript i for imaginary part
        alpha_r, beta_r, alpha_i, beta_i = prony_coeffs[r]
        ir = 2 + 2 * r
        ii = ir + 1
        Ac[1, ir] = -1.0 / M
        Ac[ir, 1] = alpha_r
        Ac[ir, ir] = beta_r
        Ac[ir, ii] = -beta_i
        Ac[ii, 1] = alpha_i
        Ac[ii, ir] = beta_i
        Ac[ii, ii] = beta_r

    Ad = la.expm(Ac * float(dt_sample)) 
    # Bd = Ac^{-1} (Ad - I) Bc  for zero-order hold on force input
    rhs = (Ad - np.eye(n_state)) @ Bc
    Bd = la.solve(Ac, rhs)

    return Ad, Bd
def solve_mpc_linearized(x_current, wave_force_pred, u_prev_traj, v_prev_traj, N_horizon, Ad, Bd, u_limit):
    # FORCE everything to be pure numerical numpy arrays immediately
    u_prev = np.array(u_prev_traj, dtype=np.float64).flatten()
    v_prev = np.array(v_prev_traj, dtype=np.float64).flatten()
    x_curr = np.array(x_current, dtype=np.float64).flatten()
    # CVXPY expects plain Python ints for variable shapes / indexing; np.integer in the shape
    # tuple can break __getitem__ (validate_key / clip) on some versions.
    Nh = int(N_horizon)

    n_state = Ad.shape[0]
    x = cp.Variable((n_state, Nh + 1))
    u = cp.Variable((1, Nh))

    cost = 0
    constraints = [x[:, 0] == x_curr] # Use the cleaned x_curr

    R_u, Q_x, Q_v = 0.1, 100.0, 50.0
    for t in range(Nh):
        # With dvdt = (... + u)/M, u*v is the rate of work BY the actuator ON the body.
        # Maximising absorbed power means resisting motion: keep u*v negative → minimise u*v
        # (first-order expansion of u*v about the previous iterate).
        power_lin_uv = u_prev[t] * x[1, t] + v_prev[t] * u[0, t] - (u_prev[t] * v_prev[t])

        cost += power_lin_uv + R_u * cp.square(u[0, t]) + Q_x * cp.square(x[0, t]) + Q_v * cp.square(x[1, t])

        f_t = float(wave_force_pred[t])
        # Use flatten() on Bd to ensure it's a 1D array for vector addition
        constraints += [x[:, t+1] == Ad @ x[:, t] + Bd.flatten() * (u[0, t] + f_t)]
        
        # Split u_limit into two constraints (sometimes faster for the solver)
        constraints += [u[0, t] <= u_limit, u[0, t] >= -u_limit] 

    problem = cp.Problem(cp.Minimize(cost), constraints)
    # Use a small timeout or verbose=False to prevent console hanging
    problem.solve(solver=cp.CLARABEL, verbose=False)

    if u.value is None or problem.status not in ["optimal", "optimal_inaccurate"]:
        return 0.0, np.zeros(Nh), np.zeros(Nh)

    # STRICT numerical extraction
    u_opt = float(u.value[0, 0])
    u_traj_values = np.array(u.value, dtype=np.float64).flatten()
    v_traj_values = np.array(x.value[1, :-1], dtype=np.float64).flatten()
    
    return u_opt, u_traj_values, v_traj_values

# def solve_mpc_linearized(x_current, wave_force_pred, u_prev_traj, v_prev_traj, N_horizon, Ad, Bd, u_limit):
#     """
#     u_prev_traj: The 'u' values calculated in the LAST simulation step
#     v_prev_traj: The 'velocity' (x[1]) values calculated in the LAST simulation step
#     """
#     n_state = Ad.shape[0]
#     x = cp.Variable((n_state, N_horizon + 1))
#     u = cp.Variable((1, N_horizon))

#     # Ensure inputs are standard numpy arrays
#     u_prev = np.asarray(u_prev_traj).flatten()
#     v_prev = np.asarray(v_prev_traj).flatten()
#     Bd_vec = np.asarray(Bd).flatten() 
#     u_limit = float(u_limit)

#     cost = 0
#     constraints = [x[:, 0] == x_current]

#     for t in range(N_horizon):
#         # print(f'solving mpc step t={t}')
#         # Linearized Power: P = u*v
#         # We minimize -Power to maximize Power
#         power_lin = u_prev[t] * x[1, t] + v_prev[t] * u[0, t] - (u_prev[t] * v_prev[t])
        
#         # Add a small quadratic penalty on 'u' and 'x' for numerical stability (regularization)
#         cost += -power_lin + 0.1 * cp.square(u[0, t]) + 10.0 * cp.square(x[0, t])

#         f_t = float(wave_force_pred[t])
#         constraints += [x[:, t+1] == Ad @ x[:, t] + Bd_vec * (u[0, t] + f_t)] # this is the implements the constraint of physics that the next step is a result of the control and state currently
#         constraints += [u[0, t] <= u_limit, u[0, t] >= 0]
#         # constraints += [cp.abs(x[0, t]) <= 2.0] # Limit buoy to +/- 2 meters

#     problem = cp.Problem(cp.Minimize(cost), constraints)
#     problem.solve(solver=cp.CLARABEL)

#     if problem.status not in ["optimal", "optimal_inaccurate"]:
#         return 0.0, np.zeros(N_horizon), np.zeros(N_horizon)

#     # We MUST extract the numerical values here
#     u_opt = float(u.value[0, 0])
#     u_traj_values = np.array(u.value).flatten().copy() 
#     v_traj_values = np.array(x.value[1, :-1]).flatten().copy()
    
#     # Store the result trajectory to use as the 'prev_traj' for the NEXT time step
#     return u_opt, u_traj_values, v_traj_values


def solve_mpc(x_current, wave_force_pred, N_horizon, Ad, Bd, u_limit=None):
    
    if u_limit is None:
        u_limit = u_max # maximum control force

    Nh = int(N_horizon)
    n_state = Ad.shape[0] # this shape is 3 (eta, etadot, Xr)
    x = cp.Variable((n_state, Nh + 1)) # create a matrix that contains displacement, velocity, radiation memory for each time step in the time horizon
    u = cp.Variable((1, Nh)) # create a vector of control values for each time step

    cost = 0 # intialise the cost as 0

    # instantiate the constraints which is the current state
    constraints = [x[:, 0] == x_current] # the previous states are part of the constraints of the problem

        # Constants for weighting
    Q_displacement = 100.0  # Penalty for moving too much
    Q_velocity = 50.0
    R_control = 0.1         # Penalty for using too much force

    for t in range(Nh):
        # Minimise u*v (work rate into body by actuator) ≈ resist motion / extract power
        cost += u[0, t] * x[1, t]
        cost += Q_displacement * cp.square(x[0, t])  # Stability Term
        cost += Q_velocity * cp.square(x[1, t])
        cost += R_control * cp.square(u[0, t])  # Smoothness Term

        f_t = float(wave_force_pred[t])

        # Ensure Bd is correctly shaped for the matrix multiplication
        constraints += [x[:, t + 1] == Ad @ x[:, t] + Bd @ (u[:, t] + f_t)]
        constraints += [cp.abs(u[0, t]) <= u_limit]

    problem = cp.Problem(cp.Minimize(cost), constraints) # solve the convex optimisation problem to get optimal controls
    problem.solve(solver=cp.CLARABEL)

    if u.value is None:
        print("Solver failed to find a solution")
        return 0.0

    return float(u.value[0, 0])



def solve_cummins_stepwise_mpc(buoy, A_heave_inf, prony_coeffs, t_kernel, kernel, K_heave, F_ex_time, t_span, Ad, Bd, x0, dt=0.05, n_horizon=20):

    print('Initialising function: solve_cummins_stepwise_mpc')

    # Plant IVP uses control as a raw force (no C_pto*v). Rebuild discrete matrices with C_pto=0 so
    # MPC dynamics match that plant; RunMPC's Ad/Bd (often with args.cpto) are not used here.
    Ad, Bd = discrete_matrices_for_mpc_from_prony(prony_coeffs, buoy.mass, A_heave_inf, 0.0, K_heave, dt)
    n_state = Ad.shape[0]

    # 2. Initialize Shadow State and History for the mpc model
    # Shadow state tracks [pos, vel, p1, q1...] for the MPC's internal logic
    x_shadow = np.array(x0, dtype=np.float64).flatten().copy()


    M_eff = buoy.mass + A_heave_inf

    # full solution output stored in dictionary
    history = {
        't': [0.0],
        'x': [x_shadow[0]],
        'v': [x_shadow[1]],
        'u': [0.0]
    }

    # Neutral linearisation point for u*v; v_prev from shadow velocity each step
    u_prev_traj = np.zeros(int(n_horizon))
    v_prev_traj = np.full(int(n_horizon), float(x_shadow[1]))

    # simulation loop — use t_step as current time (t_now was never updated, so horizon & shadow input were frozen at t_span[0])
    for t_step in np.arange(t_span[0], t_span[1], dt):
        # Horizon samples must match MPC step dt (linspace(..., n_horizon) does not give spacing dt when n_horizon > 1).
        t_horizon = t_step + np.arange(n_horizon, dtype=np.float64) * dt
        f_pred = [F_ex_time(tt) for tt in t_horizon]

        # solve MPC for optimal control U
        u_opt, u_traj, v_traj = solve_mpc_linearized(x_shadow, f_pred, u_prev_traj, v_prev_traj, n_horizon, Ad, Bd, u_limit=20000)

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

            # no kpto and cpto replaced as raw control force
            dvdt = (F_ex_time(t) - memory - K_heave * x + u_opt) / M_eff
            return [v, dvdt]
    
        # use solve ivp to solve the cummins for the step with u_opt
        sol = solve_ivp(rhs, [t_step, t_step + dt], [history['x'][-1], history['v'][-1]], max_step=dt/2)

        # update the shadow state
        # We synchronize the shadow state's pos/vel with the plant's actual output
        # then advance the hidden Prony states using the Ad matrix.
        # Step the whole shadow vector forward by dt
        x_shadow_next = Ad @ x_shadow + Bd.flatten() * (u_opt + F_ex_time(t_step))

        x_shadow_next[0] = float(sol.y[0,-1]) # update with real data to avoid mpc sol drifting from the actual sol
        x_shadow_next[1] = float(sol.y[1,-1]) # actual velocity update

        x_shadow = np.array(x_shadow_next, dtype=np.float64)

        # F. Log Results
        history['t'].append(float(sol.t[-1]))
        history['x'].append(sol.y[0, -1])
        history['v'].append(sol.y[1, -1])
        history['u'].append(u_opt)

        u_prev_traj = np.append(u_traj[1:], u_traj[-1])
        v_prev_traj = np.append(v_traj[1:], v_traj[-1])

    return history

def plot_history(history_mpc, history_no_control):
    print('Initialising function: plot_history')
    plt.title('Displacement Time graph for point absorber')
    plt.plot(history_mpc['t'], history_mpc['x'], label='with MPC')
    plt.plot(history_no_control['t'], history_no_control['x'], label='No mpc')
    plt.plot(history_no_control['t'], np.array(history_no_control['F_ex']) / (np.max(np.array(history_no_control['F_ex'])) / (max(history_no_control['x']))), ls='--', label='scaled excitation force')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m) / Force (N)/100000')
    plt.grid(True, alpha=0.25)
    plt.show()


'''------------plot power----------'''

def plot_power(history_mpc, history_no_control, p_inst_mpc, p_inst_no_control):
    print('Initialising function: plot_power')
    plt.title('Instaneous Power Plot')
    plt.plot(history_mpc['t'][50:], p_inst_mpc, label='with latch')
    plt.plot(history_no_control['t'][50:], p_inst_no_control, label='without latch')
    plt.xlabel('Time (s)')
    plt.ylabel('Power (w)')
    plt.legend()
    plt.show()


def calc_power_absorbed_mpc(history):

    print('Initialising function: calc_power_absorbed')

    v = np.array(history['v'][50:]) # remove transients
    u = np.array(history['u'][50:])
    # Mechanical power into the body from control force u in (+u) EOM is u*v; absorbed by PTO is -u*v.
    p_inst = -u * v
    p_mean = np.mean(p_inst) # average power
    return p_inst, p_mean
