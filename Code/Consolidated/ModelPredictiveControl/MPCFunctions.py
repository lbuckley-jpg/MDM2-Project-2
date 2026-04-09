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
import cvxpy as cp # module for optimisation

'''dummy parameters for now will use capytaine and hopefully andrews prony method'''
dt = 0.1            # Sampling time (T_mpc)
N = 20              # Prediction Horizon (how many steps to look ahead)
m = 1000            # Buoy mass
a_inf = 500           # Added mass
k = 5000            # Hydrostatic stiffness
M = m + a_w
u_max = 1500  
C_hydro = 5000      # Maximum PTO force (Physical constraint)
M_total = m + a_inf

order_r = 6
Ar = np.diag([-0.5, -0.6, -0.7, -0.8, -0.9, -1.0]) # example stable poles (negative real parts)
Br = np.ones((order_r, 1))
Cr = np.array([[100, 80, 60, 40, 20, 10]]) #example coupling coefficients


'''--------------solve_mpc---------------'''


# Discrete-time State-Space (x = [pos, vel])
# Derived from: x[k+1] = Ad*x[k] + Bd*u[k]
Ac = np.zeros((8,8))

# the buoy's physics
Ac[0,1] = 1.0 # intial velocity
Ac[1, 0] = - C_hydro / M_total
Ac[1, 2:] = -Cr / M_total

# Radiation state rows
Ac[2:, 1:2] = Br                                
Ac[2:, 2:] = Ar

Bc = np.zeros((8, 1))
Bc[1, 0] = 1 / M_total

# Dicscrete model matrices

Ad = la.expm(Ac * dt)
# Bd = Ac^-1 * (Ad - I) * Bc
# We could use a more robust integral method in case Ac is singular
Bd = la.inv(Ac) @ (Ad - np.eye(8)) @ Bc



# solves for the optimal controls in time horizon
def solve_mpc(x_current, wave_force_pred, N):

    # creat two matrices that describe the system. 
    # X describes the state of the system (vertical position, vertical velocity, radiation states)
    # U contains the control 

    x = cp.Variable((8, N + 1)) # dimensions are displacement and velocity + 6 radiation parameters for N + 1 timesteps
    u = cp.Variable((1, N))  # control froce for N timesteps

    # instantiate cost of objective function as 0
    cost = 0

    # instantiate the constraints which is the current state
    constraints = [x[:, 0] == x_current]

    # iterate throught the timer horizon 

    for t in range(N):

        # calculate the cost update at each timestep
        cost += - u[:,t] @ x[1,t] + 0.1 * cp.square(u[:, t]) # - force * veloctiy + 0.1 * velocity squared # added regularisation to avoid to larger force

        # update the system dynamics constraints for the next time step
        # Note: wave_force_pred should be added via the same input matrix as 'u'
        constraints += [x[:, t+1] == Ad @ x[:, t] + Bd @ (u[:, t] + wave_force_pred[t])]

        # update the physical contraints such as force limits

        constraints += [cp.abs(u[:,t]) <= u_max]

    # define the optimisation problem
    problem = cp.Problem(cp.Minimize(cost), constraints)

    # solve the problem
    problem.solve(solver=cp.ECOS)
    
    if u.value is None:
        print("Solver failed to find a solution")
        return 0.0

    return u.value[0,0]
    

