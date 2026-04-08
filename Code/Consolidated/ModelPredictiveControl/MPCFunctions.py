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
import cvxpy as xp # module for optimisation



'''--------------build_prediction_matrices-------'''

# create a discrete state space version of system that predicts the future state of buoy.

# parameters are:
# Ad matrix for [cummins dynamics input memory]
# Np is the time horizon the number of time step into the future to optimise to
# Bd is matrix which relates the change in control force to change in system
# Fd is matrix which relates the change in excitation force to change in system

# the result is we get a linear equation that predicts state updates discretely

def build_prediction_matrices(Ad, Bd, Fd, Cd, Np):
    
    nx = Ad.shape[0]
    ny = Cd.shape[0]

    P = []
    WB = np.zeros((ny*Np, Np))
    WF = np.zeros((ny*Np, Np))

    for i in range(1, Np+1):
        Ai = np.linalg.matrix_power(Ad, i)
        P.append(Cd @ Ai)

        for j in range(i):
            Aij = np.linalg.matrix_power(Ad, i-j-1)
            WB[(i-1)*ny:i*ny, j] = (Cd @ Aij @ Bd).flatten()
            WF[(i-1)*ny:i*ny, j] = (Cd @ Aij @ Fd).flatten()

    P = np.vstack(P)
    return P, WB, WF