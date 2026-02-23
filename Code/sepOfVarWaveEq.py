
import numpy as np
import scipy 
import matplotlib.pyplot as plt
from scipy.integrate import quad
import json
import time

def f(x, dom):
    intvl = dom
    mid = (intvl[0] + intvl[1]) / 2

    # Left half, including the midpoint
    if intvl[0] <= x <= mid:
        return 2 * x - 1
    # Right half, up to and including the upper bound
    elif mid < x <= intvl[1]:
        return -2 * x + 1
    else:
        # region agent log
        with open(r"c:\Users\edwar\OneDrive - University of Bristol\Year 2\Project 2\MDM2-Project-2\.cursor\debug.log", "a", encoding="utf-8") as _log_f:
            _timestamp_ms = int(time.time() * 1000)
            _log_f.write(json.dumps({
                "id": f"log_{_timestamp_ms}",
                "timestamp": _timestamp_ms,
                "location": "surfacegravitywaves.py:f",
                "message": "f fell through without matching interval (outside domain)",
                "data": {"x": float(x), "dom": dom},
                "runId": "post-fix",
                "hypothesisId": "H1"
            }) + "\n")
        # endregion
        return 0.0

def coefficient_b_n(n, lower, upper):
    dom = [lower, upper]
    integral_value, _ = quad(
        lambda x: f(x, dom) * np.sin(n * np.pi * x / upper),
        lower,
        upper
    )
    return (2.0 / upper) * integral_value


def solve(t, x , n_terms, lower, upper, c):
    sol_truncated = 0
    for n in range(1, n_terms+1):
        b = coefficient_b_n(n, lower, upper)
        sol_truncated += b * np.sin(n * np.pi * x / upper) * np.cos(n * np.pi * c * t / upper)
    return sol_truncated



def solve_wave_equation(f, n_terms, domain_interval, c):
    
    # seperate the domain

    lower = domain_interval[0]
    upper = domain_interval[1]

    # create the solution space

    t = np.arange(0, 120) * 0.5

    x = np.linspace(lower, upper, 1000)

    # create a mesh grid for all combos of t and x to define inputs of a 2d space

    T, X = np.meshgrid(t, x, indexing="ij")

    # solve the problem

    u = solve(T, X, n_terms, lower, upper, c) 

    # plot solution

    plt.figure()
    # Here T and X must have same shape as u
    pcm = plt.pcolormesh(X, T, u, shading="auto", cmap="viridis")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.colorbar(pcm, label="u(x, t)")
    plt.show()


solve_wave_equation(f, 5, [0,1], 5)
    

















