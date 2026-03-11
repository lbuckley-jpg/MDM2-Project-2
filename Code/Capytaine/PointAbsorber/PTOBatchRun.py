from numpy import pi
import numpy as np
import capytaine as cpt
from PTOPointAbsorberSimulation import generate_buoy, simulate

WATER_DENSITY = 1000.0
WATER_DEPTH   = 100.0
WAVE_AMP      = 2.0
WAVE_DIR      = pi


def run_case(body, fs, freq_hz, c_pto, k_pto=0.0):
    """Run one simulation and return (P_heave, E_cycle)."""
    return simulate(
        body, fs,
        omega=2 * pi * freq_hz,
        wave_amplitude=WAVE_AMP,
        wave_direction=WAVE_DIR,
        water_depth=WATER_DEPTH,
        water_density=WATER_DENSITY,
        c_pto=c_pto,
        k_pto=k_pto,
        visualize=False,
        save=True,
    )


def optimise_cpto(body, fs, freq_hz, k_pto=0.0,
                  c_lo=0.0, c_hi=5e5, n_pts=10, n_rounds=3):
    """Zoom search: find c_pto that maximises E_cycle for (body, freq)."""
    best_c, best_E, best_P = 0.0, -np.inf, 0.0

    for _ in range(n_rounds):
        for c in np.linspace(c_lo, c_hi, n_pts):
            P, E = run_case(body, fs, freq_hz, c, k_pto)
            if E > best_E:
                best_c, best_E, best_P = c, E, P

        step = (c_hi - c_lo) / max(n_pts - 1, 1)
        c_lo = max(0.0, best_c - step)
        c_hi = min(5e5, best_c + step)

    return best_c, best_P, best_E


def main():
    masses = np.linspace(1_000, 100_000, 15)
    freqs  = np.linspace(0.15, 0.30, 25)
    k_pto  = 0.0

    for m in masses:
        body = generate_buoy(radius=5.0, mass=m)
        fs = cpt.FreeSurface(x_range=(-100, 75), y_range=(-100, 75), nx=100, ny=100)

        for f in freqs:
            c_opt, P_opt, E_opt = optimise_cpto(
                body, fs, f, k_pto,
                c_lo=0.0, c_hi=5e5, n_pts=10, n_rounds=3,
            )
            print(f"m={m:8.0f} kg  f={f:.3f} Hz  "
                  f"c_pto*={c_opt:.2e}  E*={E_opt:.2f} J  P*={P_opt:.2f} W")


if __name__ == "__main__":
    main()



