from numpy import pi
import numpy as np
import capytaine as cpt

from PTOPointAbsorberSimulation import generate_buoy, simulate


def main():

    # Fixed environment
    water_density = 1000.0
    water_depth = 100.0
    wave_amplitude = 2.0
    wave_direction = pi  # 3.14159

    '''Grid definitions'''

    mass_min, mass_max, mass_n = 100000.0, 1000.0, 1000
    freq_min, freq_max, freq_n = 0.15, 0.30, 7
    cpto_min, cpto_max, cpto_n = 5e4, 2e5, 4
    k_pto = 5.0e5

    masses = np.linspace(mass_min, mass_max, mass_n)
    freqs = [0.2]
    cptos = [0]


    for m in masses:
        body = generate_buoy(radius=5.0, mass=m)
        fs = cpt.FreeSurface(x_range=(-100, 75), y_range=(-100, 75), nx=100, ny=100)

        for f in freqs:
            omega = 2 * pi * f
            for c in cptos:
                simulate(
                    body,
                    fs,
                    omega=omega,
                    wave_amplitude=wave_amplitude,
                    wave_direction=wave_direction,
                    water_depth=water_depth,
                    water_density=water_density,
                    c_pto=c,
                    k_pto=k_pto,
                    visualize=False,
                    save=True,
                )


if __name__ == "__main__":
    main()