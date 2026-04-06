from numpy import pi
from numpy import inf
import numpy as np
import capytaine as cpt
from capytaine.bem.airy_waves import airy_waves_free_surface_elevation
from capytaine.ui.vtk import Animation

from capytaine.post_pro import rao as rao_fn # for PTO dampning simulation
import xarray as xr

import sys
import argparse

import os # import operating system for saving results to files
import time # import time for data stamping the results. thought it would be useful


cpt.set_logging('WARNING')

bem_solver = cpt.BEMSolver()

'''
adaptation for a buoy
'''

def generate_buoy(radius= 5, mass= 500):
    # Geometry: a sphere of radius 3 centered at the free surface (z = 0)
    buoy_mesh = cpt.mesh_sphere(radius=radius, center=(0.0, 0.0, 0.0), resolution=(30, 30))

    # Use a consistent rotation center and center of mass
    rotation_center = (0.0, 0.0, 0.0)

    specified_mass = mass # kg

    buoy = cpt.FloatingBody(
        mesh=buoy_mesh,
        dofs=cpt.rigid_body_dofs(rotation_center=rotation_center),
        center_of_mass=rotation_center,
        mass=specified_mass,
        name="Point Absorber",
    )

    buoy.radius = radius # store radius with the body object for later use (when logging for example)


    # Keep the same “artificially lower” inertia and hydrostatic stiffness pattern
    buoy.inertia_matrix = buoy.compute_rigid_body_inertia() # assumes density equal to that of fluid. I assume ot avoid accidental sinking
    buoy.hydrostatic_stiffness = buoy.immersed_part().compute_hydrostatic_stiffness()

    return buoy


def simulate(body, fs, omega=2*pi/8, wave_amplitude=2, wave_direction=pi, water_depth = inf, water_density = 1000, c_pto = 0.0 , k_pto = 0.0, visualize=True, save=True):

    '''Solve Boundary Element Method Problems'''

    radiation_problems = [cpt.RadiationProblem(omega=omega, body=body.immersed_part(), radiating_dof=dof, water_depth = water_depth, rho = water_density) for dof in body.dofs]
    radiation_results = bem_solver.solve_all(radiation_problems)
    diffraction_problem = cpt.DiffractionProblem(omega=omega, body=body.immersed_part(), wave_direction=wave_direction, water_depth = water_depth, rho = water_density)
    diffraction_result = bem_solver.solve(diffraction_problem)

    dataset = cpt.assemble_dataset(radiation_results + [diffraction_result])

    # rao = cpt.post_pro.rao(dataset, wave_direction=wave_direction) # (for without PTO)(now redundant)

    '''Define PTO matrices in DOF space'''

    dofs = list(body.dofs.keys())
    ndof = len(dofs)

    # initiate PTO damping matrix B_pto (same ordering as dofs)

    B_pto = np.zeros((ndof, ndof))

    # Find index of heave DOF

    iheave = dofs.index("Heave")  

    B_pto[iheave, iheave] = c_pto # add coefficient to dampning matrix

    # Optional PTO stiffness matrix K_pto

    K_pto = np.zeros((ndof, ndof))

    if k_pto != 0.0:
        K_pto[iheave, iheave] = k_pto


    # Use the same DOF ordering and names as the hydrodynamic matrices

    rad_dims = dataset["radiation_damping"].dims[1:]  # ('radiating_dof','influenced_dof')
    rad_coords = {
        rad_dims[0]: dataset["radiation_damping"].coords[rad_dims[0]],
        rad_dims[1]: dataset["radiation_damping"].coords[rad_dims[1]],
    }

    B_pto_da = xr.DataArray(B_pto, coords=rad_coords, dims=rad_dims)
    K_pto_da = xr.DataArray(K_pto, coords=rad_coords, dims=rad_dims)

    print("Heave index:", iheave, "c_pto:", c_pto, "k_pto:", k_pto)
    
    
    rao = cpt.post_pro.rao(
        dataset,
        wave_direction=wave_direction,
        dissipation=B_pto_da,
        stiffness=K_pto_da,
        )

    '''Compute analysis energy and power metrics'''

    rao_heave = rao.sel(omega=omega, radiating_dof="Heave")

    X = rao_heave.data * wave_amplitude # calculate the complex heave displacement

    vel_amp = np.abs(1j * omega * X) # Velocity amplitude [m/s]

    P_heave = 0.5 * c_pto * vel_amp ** 2 # Calcluate the mean absorbed power in heave pto [W]

    T = 2 * pi / omega # time period

    E_cycle = P_heave * T # energy equals power multiplied by time

    if save == True:

        '''Write results to file'''

        logfile = 'results/pto_results.csv'
        file_exists = os.path.isfile(logfile)

        with open(logfile, 'a') as f:
            if not file_exists:
                # first time running create the file
                f.write("timestamp,omega_rad_s,frequency_Hz,wave_amplitude_m,"
                "c_pto_Ns_m,k_pto_N_m,buoy_mass_kg,buoy_radius_m,"
                    "water_depth_m,water_density_kg_m3,"
                    "P_absorbed_W,E_cycle_J\n")

            freq = omega / (2 * np.pi)

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S") # format data into a readable string

            # log the results as a new row in the file

            

            f.write(f"{timestamp},{omega}, {freq}, {wave_amplitude},"
                f"{c_pto}, {k_pto}, {body.mass}, {body.radius},"  # radius hard-coded or parameter
                f"{water_depth}, {water_density},"
                f"{P_heave}, {E_cycle}\n")

            print('Run logged succesfully')


    if not visualize:
        return float(P_heave), float(E_cycle)

    if visualize:
        '''Compute Free Surface Elevation'''

        # Compute the diffracted wave pattern
        incoming_waves_elevation = airy_waves_free_surface_elevation(fs, diffraction_result)
        diffraction_elevation = bem_solver.compute_free_surface_elevation(fs, diffraction_result)

        # Compute the wave pattern radiated by the RAO
        radiation_elevations_per_dof = {res.radiating_dof: bem_solver.compute_free_surface_elevation(fs, res) for res in radiation_results}
        radiation_elevation = sum(rao.sel(omega=omega, radiating_dof=dof).data * radiation_elevations_per_dof[dof] for dof in body.dofs)


        '''Set up Animation'''

        # Compute the motion of each face of the mesh for the animation
        rao_faces_motion = sum(rao.sel(omega=omega, radiating_dof=dof).data * body.dofs[dof] for dof in body.dofs)

        # Set up scene
        animation = Animation(loop_duration=2*pi/omega)
        animation.add_body(body, faces_motion=wave_amplitude*rao_faces_motion)
        animation.add_free_surface(fs, wave_amplitude * (incoming_waves_elevation + diffraction_elevation + radiation_elevation))
        return animation


if __name__ == '__main__':
    
    # argument parser

    parser = argparse.ArgumentParser(description="Run Wave Simulation")

    # script run parameters

    parser.add_argument("--visualize", type=bool, required= False)
    parser.add_argument("--save", type=bool, required= False)

    # parameters for the buoy
    parser.add_argument("--buoymass", type=float, required=False)
    parser.add_argument("--buoyradius", type=float, required=False)

    # parameters for the water

    parser.add_argument("--waterdensity", type=float, required=False)
    parser.add_argument("--waterdepth", type=float, required=False)

    # paramters for the wave
    parser.add_argument("--wavefrequency", type=float, required=False)
    parser.add_argument("--wavedirection", type=float, required=False)
    parser.add_argument("--waveamplitude", type=float, required=False)

    # parameters for power take off

    parser.add_argument("--cpto", type=float, required = False)
    parser.add_argument("--kpto", type=float, required = False)

    # define args from parseer

    args = parser.parse_args()

    body = generate_buoy(radius = args.buoyradius, mass = args.buoymass)

    fs = cpt.FreeSurface(x_range=(-50, 50), y_range=(-50, 50), nx=150, ny=150)

    omega = 2 * pi * args.wavefrequency # convert frequency to angular frequency

    anim = simulate(body, fs, omega = omega, wave_amplitude = args.waveamplitude, wave_direction = args.wavedirection, water_depth= args.waterdepth, water_density = args.waterdensity, c_pto=args.cpto, k_pto = args.kpto, visualize = args.visualize, save= args.save)
    anim.run(camera_position=(0, 80, 8), resolution=(1280, 720))
    # anim.save("point_absorber_simulation_animation.ogv", camera_position=(0, 80, 8), resolution=(800, 600))




    