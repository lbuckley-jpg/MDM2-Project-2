from numpy import pi
from numpy import inf
import capytaine as cpt
from capytaine.bem.airy_waves import airy_waves_free_surface_elevation
from capytaine.ui.vtk import Animation

# for 

import sys
import argparse

cpt.set_logging('INFO')

bem_solver = cpt.BEMSolver()

'''
adaptation for a buoy
'''

def generate_buoy(radius= 5, mass= 500):
    # Geometry: a sphere of radius 3 centered at the free surface (z = 0)
    buoy_mesh = cpt.mesh_sphere(radius=radius, center=(0.0, 0.0, 0.0), resolution=(20, 20))

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

    # Keep the same “artificially lower” inertia and hydrostatic stiffness pattern
    buoy.inertia_matrix = buoy.compute_rigid_body_inertia() # assumes density equal to that of fluid. I assume ot avoid accidental sinking
    buoy.hydrostatic_stiffness = buoy.immersed_part().compute_hydrostatic_stiffness()

    return buoy


def setup_animation(body, fs, omega=2*pi/8, wave_amplitude=2, wave_direction=pi, water_depth = inf, water_density = 1000):

    '''Solve Boundary Element Method Problems'''

    radiation_problems = [cpt.RadiationProblem(omega=omega, body=body.immersed_part(), radiating_dof=dof, water_depth = water_depth, rho = water_density) for dof in body.dofs]
    radiation_results = bem_solver.solve_all(radiation_problems)
    diffraction_problem = cpt.DiffractionProblem(omega=omega, body=body.immersed_part(), wave_direction=wave_direction, water_depth = water_depth, rho = water_density)
    diffraction_result = bem_solver.solve(diffraction_problem)

    dataset = cpt.assemble_dataset(radiation_results + [diffraction_result])
    rao = cpt.post_pro.rao(dataset, wave_direction=wave_direction)

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

    # define args from parseer

    args = parser.parse_args()

    body = generate_buoy(radius = args.buoyradius, mass = args.buoymass)

    fs = cpt.FreeSurface(x_range=(-100, 75), y_range=(-100, 75), nx=100, ny=100)

    omega = 2 * pi * args.wavefrequency # convert frequency to angular frequency

    anim = setup_animation(body, fs, omega = omega, wave_amplitude = args.waveamplitude, wave_direction = args.wavedirection, water_depth= args.waterdepth, water_density = args.waterdensity)
    anim.run(camera_position=(0, 80, 8), resolution=(800, 600))
    # anim.save("point_absorber_simulation_animation.ogv", camera_position=(0, 80, 8), resolution=(800, 600))




    