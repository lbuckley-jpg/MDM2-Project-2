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

def generate_buoy(radius=5, height=10, mass=500, draft=None):
    # If draft is not specified, make it half-submerged
    if draft is None:
        draft = height / 2.0

    # Geometry: a vertical cylinder.
    # The bottom is at z = -draft. The top is at z = height - draft.
    # The cylinder centre is placed at z = height/2 - draft.
    center_z = (height / 2.0) - draft
    buoy_mesh = cpt.mesh_vertical_cylinder(
        radius=radius,
        length=height,
        center=(0.0, 0.0, center_z),
        resolution=(20, 20, 20),
    )

    # Rotation center and center of mass at the geometric centroid of the cylinder
    rotation_center = (0.0, 0.0, center_z)

    specified_mass = mass  # kg

    buoy = cpt.FloatingBody(
        mesh=buoy_mesh,
        dofs=cpt.rigid_body_dofs(rotation_center=rotation_center),
        center_of_mass=rotation_center,
        mass=specified_mass,
        name="Point Absorber",
    )

    buoy.keep_only_dofs(['Heave'])

    buoy.inertia_matrix = buoy.compute_rigid_body_inertia()
    buoy.inertia_matrix.loc[['Heave'], ['Heave']]
    buoy.hydrostatic_stiffness = buoy.immersed_part().compute_hydrostatic_stiffness()
    buoy.hydrostatic_stiffness = buoy.hydrostatic_stiffness.loc[['Heave'], ['Heave']]

    return buoy


def setup_animation(body, fs, omega=2*pi/8, wave_amplitude=2, wave_direction=pi, water_depth = inf, water_density = 1000):

    '''Solve Boundary Element Method Problems'''

    radiation_problems = [cpt.RadiationProblem(omega=omega, body=body.immersed_part(), radiating_dof='Heave', water_depth = water_depth, rho = water_density)]
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
    parser.add_argument("--buoyheight", type=float, required=False)
    parser.add_argument("--buoydraft", type=float, required=False)

    # parameters for the water

    parser.add_argument("--waterdensity", type=float, required=False)
    parser.add_argument("--waterdepth", type=float, required=False)

    # paramters for the wave
    parser.add_argument("--wavefrequency", type=float, required=False)
    parser.add_argument("--wavedirection", type=float, required=False)
    parser.add_argument("--waveamplitude", type=float, required=False)

    # define args from parseer

    args = parser.parse_args()

    body = generate_buoy(radius=args.buoyradius, height=args.buoyheight, mass=args.buoymass, draft=args.buoydraft)

    fs = cpt.FreeSurface(x_range=(-100, 75), y_range=(-100, 75), nx=100, ny=100)

    omega = 2 * pi * args.wavefrequency # convert frequency to angular frequency

    anim = setup_animation(body, fs, omega = omega, wave_amplitude = args.waveamplitude, wave_direction = args.wavedirection, water_depth= args.waterdepth, water_density = args.waterdensity)
    anim.run(camera_position=(0, 80, 8), resolution=(800, 600))
    anim.save("point_absorber_simulation_animation.ogv", camera_position=(0, 80, 8), resolution=(800, 600))




    