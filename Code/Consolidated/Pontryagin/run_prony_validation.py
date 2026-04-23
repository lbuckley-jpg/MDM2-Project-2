import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for SharedCapytaineFunctions
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from SharedCapytaineFunctions import (
    generate_frequencies,
    generate_buoy,
    solve_with_capytaine,
)
from PontryaginFunctions import (
    compute_K,
    fit_prony_coefficients,
    prony_model
)
from prony_validation import validate_prony_coefficients, plot_prony_relative_error

def run_validation():
    print("Setting up Prony Validation with real data...")
    
    # 1. Setup parameters (similar to RunPontryagin defaults)
    buoy_radius = 5.0
    buoy_mass = 314000.0
    peak_period = 12.0
    n_freq = 52
    tspan = 60.0  # Sufficient time to see decay
    dt = 0.05
    n_prony = 7
    
    # 2. Build buoy and solve hydrodynamics
    print("Generating buoy and solving hydrodynamics (Capytaine)...")
    buoy = generate_buoy(radius=buoy_radius, mass=buoy_mass)
    omegas, delta_omega = generate_frequencies(N=n_freq, Tp=peak_period)
    
    capytaine_dataset = solve_with_capytaine(
        body=buoy, 
        omegas=omegas, 
        water_density=1025.0
    )
    
    radiation_damping = capytaine_dataset["radiation_damping"].sel(
        radiating_dof="Heave", influenced_dof="Heave"
    ).values
    
    added_mass_inf = float(
        capytaine_dataset["added_mass"].sel(radiating_dof="Heave", influenced_dof="Heave").values[-1]
    )
    
    hydrostatic_stiffness = float(
        buoy.hydrostatic_stiffness.sel(influenced_dof="Heave", radiating_dof="Heave")
    )
    
    t_grid = np.arange(0.0, tspan + dt, dt)
    
    # 3. Fit Prony coefficients
    print(f"Fitting {n_prony} Prony terms...")
    
    prony_coeffs, K_data = fit_prony_coefficients(
        t_grid=t_grid,
        omega=omegas,
        radiation_damping=radiation_damping,
        mass=buoy_mass,
        added_mass_inf=added_mass_inf,
        pto_damping=50000.0,
        hydrostatic_stiffness=hydrostatic_stiffness,
        n_terms=n_prony
    )
    
    # 4. Run Validation (Specific plot requested)
    print("\nGenerating percentage error plot...")
    output_filename = 'prony_percentage_error.png'
    
    plot_prony_relative_error(
        t_grid=t_grid,
        K_data=K_data,
        prony_coeffs=prony_coeffs,
        prony_model=prony_model,
        save_path=output_filename
    )
    
    # Still run validation metrics for the console output
    results, is_valid = validate_prony_coefficients(
        t_grid=t_grid,
        K_data=K_data,
        prony_coeffs=prony_coeffs,
        prony_model=prony_model,
        plot=False
    )
    
    print("\n" + "="*60)
    print(f"Validation Result: {'PASSED' if is_valid else 'FAILED'}")
    print(f"R-squared: {results['r_squared']:.6f}")
    print(f"RMSE: {results['rmse']:.6e}")
    print("="*60)
    print(f"\nPercentage error plot saved to: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    run_validation()
