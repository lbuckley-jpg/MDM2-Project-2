"""
QUICK PRONY VALIDATION - Easy Integration
==========================================
Simple validation checks you can add to your pontryagin_sol function.
"""

import numpy as np
import warnings


def quick_prony_check(t_grid, K_data, prony_coeffs, prony_model_func, 
                      plot=False, verbose=True):
    """
    Quick validation of Prony coefficients with pass/fail result.
    
    This is a simplified version designed for easy integration into your
    existing pontryagin_sol function.
    
    Parameters:
    -----------
    t_grid : array_like
        Time points
    K_data : array_like
        Original K(t) from Riemann sum
    prony_coeffs : ndarray
        Fitted Prony coefficients, shape (N, 4)
    prony_model_func : callable
        Your prony_model function
    plot : bool
        If True, make a simple comparison plot
    verbose : bool
        If True, print validation summary
        
    Returns:
    --------
    is_valid : bool
        True if fit passes basic validation checks
    metrics : dict
        Dictionary with R², RMSE, unstable_poles, etc.
    """
    
    # Reconstruct fitted K(t)
    K_fitted = prony_model_func(t_grid, *prony_coeffs.flatten())
    
    # Calculate metrics
    residuals = K_data - K_fitted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((K_data - np.mean(K_data))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Check for unstable poles
    beta_R = prony_coeffs[:, 1]  # Real parts of beta
    n_unstable = np.sum(beta_R >= 0)
    
    # Validation criteria
    r2_ok = r_squared > 0.80
    stability_ok = n_unstable == 0
    
    is_valid = r2_ok and stability_ok
    
    metrics = {
        'r_squared': r_squared,
        'rmse': rmse,
        'n_unstable_poles': n_unstable,
        'beta_R': beta_R,
        'is_valid': is_valid
    }
    
    if verbose:
        print("\n" + "="*50)
        print("PRONY VALIDATION QUICK CHECK")
        print("="*50)
        print(f"R² = {r_squared:.4f}  {'✓' if r2_ok else '✗'}")
        print(f"Unstable poles: {n_unstable}  {'✓' if stability_ok else '✗'}")
        print(f"\nResult: {'PASS ✓' if is_valid else 'FAIL ✗'}")
        print("="*50)
        
        if not is_valid:
            if not r2_ok:
                print("⚠ Poor fit - try increasing N_prony")
            if not stability_ok:
                print("⚠ Unstable poles detected - FIT IS INVALID")
                print(f"  Unstable pole indices: {np.where(beta_R >= 0)[0]}")
    
    if plot:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # K(t) comparison
        ax1.plot(t_grid, K_data, 'b-', label='Original', linewidth=2, alpha=0.7)
        ax1.plot(t_grid, K_fitted, 'r--', label='Fitted', linewidth=2)
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('K(t)')
        ax1.set_title(f'K(t) Comparison (R²={r_squared:.3f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Pole locations
        beta_I = prony_coeffs[:, 3]
        ax2.scatter(beta_R, beta_I, s=100, alpha=0.6, edgecolors='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax2.fill_betweenx(ax2.get_ylim(), 0, ax2.get_xlim()[1], 
                         alpha=0.2, color='red', label='Unstable')
        ax2.set_xlabel('Re(β) - Decay Rate')
        ax2.set_ylabel('Im(β) - Frequency')
        ax2.set_title('Pole Locations')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    return is_valid, metrics


# ================================================================
# HOW TO INTEGRATE INTO YOUR pontryagin_sol FUNCTION
# ================================================================

def pontryagin_sol_with_validation(omega, B, times, params, f_ex, 
                                   N_prony=4, max_iter=100):
    """
    Example showing how to integrate quick_prony_check into your code.
    """
    from scipy.optimize import curve_fit
    from scipy.integrate import solve_ivp
    from scipy.interpolate import interp1d
    
    # Your existing prony_model function
    def prony_model(t_grid, *params):
        params = np.array(params)
        N = len(params) // 4
        result = np.zeros_like(t_grid)
        for i in range(N):
            alpha_R = params[4*i]
            beta_R  = params[4*i+1]
            alpha_I = params[4*i+2]
            beta_I  = params[4*i+3]
            result += alpha_R*np.exp(beta_R*t_grid)*np.cos(beta_I*t_grid)
            result -= alpha_I*np.exp(beta_R*t_grid)*np.sin(beta_I*t_grid)
        return result
    
    m, m_inf, d, G, k = params
    t_grid = times
    
    # STEP 1: Compute K(t)
    K_data = np.zeros_like(t_grid, dtype=float)
    for j in range(len(omega)-1):
        dw = omega[j+1] - omega[j]
        K_data += B[j] * np.cos(omega[j] * t_grid) * dw
    K_data = (2/np.pi) * K_data
    
    # STEP 2: Fit Prony coefficients
    prony_guess = np.array([
        [1.0, -0.1, 0.2, 0.1],
        [2.0, -0.4, 0.5, 0.2],
        [2.0, -0.4, 0.5, 0.2],
        [2.0, -0.4, 0.5, 0.2]
    ][:N_prony])
    
    try:
        popt, _ = curve_fit(prony_model, t_grid, K_data, 
                           p0=prony_guess.flatten(), maxfev=5000)
        prony_coeffs = popt.reshape(-1, 4)
        print("✓ Prony fit completed")
    except Exception as e:
        print(f"⚠ Prony fit failed: {e}")
        print("  Using initial guess")
        prony_coeffs = prony_guess
    
    # ============================================================
    # ADD THIS: Validate the Prony fit
    # ============================================================
    is_valid, metrics = quick_prony_check(
        t_grid, K_data, prony_coeffs, prony_model,
        plot=False,  # Set to True to see plots
        verbose=True
    )
    
    if not is_valid:
        warnings.warn(
            "Prony fit validation failed! Results may be inaccurate. "
            "Consider increasing N_prony or checking your K(t) data.",
            UserWarning
        )
        
        # Optional: Auto-retry with more terms
        if metrics['r_squared'] < 0.80:
            print(f"\n→ Auto-retrying with N_prony={N_prony+2}...")
            return pontryagin_sol_with_validation(
                omega, B, times, params, f_ex,
                N_prony=N_prony+2, max_iter=max_iter
            )
    # ============================================================
    
    # Continue with rest of algorithm...
    # (your existing iteration code here)
    
    return None  # Placeholder


# ================================================================
# ALTERNATIVE: Constrained optimization for stability
# ================================================================

def fit_prony_with_constraints(t_grid, K_data, N_prony=4):
    """
    Fit Prony coefficients with constraints ensuring stability.
    
    This guarantees all Re(β) < 0 (stable poles).
    """
    from scipy.optimize import minimize
    
    def prony_model(params, t):
        N = len(params) // 4
        result = np.zeros_like(t)
        for i in range(N):
            alpha_R = params[4*i]
            beta_R  = params[4*i+1]
            alpha_I = params[4*i+2]
            beta_I  = params[4*i+3]
            result += alpha_R*np.exp(beta_R*t)*np.cos(beta_I*t)
            result -= alpha_I*np.exp(beta_R*t)*np.sin(beta_I*t)
        return result
    
    def objective(params):
        """Sum of squared residuals"""
        K_fit = prony_model(params, t_grid)
        return np.sum((K_data - K_fit)**2)
    
    # Initial guess
    x0 = np.array([
        1.0, -0.1, 0.2, 0.1,
        2.0, -0.4, 0.5, 0.2,
    ] * (N_prony // 2))[:4*N_prony]
    
    # Bounds: force beta_R < 0 for stability
    bounds = []
    for i in range(N_prony):
        bounds.append((-np.inf, np.inf))    # alpha_R: unbounded
        bounds.append((-10, -0.001))        # beta_R: MUST be negative
        bounds.append((-np.inf, np.inf))    # alpha_I: unbounded
        bounds.append((-10, 10))            # beta_I: bounded frequency
    
    print("Fitting with stability constraints...")
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': 5000})
    
    if result.success:
        prony_coeffs = result.x.reshape(-1, 4)
        print("✓ Constrained fit successful")
        # Verify all poles are stable
        assert np.all(prony_coeffs[:, 1] < 0), "Unstable poles found!"
        return prony_coeffs
    else:
        print(f"✗ Constrained fit failed: {result.message}")
        return None


# ================================================================
# EXAMPLE USAGE
# ================================================================

if __name__ == "__main__":
    print("Quick Prony Validation - Usage Examples\n")
    
    # Example 1: Basic validation check
    print("="*60)
    print("Example 1: Quick validation check")
    print("="*60)
    
    # Synthetic data
    t_test = np.linspace(0, 20, 200)
    
    def prony_model_test(t, *params):
        params = np.array(params)
        N = len(params) // 4
        result = np.zeros_like(t)
        for i in range(N):
            alpha_R = params[4*i]
            beta_R  = params[4*i+1]
            alpha_I = params[4*i+2]
            beta_I  = params[4*i+3]
            result += alpha_R*np.exp(beta_R*t)*np.cos(beta_I*t)
            result -= alpha_I*np.exp(beta_R*t)*np.sin(beta_I*t)
        return result
    
    # True K(t)
    true_coeffs = np.array([[1.0, -0.2, 0.3, 0.5]])
    K_true = prony_model_test(t_test, *true_coeffs.flatten())
    
    # Test with good fit
    print("\nCase 1: Good fit (should PASS)")
    fitted_coeffs = np.array([[0.98, -0.19, 0.31, 0.48]])
    is_valid, metrics = quick_prony_check(
        t_test, K_true, fitted_coeffs, prony_model_test, verbose=True
    )
    
    # Test with unstable poles
    print("\n\nCase 2: Unstable poles (should FAIL)")
    bad_coeffs = np.array([[1.0, 0.2, 0.3, 0.5]])  # beta_R > 0!
    is_valid, metrics = quick_prony_check(
        t_test, K_true, bad_coeffs, prony_model_test, verbose=True
    )
    
    print("\n" + "="*60)
    print("INTEGRATION GUIDE")
    print("="*60)
    print("""
To add validation to your code:

1. QUICK CHECK (recommended):
   
   After fitting Prony coefficients, add:
   
   from prony_validation import quick_prony_check
   
   is_valid, metrics = quick_prony_check(
       t_grid, K_data, prony_coeffs, prony_model,
       verbose=True
   )
   
   if not is_valid:
       warnings.warn("Prony fit failed validation!")

2. FULL DIAGNOSTICS (for development):
   
   from prony_validation import validate_prony_coefficients
   
   results, is_valid = validate_prony_coefficients(
       t_grid, K_data, prony_coeffs, prony_model,
       plot=True,
       save_path='prony_diagnostics.png'
   )

3. CONSTRAINED FITTING (prevents unstable poles):
   
   from prony_validation import fit_prony_with_constraints
   
   prony_coeffs = fit_prony_with_constraints(
       t_grid, K_data, N_prony=4
   )
   
   # This GUARANTEES all Re(β) < 0
""")
