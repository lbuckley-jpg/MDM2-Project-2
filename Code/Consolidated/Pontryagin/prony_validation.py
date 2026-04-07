"""
PRONY COEFFICIENT VALIDATION TOOLS
===================================
Tools to validate that curve_fit found physically valid and accurate Prony coefficients.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def validate_prony_coefficients(t_grid, K_data, prony_coeffs, prony_model, 
                                  plot=True, save_path=None):
    """
    Comprehensive validation of Prony coefficient fitting.
    
    Checks:
    1. Goodness of fit (R², RMSE)
    2. Physical validity (stable poles: Re(β) < 0)
    3. Visual comparison
    4. Residual analysis
    5. Energy/integral conservation
    
    Parameters:
    -----------
    t_grid : array_like
        Time points where K(t) was computed
    K_data : array_like
        Original K(t) data from Riemann sum
    prony_coeffs : ndarray
        Fitted Prony coefficients, shape (N, 4)
    prony_model : callable
        Prony model function
    plot : bool
        Whether to generate diagnostic plots
    save_path : str, optional
        Path to save diagnostic plots
        
    Returns:
    --------
    validation_results : dict
        Dictionary containing all validation metrics
    is_valid : bool
        True if coefficients pass all validation checks
    """
    
    # Reconstruct fitted K(t) from Prony coefficients
    K_fitted = prony_model(t_grid, *prony_coeffs.flatten())
    
    results = {}
    
    # ============================================
    # 1. GOODNESS OF FIT METRICS
    # ============================================
    
    # Residuals
    residuals = K_data - K_fitted
    
    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((K_data - np.mean(K_data))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # RMSE
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Normalized RMSE
    nrmse = rmse / (np.max(K_data) - np.min(K_data)) if np.max(K_data) != np.min(K_data) else np.inf
    
    # Mean Absolute Error
    mae = np.mean(np.abs(residuals))
    
    # Max absolute error
    max_error = np.max(np.abs(residuals))
    
    results['r_squared'] = r_squared
    results['rmse'] = rmse
    results['nrmse'] = nrmse
    results['mae'] = mae
    results['max_error'] = max_error
    
    print("="*60)
    print("PRONY COEFFICIENT VALIDATION")
    print("="*60)
    print(f"\n1. GOODNESS OF FIT:")
    print(f"   R² = {r_squared:.6f}  {'✓ GOOD' if r_squared > 0.95 else '✗ POOR' if r_squared < 0.8 else '⚠ FAIR'}")
    print(f"   RMSE = {rmse:.6e}")
    print(f"   Normalized RMSE = {nrmse:.4f}  {'✓ GOOD' if nrmse < 0.05 else '✗ POOR' if nrmse > 0.2 else '⚠ FAIR'}")
    print(f"   Mean Absolute Error = {mae:.6e}")
    print(f"   Max Absolute Error = {max_error:.6e}")
    
    # ============================================
    # 2. PHYSICAL VALIDITY CHECKS
    # ============================================
    
    N = prony_coeffs.shape[0]
    beta_R = prony_coeffs[:, 1]  # Real parts of beta (decay rates)
    beta_I = prony_coeffs[:, 3]  # Imaginary parts (oscillation frequencies)
    
    # Check for stability: All Re(β) should be negative for stable system
    unstable_poles = beta_R >= 0
    n_unstable = np.sum(unstable_poles)
    
    # Check for unreasonably large values
    large_decay = np.abs(beta_R) > 10  # Adjust threshold based on your system
    large_freq = np.abs(beta_I) > 10
    
    results['n_unstable_poles'] = n_unstable
    results['beta_R'] = beta_R
    results['beta_I'] = beta_I
    results['unstable_indices'] = np.where(unstable_poles)[0]
    
    print(f"\n2. PHYSICAL VALIDITY:")
    print(f"   Number of Prony terms: {N}")
    print(f"   Unstable poles (Re(β) ≥ 0): {n_unstable}/{N}  {'✗ INVALID' if n_unstable > 0 else '✓ VALID'}")
    
    if n_unstable > 0:
        print(f"   Unstable pole indices: {results['unstable_indices']}")
        for idx in results['unstable_indices']:
            print(f"      Term {idx+1}: βᴿ = {beta_R[idx]:.4f}, βᴵ = {beta_I[idx]:.4f}")
    
    print(f"\n   Decay rates (Re(β)):")
    for i in range(N):
        status = "✓" if beta_R[i] < 0 else "✗"
        print(f"      Term {i+1}: {beta_R[i]:+.6f}  {status}")
    
    print(f"\n   Oscillation frequencies (Im(β)):")
    for i in range(N):
        print(f"      Term {i+1}: {beta_I[i]:+.6f}")
    
    # ============================================
    # 3. ENERGY/INTEGRAL CHECK
    # ============================================
    
    # Compare integrals (important for radiation force calculation)
    # Use trapezoidal rule for numerical integration
    try:
        # NumPy >= 2.0
        integral_original = np.trapezoid(K_data, t_grid)
        integral_fitted = np.trapezoid(K_fitted, t_grid)
    except AttributeError:
        # NumPy < 2.0
        integral_original = np.trapz(K_data, t_grid)
        integral_fitted = np.trapz(K_fitted, t_grid)
    
    integral_error = np.abs(integral_original - integral_fitted)
    integral_rel_error = integral_error / np.abs(integral_original) if integral_original != 0 else np.inf
    
    results['integral_original'] = integral_original
    results['integral_fitted'] = integral_fitted
    results['integral_rel_error'] = integral_rel_error
    
    print(f"\n3. INTEGRAL/ENERGY CONSERVATION:")
    print(f"   ∫K(t)dt (original) = {integral_original:.6e}")
    print(f"   ∫K(t)dt (fitted)   = {integral_fitted:.6e}")
    print(f"   Relative error = {integral_rel_error:.4%}  {'✓ GOOD' if integral_rel_error < 0.05 else '✗ POOR' if integral_rel_error > 0.2 else '⚠ FAIR'}")
    
    # ============================================
    # 4. EARLY TIME vs LATE TIME BEHAVIOR
    # ============================================
    
    # Check fit quality at different time scales
    mid_point = len(t_grid) // 2
    
    early_error = np.mean(np.abs(residuals[:mid_point]))
    late_error = np.mean(np.abs(residuals[mid_point:]))
    
    results['early_mae'] = early_error
    results['late_mae'] = late_error
    
    print(f"\n4. TIME-DEPENDENT FIT QUALITY:")
    print(f"   Early time MAE (t < {t_grid[mid_point]:.2f}s): {early_error:.6e}")
    print(f"   Late time MAE (t ≥ {t_grid[mid_point]:.2f}s): {late_error:.6e}")
    if late_error > 2 * early_error:
        print(f"   ⚠ WARNING: Late-time fit is poor (error ratio = {late_error/early_error:.2f})")
    
    # ============================================
    # 5. OVERALL VALIDITY DECISION
    # ============================================
    
    is_valid = (
        r_squared > 0.80 and           # Reasonable fit
        n_unstable == 0 and            # All poles stable
        nrmse < 0.2 and                # Normalized error acceptable
        integral_rel_error < 0.2       # Energy approximately conserved
    )
    
    print(f"\n{'='*60}")
    print(f"OVERALL VALIDATION: {'✓ PASSED' if is_valid else '✗ FAILED'}")
    print(f"{'='*60}")
    
    if not is_valid:
        print("\nRECOMMENDATIONS:")
        if r_squared < 0.80:
            print("  - Increase number of Prony terms (N)")
            print("  - Try different initial guess")
        if n_unstable > 0:
            print("  - Unstable poles detected - fit is physically invalid")
            print("  - Try constraining beta_R < 0 in optimization")
        if nrmse > 0.2:
            print("  - Poor fit quality - check K(t) data for issues")
        if integral_rel_error > 0.2:
            print("  - Energy not conserved - check time grid resolution")
    
    # ============================================
    # 6. DIAGNOSTIC PLOTS
    # ============================================
    
    if plot:
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        
        # Plot 1: K(t) comparison
        axes[0, 0].plot(t_grid, K_data, 'b-', linewidth=2, label='Original K(t)', alpha=0.7)
        axes[0, 0].plot(t_grid, K_fitted, 'r--', linewidth=2, label='Fitted K(t)')
        axes[0, 0].set_xlabel('Time [s]')
        axes[0, 0].set_ylabel('K(t)')
        axes[0, 0].set_title(f'K(t) Comparison (R² = {r_squared:.4f})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        axes[0, 1].plot(t_grid, residuals, 'g-', linewidth=1)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0, 1].fill_between(t_grid, residuals, alpha=0.3, color='green')
        axes[0, 1].set_xlabel('Time [s]')
        axes[0, 1].set_ylabel('Residual')
        axes[0, 1].set_title(f'Residuals (RMSE = {rmse:.4e})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Log scale comparison (to see late-time behavior)
        axes[1, 0].semilogy(t_grid, np.abs(K_data) + 1e-10, 'b-', linewidth=2, 
                            label='|Original K(t)|', alpha=0.7)
        axes[1, 0].semilogy(t_grid, np.abs(K_fitted) + 1e-10, 'r--', linewidth=2, 
                            label='|Fitted K(t)|')
        axes[1, 0].set_xlabel('Time [s]')
        axes[1, 0].set_ylabel('|K(t)| (log scale)')
        axes[1, 0].set_title('K(t) - Log Scale (decay behavior)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, which='both')
        
        # Plot 4: Relative error
        rel_error = np.abs(residuals) / (np.abs(K_data) + 1e-10) * 100
        axes[1, 1].plot(t_grid, rel_error, 'purple', linewidth=1)
        axes[1, 1].axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='5% threshold')
        axes[1, 1].axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% threshold')
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 1].set_ylabel('Relative Error [%]')
        axes[1, 1].set_title('Relative Error vs Time')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, min(50, np.max(rel_error))])
        
        # Plot 5: Pole locations in complex plane
        axes[2, 0].scatter(beta_R, beta_I, c='blue', s=100, alpha=0.6, edgecolors='black')
        for i in range(N):
            axes[2, 0].annotate(f'{i+1}', (beta_R[i], beta_I[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[2, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        axes[2, 0].set_xlabel('Re(β) - Decay Rate')
        axes[2, 0].set_ylabel('Im(β) - Frequency')
        axes[2, 0].set_title('Pole Locations (should all have Re(β) < 0)')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Add shading for unstable region
        ylims = axes[2, 0].get_ylim()
        axes[2, 0].fill_betweenx(ylims, 0, axes[2, 0].get_xlim()[1], 
                                 alpha=0.2, color='red', label='Unstable region')
        axes[2, 0].set_ylim(ylims)
        axes[2, 0].legend()
        
        # Plot 6: Histogram of residuals
        axes[2, 1].hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[2, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[2, 1].set_xlabel('Residual Value')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].set_title(f'Residual Distribution (μ={np.mean(residuals):.2e})')
        axes[2, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nDiagnostic plots saved to: {save_path}")
        else:
            plt.savefig('/home/claude/prony_diagnostics.png', dpi=150, bbox_inches='tight')
            print(f"\nDiagnostic plots saved to: /home/claude/prony_diagnostics.png")
    
    return results, is_valid


def suggest_prony_improvements(validation_results, K_data, t_grid):
    """
    Provide specific suggestions for improving Prony fit based on validation results.
    
    Parameters:
    -----------
    validation_results : dict
        Results from validate_prony_coefficients
    K_data : array_like
        Original K(t) data
    t_grid : array_like
        Time grid
        
    Returns:
    --------
    suggestions : list of str
        Specific actionable suggestions
    """
    suggestions = []
    
    # Check R²
    if validation_results['r_squared'] < 0.80:
        suggestions.append("POOR FIT (R² < 0.8):")
        suggestions.append("  → Increase N_prony (number of terms) from current value")
        suggestions.append("  → Try N_prony = 6 or 8 instead of 4")
    elif validation_results['r_squared'] < 0.95:
        suggestions.append("FAIR FIT (0.8 < R² < 0.95):")
        suggestions.append("  → Consider increasing N_prony by 2-4 terms")
    
    # Check unstable poles
    if validation_results['n_unstable_poles'] > 0:
        suggestions.append("\nUNSTABLE POLES DETECTED (CRITICAL):")
        suggestions.append("  → This makes the fit physically invalid!")
        suggestions.append("  → Option 1: Use constrained optimization (bounds on beta_R < 0)")
        suggestions.append("  → Option 2: Try different initial guess for Prony coefficients")
        suggestions.append("  → Option 3: Manually set unstable poles' beta_R to small negative value (-0.01)")
    
    # Check normalized RMSE
    if validation_results['nrmse'] > 0.2:
        suggestions.append("\nLARGE NORMALIZED RMSE (>20%):")
        suggestions.append("  → Check if K(t) data is noisy - consider smoothing")
        suggestions.append("  → Increase time grid resolution (more points in t_grid)")
        suggestions.append("  → Try longer simulation time to capture full K(t) decay")
    
    # Check energy conservation
    if validation_results['integral_rel_error'] > 0.2:
        suggestions.append("\nPOOR ENERGY CONSERVATION (>20% integral error):")
        suggestions.append("  → Extend t_grid to later times (K(t) not fully decayed)")
        suggestions.append("  → Increase N_prony to better capture integral")
    
    # Check late-time behavior
    if validation_results['late_mae'] > 2 * validation_results['early_mae']:
        suggestions.append("\nPOOR LATE-TIME FIT:")
        suggestions.append("  → Add more Prony terms with slower decay (smaller |beta_R|)")
        suggestions.append("  → Check if t_grid extends far enough")
    
    # Suggest optimal N_prony based on characteristics
    K_range = np.max(K_data) - np.min(K_data)
    if K_range > 0:
        # Count zero crossings as proxy for complexity
        zero_crossings = np.sum(np.diff(np.sign(K_data)) != 0)
        suggested_N = max(4, min(12, zero_crossings // 2 + 2))
        
        suggestions.append(f"\nSUGGESTED N_prony based on K(t) complexity: {suggested_N}")
        suggestions.append(f"  (Detected ~{zero_crossings} oscillations in K(t))")
    
    if len(suggestions) == 0:
        suggestions.append("✓ Prony fit looks good! No major improvements needed.")
    
    return suggestions


# Example integration with main code
def improved_prony_fitting(t_grid, omega, B, N_prony=4, prony_guess=None, 
                           prony_model_func=None, validate=True):
    """
    Enhanced Prony fitting with automatic validation.
    
    This is a drop-in replacement for the Prony fitting section in pontryagin_sol.
    """
    from scipy.optimize import curve_fit
    
    # Import your prony_model function
    if prony_model_func is None:
        raise ValueError("Must provide prony_model function")
    
    # Compute K(t)
    K_data = np.zeros_like(t_grid, dtype=float)
    for j in range(len(omega)-1):
        dw = omega[j+1] - omega[j]
        K_data += B[j] * np.cos(omega[j] * t_grid) * dw
    K_data = (2/np.pi) * K_data
    
    # Set up initial guess
    if prony_guess is None:
        prony_guess = np.array([
            [1.0, -0.1, 0.2, 0.1],
            [2.0, -0.4, 0.5, 0.2],
            [2.0, -0.4, 0.5, 0.2],
            [2.0, -0.4, 0.5, 0.2]
        ][:N_prony])
    
    # Fit Prony coefficients
    try:
        popt, pcov = curve_fit(prony_model_func, t_grid, K_data, 
                               p0=prony_guess.flatten(),
                               maxfev=5000)
        prony_coeffs = popt.reshape(-1, 4)
        fit_successful = True
        print("Prony fit completed")
    except Exception as e:
        print(f"Warning: Prony fit failed ({e})")
        print("Using initial guess as coefficients")
        prony_coeffs = prony_guess
        fit_successful = False
    
    # Validate if requested
    if validate:
        print("\nValidating Prony coefficients...")
        results, is_valid = validate_prony_coefficients(
            t_grid, K_data, prony_coeffs, prony_model_func, 
            plot=True
        )
        
        if not is_valid:
            print("\n⚠ WARNING: Prony fit validation FAILED!")
            suggestions = suggest_prony_improvements(results, K_data, t_grid)
            print("\nSUGGESTIONS FOR IMPROVEMENT:")
            for suggestion in suggestions:
                print(suggestion)
            
            # Option: Auto-retry with more terms
            user_input = input("\nRetry with more Prony terms? (y/n): ")
            if user_input.lower() == 'y':
                new_N = int(input(f"Enter new N_prony (current={N_prony}): "))
                return improved_prony_fitting(t_grid, omega, B, N_prony=new_N, 
                                             prony_model_func=prony_model_func)
    
    return prony_coeffs, K_data


if __name__ == "__main__":
    # Example usage
    print("Prony Validation Tools - Example\n")
    
    # This would normally be imported from your main code
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
    
    # Example with synthetic data
    t_test = np.linspace(0, 20, 200)
    
    # True coefficients (what we're trying to recover)
    true_coeffs = np.array([
        [1.0, -0.2, 0.3, 0.5],
        [0.5, -0.5, 0.1, 1.0]
    ])
    
    # Generate "true" K(t)
    K_true = prony_model(t_test, *true_coeffs.flatten())
    
    # Add some noise
    K_noisy = K_true + 0.05 * np.random.randn(len(t_test))
    
    # Try to fit it back
    guess = np.array([
        [1.5, -0.15, 0.2, 0.4],
        [0.8, -0.4, 0.15, 0.8]
    ])
    
    popt, _ = curve_fit(prony_model, t_test, K_noisy, p0=guess.flatten())
    fitted_coeffs = popt.reshape(-1, 4)
    
    # Validate
    results, is_valid = validate_prony_coefficients(
        t_test, K_noisy, fitted_coeffs, prony_model, plot=True
    )
    
    print("\n" + "="*60)
    print("Example completed. Check diagnostic plots.")
