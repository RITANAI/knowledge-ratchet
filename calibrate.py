"""
calibrate.py - A-Lab data calibration script.

Nonlinear least-squares parameter fitting to round-level success rates
from Szymanski et al. (Nature 2023) A-Lab autonomous laboratory data.

Usage:
    python calibrate.py --data data/alab_success_rates.csv --output results/
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm
from pathlib import Path
import json
import argparse


# A-Lab round-level success rates (41 rounds, extracted from Szymanski et al.)
# These are illustrative values; replace with actual extracted data
ALAB_SUCCESS_RATES = np.array([
    0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.36,
    0.38, 0.40, 0.42, 0.43, 0.45, 0.46, 0.48, 0.50, 0.51, 0.52,
    0.54, 0.55, 0.56, 0.58, 0.59, 0.60, 0.62, 0.63, 0.64, 0.65,
    0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75,
    0.76
])


def model_success_rate(params, K_prev, N_checkpoint, M=50, H=50, c_star=-0.64, sigma_eps=1.0):
    """
    Predict success rate given parameters and state.
    
    params = [alpha_0, gamma_0, eta_0, p_max]
    """
    alpha_0, gamma_0, eta_0, p_max = params
    
    # Simplified: assume theta_t tracks the observed success rate
    # with the SAPE update applied to the mean
    theta_t = min(0.95, 0.4 + eta_0 * np.mean(ALAB_SUCCESS_RATES[:len(K_prev)]))
    
    p_pred = []
    for i, K in enumerate(K_prev):
        experience = gamma_0 * K / (H + K)
        capability = alpha_0 * theta_t
        p = min(p_max, capability + experience)
        p_pred.append(p)
    
    return np.array(p_pred)


def residuals(params, observed_rates, K_prev, N_checkpoint):
    """Residual function for least squares."""
    predicted = model_success_rate(params, K_prev, N_checkpoint)
    return observed_rates - predicted


def calibrate_alab(observed_rates: np.ndarray, M: int = 50, H: float = 50.0):
    """Calibrate SAPE parameters to A-Lab data."""
    n_rounds = len(observed_rates)
    
    # Construct knowledge stock trajectory (cumulative successes)
    K_prev = np.cumsum(observed_rates * M).astype(int)
    K_prev = np.insert(K_prev[:-1], 0, 0)
    
    N_checkpoint = np.full(n_rounds, M * 0.5)  # Estimated checkpoint count
    
    # Initial parameter guess
    p0 = [0.6, 0.2, 0.03, 0.8]
    
    # Bounds
    lower_bounds = [0.1, 0.0, 0.001, 0.3]
    upper_bounds = [1.0, 0.5, 0.1, 1.0]
    
    # Fit
    result = least_squares(
        residuals, p0,
        args=(observed_rates, K_prev, N_checkpoint),
        bounds=(lower_bounds, upper_bounds),
        method='trf',
        max_nfev=10000
    )
    
    alpha_0, gamma_0, eta_0, p_max = result.x
    
    # Compute R^2
    predicted = model_success_rate(result.x, K_prev, N_checkpoint)
    ss_res = np.sum((observed_rates - predicted) ** 2)
    ss_tot = np.sum((observed_rates - np.mean(observed_rates)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    n = len(observed_rates)
    k = len(result.x)
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
    
    # Bootstrap standard errors
    n_bootstrap = 1000
    boot_params = []
    rng = np.random.default_rng(42)
    
    for _ in range(n_bootstrap):
        idx = rng.choice(n_rounds, size=n_rounds, replace=True)
        boot_rates = observed_rates[idx]
        boot_K = K_prev[idx]
        boot_N = N_checkpoint[idx]
        
        try:
            boot_result = least_squares(
                residuals, p0,
                args=(boot_rates, boot_K, boot_N),
                bounds=(lower_bounds, upper_bounds),
                method='trf',
                max_nfev=5000
            )
            boot_params.append(boot_result.x)
        except:
            pass
    
    boot_params = np.array(boot_params)
    se = boot_params.std(axis=0)
    
    return {
        'alpha_0': alpha_0,
        'gamma_0': gamma_0,
        'eta_0': eta_0,
        'p_max': p_max,
        'alpha_0_se': se[0],
        'gamma_0_se': se[1],
        'eta_0_se': se[2],
        'p_max_se': se[3],
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'n_rounds': n_rounds,
        'n_params': k,
    }


def main():
    parser = argparse.ArgumentParser(description='A-Lab Data Calibration')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to A-Lab success rate CSV (optional)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    if args.data:
        import pandas as pd
        df = pd.read_csv(args.data)
        observed_rates = df['success_rate'].values
    else:
        print("Using built-in illustrative A-Lab data.")
        print("Replace with actual extracted data for publication-grade calibration.")
        observed_rates = ALAB_SUCCESS_RATES
    
    # Calibrate
    print(f"Calibrating to {len(observed_rates)} rounds of A-Lab data...")
    results = calibrate_alab(observed_rates)
    
    # Print results
    print("\n" + "="*60)
    print("A-LAB CALIBRATION RESULTS")
    print("="*60)
    print(f"alpha_0 = {results['alpha_0']:.3f} (SE = {results['alpha_0_se']:.3f})")
    print(f"gamma_0 = {results['gamma_0']:.3f} (SE = {results['gamma_0_se']:.3f})")
    print(f"eta_0   = {results['eta_0']:.3f} (SE = {results['eta_0_se']:.3f})")
    print(f"p_max   = {results['p_max']:.3f} (SE = {results['p_max_se']:.3f})")
    print(f"R^2     = {results['r_squared']:.3f}")
    print(f"Adj R^2 = {results['adj_r_squared']:.3f}")
    print("="*60)
    
    # Save
    with open(output_dir / 'calibration_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'calibration_results.json'}")


if __name__ == "__main__":
    main()
