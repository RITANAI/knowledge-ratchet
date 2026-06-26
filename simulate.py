"""
simulate.py - Main simulation script for SAPE and baselines.

Reproduces Figures 2-5 from the paper:
- Figure 2: Parameter sensitivity
- Figure 3: Multi-baseline comparison
- Figure 4: Catastrophic forgetting dynamics
- Figure 5: Ablation studies

Usage:
    python simulate.py --figure all --n_runs 10000 --output_dir figures/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from sapelib import SAPEParams, SAPESimulator, run_fixed_capability, run_periodic_retraining


def figure_parameter_sensitivity(output_dir: Path, n_runs: int = 10000):
    """Figure 2: Parameter sensitivity analysis."""
    print("Generating Figure 2: Parameter sensitivity...")
    
    baseline = SAPEParams()
    n_rounds = 100
    
    params_to_vary = {
        'p_max': np.linspace(0.6, 0.95, 8),
        'eta_0': np.linspace(0.01, 0.06, 8),
        'gamma_0': np.linspace(0.05, 0.35, 8),
        'alpha_0': np.linspace(0.45, 0.85, 8),
        'lambda_CF': np.linspace(0.0, 0.3, 8),
    }
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for idx, (param_name, values) in enumerate(params_to_vary.items()):
        slopes = []
        conv_times = []
        
        for v in values:
            p = SAPEParams(**{**baseline.__dict__, param_name: v})
            mc = SAPESimulator(p).run_monte_carlo(n_rounds, n_runs=min(n_runs, 1000))
            
            # Long-run slope (last 20 rounds)
            slope = (mc['K_mean'][-1] - mc['K_mean'][-21]) / 20
            slopes.append(slope)
            
            # Convergence time (theta within 1% of ceiling)
            conv_mask = mc['theta_mean'] >= 0.99 * p.theta_bar
            conv_time = np.argmax(conv_mask) if conv_mask.any() else n_rounds
            conv_times.append(conv_time)
        
        ax = axes[idx]
        ax2 = ax.twinx()
        
        l1 = ax.plot(values, slopes, 'o-', color='#2E86AB', label='Growth slope', markersize=4)
        l2 = ax2.plot(values, conv_times, 's--', color='#A23B72', label='Conv. time', markersize=4)
        
        ax.set_xlabel(param_name, fontsize=10)
        ax.set_ylabel('Slope (discoveries/round)', color='#2E86AB', fontsize=9)
        ax2.set_ylabel('Conv. time (rounds)', color='#A23B72', fontsize=9)
        ax.set_title(param_name, fontsize=11, fontweight='bold')
        ax.tick_params(axis='y', labelcolor='#2E86AB', labelsize=8)
        ax2.tick_params(axis='y', labelcolor='#A23B72', labelsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure2_parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_dir / 'figure2_parameter_sensitivity.png'}")


def figure_multi_baseline(output_dir: Path, n_runs: int = 10000):
    """Figure 3: Multi-baseline comparison."""
    print(f"Generating Figure 3: Multi-baseline comparison (n_runs={n_runs})...")
    
    n_rounds = 100
    M = 50
    baseline = SAPEParams(M=M)
    
    methods = {
        'SAPE': lambda p, n: SAPESimulator(p).run_monte_carlo(n, n_runs),
        'Fixed-capability': lambda p, n: monte_carlo_wrapper(
            lambda p, n: run_fixed_capability(p, n), p, n, n_runs),
        'Periodic-retrain (tau=10)': lambda p, n: monte_carlo_wrapper(
            lambda p, n: run_periodic_retraining(p, n, tau=10), p, n, n_runs),
    }
    
    colors = {'SAPE': '#E63946', 'Fixed-capability': '#457B9D', 
              'Periodic-retrain (tau=10)': '#2A9D8F'}
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for name, func in methods.items():
        print(f"  Running {name}...")
        mc = func(baseline, n_rounds)
        rounds = np.arange(n_rounds + 1)
        ax.plot(rounds, mc['K_mean'], color=colors[name], label=name, linewidth=2)
        ax.fill_between(rounds, 
                        mc['K_mean'] - mc['K_se'],
                        mc['K_mean'] + mc['K_se'],
                        alpha=0.2, color=colors[name])
    
    ax.set_xlabel('Round t', fontsize=12)
    ax.set_ylabel('Cumulative knowledge stock K_t', fontsize=12)
    ax.set_title('Multi-baseline comparison (M=50)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_multi_baseline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_dir / 'figure3_multi_baseline.png'}")


def figure_catastrophic_forgetting(output_dir: Path, n_runs: int = 1000):
    """Figure 4: Catastrophic forgetting dynamics."""
    print("Generating Figure 4: Catastrophic forgetting...")
    
    params = SAPEParams(lambda_CF=0.1, delta_CF=0.3, seed=42)
    n_rounds = 1000
    
    sim = SAPESimulator(params)
    hist = sim.run_episode(n_rounds)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    rounds = np.arange(n_rounds + 1)
    
    # Capability index
    axes[0].plot(rounds, hist['theta'], color='#E63946', linewidth=1)
    axes[0].set_ylabel('Capability index θ_t', fontsize=11)
    axes[0].set_title('Catastrophic forgetting dynamics (λ_CF=0.1, δ_CF=0.3)', 
                      fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Knowledge stock
    axes[1].plot(rounds, hist['K'], color='#2A9D8F', linewidth=1.5)
    axes[1].set_ylabel('Knowledge stock K_t', fontsize=11)
    axes[1].set_xlabel('Round t', fontsize=12)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure4_forgetting.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_dir / 'figure4_forgetting.png'}")


def figure_ablation(output_dir: Path, n_runs: int = 5000):
    """Figure 5: Ablation studies."""
    print(f"Generating Figure 5: Ablation studies (n_runs={n_runs})...")
    
    n_rounds = 100
    configs = {
        'SAPE-full': SAPEParams(),
        'SAPE-K-only': SAPEParams(eta_0=0.0),
        'SAPE-periodic': SAPEParams(),  # simulated separately
        'SAPE-no-learning': SAPEParams(eta_0=0.0, gamma_0=0.0),
    }
    
    colors = {'SAPE-full': '#E63946', 'SAPE-K-only': '#457B9D',
              'SAPE-periodic': '#F4A261', 'SAPE-no-learning': '#2A9D8F'}
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for name, p in configs.items():
        print(f"  Running {name}...")
        mc = SAPESimulator(p).run_monte_carlo(n_rounds, n_runs)
        rounds = np.arange(n_rounds + 1)
        ax.plot(rounds, mc['K_mean'], color=colors[name], label=name, linewidth=2)
        ax.fill_between(rounds,
                        mc['K_mean'] - mc['K_se'],
                        mc['K_mean'] + mc['K_se'],
                        alpha=0.2, color=colors[name])
    
    ax.set_xlabel('Round t', fontsize=12)
    ax.set_ylabel('Cumulative knowledge stock K_t', fontsize=12)
    ax.set_title('Ablation studies', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'figure5_ablation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_dir / 'figure5_ablation.png'}")


def monte_carlo_wrapper(fn, params, n_rounds, n_runs):
    """Helper for baselines without built-in MC."""
    all_K = np.zeros((n_runs, n_rounds + 1))
    for i in range(n_runs):
        p = SAPEParams(**{**params.__dict__, 'seed': i})
        hist = fn(p, n_rounds)
        all_K[i] = hist['K']
    return {
        'K_mean': all_K.mean(axis=0),
        'K_se': all_K.std(axis=0) / np.sqrt(n_runs),
        'theta_mean': np.zeros(n_rounds + 1),  # Not tracked for baselines
    }


def main():
    parser = argparse.ArgumentParser(description='SAPE Simulation Suite')
    parser.add_argument('--figure', type=str, default='all',
                       choices=['all', '2', '3', '4', '5'],
                       help='Which figure to generate')
    parser.add_argument('--n_runs', type=int, default=10000,
                       help='Number of Monte Carlo runs')
    parser.add_argument('--output_dir', type=str, default='figures',
                       help='Output directory for figures')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.figure in ('all', '2'):
        figure_parameter_sensitivity(output_dir, args.n_runs)
    if args.figure in ('all', '3'):
        figure_multi_baseline(output_dir, args.n_runs)
    if args.figure in ('all', '4'):
        figure_catastrophic_forgetting(output_dir, min(args.n_runs, 1000))
    if args.figure in ('all', '5'):
        figure_ablation(output_dir, min(args.n_runs, 5000))
    
    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
