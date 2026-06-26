"""
analyze.py - Analysis utilities and plotting helpers.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compute_relative_productivity(K_sape, K_baseline):
    """Compute relative productivity ratio."""
    return K_sape[-1] / max(K_baseline[-1], 1)


def plot_convergence_trajectory(history, output_path=None):
    """Plot single-run convergence trajectory."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    rounds = np.arange(len(history['theta']))
    
    axes[0].plot(rounds, history['theta'], color='#E63946', linewidth=1.5)
    axes[0].set_ylabel('Capability index θ_t', fontsize=11)
    axes[0].set_title('SAPE Convergence Trajectory', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(rounds, history['K'], color='#2A9D8F', linewidth=1.5)
    axes[1].set_ylabel('Knowledge stock K_t', fontsize=11)
    axes[1].set_xlabel('Round t', fontsize=12)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compare_productivity_table(mc_results: dict, output_path=None):
    """Generate productivity comparison table."""
    sape_K = mc_results['SAPE']['K_mean'][-1]
    
    table = []
    for name, results in mc_results.items():
        if name == 'SAPE':
            continue
        K = results['K_mean'][-1]
        ratio = sape_K / K
        premium = (ratio - 1) * 100
        table.append((name, f"{ratio:.2f}x", f"{premium:.0f}% premium"))
    
    print("\n" + "="*60)
    print("PRODUCTIVITY COMPARISON (Relative to SAPE)")
    print("="*60)
    for name, ratio, premium in table:
        print(f"{name:40s} {ratio:>10s} ({premium})")
    print("="*60)
    
    if output_path:
        import csv
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Baseline', 'Relative Productivity', 'Description'])
            for name, ratio, premium in table:
                writer.writerow([name, ratio, premium])


if __name__ == "__main__":
    print("Analysis utilities loaded. Import and use functions in your scripts.")
