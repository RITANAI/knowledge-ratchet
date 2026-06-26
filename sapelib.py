"""
sapelib.py - Core SAPE (Stochastic Approximation from Physical Evidence) simulation library.

Implements the Knowledge Ratchet model with:
- Scalar capability-index tracking
- Knowledge stock accumulation
- Failure-driven hypothesis pruning
- Stochastic capability resets (catastrophic forgetting)
- Multiple baseline comparisons

Reference: Yan (2026) "The Knowledge Ratchet" 
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable


@dataclass
class SAPEParams:
    """Parameters for the SAPE model."""
    alpha_0: float = 0.65      # Capability scaling
    gamma_0: float = 0.2       # Experience-driven learning
    eta_0: float = 0.03        # SAPE learning rate
    zeta_0: float = 0.0        # Failure learning rate (pruning)
    theta_0: float = 0.4       # Initial capability
    theta_bar: float = 0.95    # Physical ceiling
    p_max: float = 0.8         # Max validation success
    M: int = 50                # Batch size
    H: float = 50.0            # Saturation parameter
    c_star: float = -0.64      # Checkpoint threshold (Phi^{-1}(0.26))
    sigma_eps: float = 1.0     # Noise std
    kappa: float = 0.0         # Pruning stringency
    psi_bar: float = 1.0       # Pruning ceiling
    lambda_CF: float = 0.0     # Forgetting probability
    delta_CF: float = 0.0      # Forgetting loss fraction
    seed: Optional[int] = None


class SAPESimulator:
    """SAPE simulator with the Knowledge Ratchet architecture."""
    
    def __init__(self, params: SAPEParams):
        self.p = params
        self.rng = np.random.default_rng(params.seed)
        
    def phi(self, theta: float, psi: float = 0.0) -> float:
        """Checkpoint pass probability."""
        c_star_t = self.p.c_star + self.p.kappa * psi
        z = (norm.ppf(self.p.alpha_0 * theta) - c_star_t) / self.p.sigma_eps
        return norm.cdf(z)
    
    def p_success(self, theta: float, K_prev: int) -> float:
        """Validation success probability (Eq. 1)."""
        experience = self.p.gamma_0 * K_prev / (self.p.H + K_prev)
        capability = self.p.alpha_0 * theta
        return min(self.p.p_max, capability + experience)
    
    def run_episode(self, n_rounds: int, 
                    theta_init: Optional[float] = None,
                    K_init: int = 0,
                    psi_init: float = 0.0) -> dict:
        """Run a single SAPE episode."""
        theta = theta_init if theta_init is not None else self.p.theta_0
        K = K_init
        psi = psi_init
        
        history = {
            'theta': np.zeros(n_rounds + 1),
            'K': np.zeros(n_rounds + 1, dtype=int),
            'psi': np.zeros(n_rounds + 1),
            'N': np.zeros(n_rounds, dtype=int),
            'Delta_K': np.zeros(n_rounds, dtype=int),
            'p_t': np.zeros(n_rounds),
            'phi_t': np.zeros(n_rounds),
        }
        history['theta'][0] = theta
        history['K'][0] = K
        history['psi'][0] = psi
        
        for t in range(n_rounds):
            # Catastrophic forgetting
            if self.p.lambda_CF > 0:
                if self.rng.random() < self.p.lambda_CF:
                    theta_min = (1 - self.p.delta_CF) * self.p.theta_0
                    theta = max((1 - self.p.delta_CF) * theta, theta_min)
            
            # Checkpoint
            phi_t = self.phi(theta, psi)
            N_t = self.rng.binomial(self.p.M, phi_t)
            history['phi_t'][t] = phi_t
            history['N'][t] = N_t
            
            # Validation
            if N_t > 0:
                p_t = self.p_success(theta, K)
                Delta_K_t = self.rng.binomial(N_t, p_t)
            else:
                p_t = self.p_success(theta, K)
                Delta_K_t = 0
            
            history['p_t'][t] = p_t
            history['Delta_K'][t] = Delta_K_t
            
            # Update knowledge stock (ratchet - never decreases)
            K = K + Delta_K_t
            history['K'][t + 1] = K
            
            # SAPE update (Eq. 2)
            if N_t > 0:
                theta_new = min(self.p.theta_bar, 
                               theta + self.p.eta_0 * (Delta_K_t / N_t))
            else:
                theta_new = theta
            theta = theta_new
            history['theta'][t + 1] = theta
            
            # Pruning update (Eq. 3)
            if N_t > 0 and self.p.zeta_0 > 0:
                psi = min(self.p.psi_bar,
                         psi + self.p.zeta_0 * ((N_t - Delta_K_t) / N_t))
            history['psi'][t + 1] = psi
        
        return history
    
    def run_monte_carlo(self, n_rounds: int, n_runs: int = 10000,
                       seeds: Optional[List[int]] = None) -> dict:
        """Run Monte Carlo simulation."""
        if seeds is None:
            seeds = list(range(n_runs))
        
        all_theta = np.zeros((n_runs, n_rounds + 1))
        all_K = np.zeros((n_runs, n_rounds + 1), dtype=int)
        
        for i in range(n_runs):
            p_run = SAPEParams(**{**self.p.__dict__, 'seed': seeds[i]})
            sim = SAPESimulator(p_run)
            hist = sim.run_episode(n_rounds)
            all_theta[i] = hist['theta']
            all_K[i] = hist['K']
        
        return {
            'theta_mean': all_theta.mean(axis=0),
            'theta_se': all_theta.std(axis=0) / np.sqrt(n_runs),
            'K_mean': all_K.mean(axis=0),
            'K_se': all_K.std(axis=0) / np.sqrt(n_runs),
            'theta_all': all_theta,
            'K_all': all_K,
        }


# ----- Baseline implementations -----

def run_fixed_capability(params: SAPEParams, n_rounds: int) -> dict:
    """Fixed-capability baseline (no learning, only experience)."""
    p = SAPEParams(**{**params.__dict__, 'eta_0': 0.0})
    sim = SAPESimulator(p)
    return sim.run_episode(n_rounds)

def run_periodic_retraining(params: SAPEParams, n_rounds: int, 
                            tau: int = 10) -> dict:
    """Periodic retraining baseline."""
    rng = np.random.default_rng(params.seed)
    theta = params.theta_0
    K = 0
    
    history = {'theta': [theta], 'K': [K], 'p_t': []}
    
    for t in range(n_rounds):
        sim = SAPESimulator(params)
        # Run for tau rounds with current theta, then retrain
        sub_hist = sim.run_episode(1, theta_init=theta, K_init=K)
        theta = sub_hist['theta'][-1]
        K = sub_hist['K'][-1]
        
        if (t + 1) % tau == 0:
            # Retrain: partial reset with knowledge gain
            theta = min(params.theta_bar, 
                       params.theta_0 + 0.1 * (K / (params.H + K)))
        
        history['theta'].append(theta)
        history['K'].append(K)
    
    history['theta'] = np.array(history['theta'])
    history['K'] = np.array(history['K'])
    return history


if __name__ == "__main__":
    # Quick sanity check
    params = SAPEParams(seed=42)
    sim = SAPESimulator(params)
    hist = sim.run_episode(100)
    print(f"Final K_100 = {hist['K'][-1]}")
    print(f"Final theta_100 = {hist['theta'][-1]:.4f}")
