"""
knowledge_ratchet.py — Core simulation library for the S4AI / RLPE framework.

Implements the model of Yan (2026), "The Knowledge Ratchet: Physical Feedback
as an Endogenous Driver of AI-Driven Scientific Discovery" (SSRN preprint).

Model (per round t):
  checkpoint:   phi_t = Phi( (Phi^{-1}(alpha_0 * theta_t) - c*_t) / sigma_eps )
                N_t   ~ Binomial(M, phi_t),   c*_t = c* + kappa * psi_t
  validation:   p_t   = min(p_max, alpha_0 * theta_t + gamma_0 * K_t / (H + K_t))
                dK_t  ~ Binomial(N_t, p_t)
  ratchet:      K_{t+1} = K_t + dK_t                      (non-decreasing)
  RLPE update:  theta_{t+1} = min(theta_bar, theta_t + eta_0 * dK_t / N_t)
  forgetting:   w.p. lambda_CF, theta <- max((1-delta_CF) theta, (1-delta_CF) theta_0)
  pruning:      psi <- min(psi_bar, psi + zeta_0 (N_t - dK_t)/N_t)   (optional)

All simulators are vectorized across Monte Carlo runs and use a fresh
PCG64 generator per round (seed = seed0 * 100003 + t), which gives common
random numbers across parameter variants for clean comparisons.
"""

from dataclasses import dataclass
import numpy as np
from scipy.stats import norm

Phi = norm.cdf
Phiinv = norm.ppf


@dataclass
class Params:
    alpha_0: float = 0.65      # capability scaling
    gamma_0: float = 0.20      # experience coefficient
    eta_0: float = 0.03        # RLPE learning rate
    theta_0: float = 0.40      # initial capability
    theta_bar: float = 0.95    # physical capability ceiling
    p_max: float = 0.80        # validation success ceiling
    M: int = 50                # hypotheses per round (throughput)
    H: float = 50.0            # experience saturation constant
    c_star: float = float(Phiinv(0.26))  # checkpoint threshold
    sigma_eps: float = 1.0     # hypothesis score noise
    # forgetting
    lambda_CF: float = 0.0     # forgetting probability per round
    delta_CF: float = 0.3      # capability loss fraction on forgetting
    # pruning (failure-driven)
    zeta_0: float = 0.0        # failure learning rate
    kappa: float = 0.0         # pruning stringency (raises c*)
    psi_bar: float = 1.0       # pruning state ceiling
    # reversible knowledge stock (ablation 3)
    rho_d: float = 0.0         # per-round geometric decay of K
    rho_f: float = 0.0         # fractional K loss on forgetting events


# ---------------------------------------------------------------- core model

def run_s4ai(T=100, R=10_000, seed0=0, par=None, ret_traj=False):
    """Vectorized Monte Carlo of the S4AI/RLPE model.

    Returns dict with mean trajectories and (optionally) per-run finals.
    """
    p = par or Params()
    th = np.full(R, p.theta_0)
    K = np.zeros(R)
    psi = np.zeros(R)
    Km = np.zeros(T)
    THm = np.zeros(T)
    for t in range(T):
        rng = np.random.default_rng(seed0 * 100003 + t)
        if p.lambda_CF > 0:
            f = rng.random(R) < p.lambda_CF
            th = np.where(f, np.maximum((1 - p.delta_CF) * th,
                                        (1 - p.delta_CF) * p.theta_0), th)
            if p.rho_f > 0:
                K = np.where(f, (1 - p.rho_f) * K, K)
        cs = p.c_star + p.kappa * psi
        phi = Phi((Phiinv(np.clip(p.alpha_0 * th, 1e-9, 1 - 1e-9)) - cs)
                  / p.sigma_eps)
        N = rng.binomial(p.M, phi)
        pv = np.minimum(p.p_max,
                        p.alpha_0 * th + p.gamma_0 * K / (p.H + K))
        dK = rng.binomial(N, pv)
        K = (1 - p.rho_d) * K + dK
        inc = np.where(N > 0, p.eta_0 * dK / np.maximum(N, 1), 0.0)
        th = np.minimum(p.theta_bar, th + inc)
        if p.zeta_0 > 0 and p.kappa > 0:
            psi = np.where(N > 0,
                           np.minimum(p.psi_bar,
                                      psi + p.zeta_0 * (N - dK)
                                      / np.maximum(N, 1)),
                           psi)
        Km[t] = K.mean()
        THm[t] = th.mean()
    out = {"K_mean": Km, "theta_mean": THm, "K_final": Km[-1],
           "theta_final": THm[-1], "K_se": K.std() / np.sqrt(R)}
    if ret_traj:
        out["K_all"] = K
        out["theta_all"] = th
    return out


# ---------------------------------------------------------------- baselines

def run_nolearn(T=100, R=10_000, seed0=0, par=None):
    """Fixed-capability AI4S: theta frozen, experience bonus accrues."""
    p = par or Params()
    p.eta_0 = 0.0
    return run_s4ai(T, R, seed0, p)


def run_random(T=100, R=10_000, seed0=0, par=None):
    """Random screening: phi = 0.5, p = alpha_0 * theta_0 (no learning)."""
    p = par or Params()
    K = np.zeros(R)
    Km = np.zeros(T)
    p0 = p.alpha_0 * p.theta_0
    for t in range(T):
        rng = np.random.default_rng(seed0 * 100003 + t)
        N = rng.binomial(p.M, 0.5, R)
        K += rng.binomial(N, p0)
        Km[t] = K.mean()
    return {"K_mean": Km, "K_final": Km[-1],
            "K_se": K.std() / np.sqrt(R)}


def run_periodic(T=100, R=10_000, tau=10, seed0=0, par=None):
    """Periodic retraining: one pooled RLPE update every tau rounds.

    Delta-theta = eta_0 * (sum_w dK)/(sum_w N) applied at window end;
    theta frozen between updates (offline-retraining analogue).
    """
    p = par or Params()
    th = np.full(R, p.theta_0)
    K = np.zeros(R)
    bufK = np.zeros(R)
    bufN = np.zeros(R)
    Km = np.zeros(T)
    cnt = 0
    for t in range(T):
        rng = np.random.default_rng(seed0 * 100003 + t)
        phi = Phi((Phiinv(np.clip(p.alpha_0 * th, 1e-9, 1 - 1e-9)) - p.c_star)
                  / p.sigma_eps)
        N = rng.binomial(p.M, phi)
        pv = np.minimum(p.p_max,
                        p.alpha_0 * th + p.gamma_0 * K / (p.H + K))
        dK = rng.binomial(N, pv)
        K += dK
        bufK += dK
        bufN += N
        cnt += 1
        if cnt == tau:
            inc = np.where(bufN > 0, p.eta_0 * bufK / np.maximum(bufN, 1), 0.0)
            th = np.minimum(p.theta_bar, th + inc)
            bufK[:] = 0
            bufN[:] = 0
            cnt = 0
        Km[t] = K.mean()
    return {"K_mean": Km, "K_final": Km[-1],
            "K_se": K.std() / np.sqrt(R)}


def run_online_cl(T=100, R=10_000, lr=0.01, seed0=0, par=None):
    """Online continual learning: one projected gradient step per round."""
    p = par or Params()
    th = np.full(R, p.theta_0)
    K = np.zeros(R)
    Km = np.zeros(T)
    for t in range(T):
        rng = np.random.default_rng(seed0 * 100003 + t)
        phi = Phi((Phiinv(np.clip(p.alpha_0 * th, 1e-9, 1 - 1e-9)) - p.c_star)
                  / p.sigma_eps)
        N = rng.binomial(p.M, phi)
        pv = np.minimum(p.p_max,
                        p.alpha_0 * th + p.gamma_0 * K / (p.H + K))
        dK = rng.binomial(N, pv)
        K += dK
        phat = np.where(N > 0, dK / np.maximum(N, 1), p.alpha_0 * th)
        th = np.clip(th + lr * (phat - p.alpha_0 * th) * p.alpha_0, 0, 0.99)
        Km[t] = K.mean()
    return {"K_mean": Km, "K_final": Km[-1],
            "K_se": K.std() / np.sqrt(R)}


# ------------------------------------------- Bayesian optimization baselines
# Both optimize the checkpoint-threshold offset x in [-1.5, 1.5] with
# capability frozen at theta_0 (selection-policy optimization without
# capability learning).

def _bo_step(x, th, K, rng, p):
    phi = Phi((Phiinv(np.clip(p.alpha_0 * th, 1e-9, 1 - 1e-9)) - p.c_star - x)
              / p.sigma_eps)
    N = rng.binomial(p.M, phi)
    pv = np.minimum(p.p_max, p.alpha_0 * th + p.gamma_0 * K / (p.H + K))
    return rng.binomial(N, pv)


def run_gp_ucb(T=100, R=200, seed0=0, beta=2.0, ls=1.0, par=None,
               ret_traj=False):
    """GP-UCB over the threshold offset (RBF kernel, noise var M/4)."""
    p = par or Params()
    xs = np.linspace(-1.5, 1.5, 60)
    sn2 = p.M * 0.25
    Kf = np.zeros(R)
    KT = np.zeros((R, T))
    for r in range(R):
        rng = np.random.default_rng(seed0 + r)
        th = p.theta_0
        K = 0.0
        X, Y = [], []
        for t in range(T):
            if len(X) < 3:
                xi = xs[rng.integers(60)]
            else:
                Xa, Ya = np.array(X), np.array(Y)
                Kg = np.exp(-(xs[:, None] - Xa[None, :]) ** 2 / (2 * ls ** 2))
                Kxx = (np.exp(-(Xa[:, None] - Xa[None, :]) ** 2 / (2 * ls ** 2))
                       + sn2 * np.eye(len(Xa)))
                A = np.linalg.solve(Kxx, Ya)
                B = np.linalg.solve(Kxx, Kg.T)
                mu = Kg @ A
                sd = np.sqrt(np.maximum(1e-12, 1.0 - np.sum(Kg * B.T, axis=1)))
                xi = xs[np.argmax(mu + np.sqrt(beta) * sd)]
            y = _bo_step(xi, th, K, rng, p)
            K += y
            X.append(xi)
            Y.append(y)
            KT[r, t] = K
        Kf[r] = K
    out = {"K_final": Kf.mean(), "K_se": Kf.std() / np.sqrt(R)}
    if ret_traj:
        out["K_mean"] = KT.mean(0)
    return out


def run_turbo1(T=100, R=200, seed0=0, L0=0.4, ls=1.0, par=None,
               ret_traj=False):
    """TuRBO-1 [Eriksson et al., 2019]: GP + Thompson sampling in an
    adaptive trust region, over the threshold offset."""
    p = par or Params()
    xs = np.linspace(-1.5, 1.5, 60)
    sn2 = p.M * 0.25
    Kf = np.zeros(R)
    KT = np.zeros((R, T))
    for r in range(R):
        rng = np.random.default_rng(seed0 + r)
        th = p.theta_0
        K = 0.0
        X, Y = [], []
        L, xb, yb = L0, 0.0, -np.inf
        fails = succs = 0
        for t in range(T):
            if len(X) < 5:
                xi = np.clip(xb + rng.uniform(-L, L), -1.5, 1.5)
            else:
                Xa, Ya = np.array(X), np.array(Y)
                Ys = (Ya - Ya.mean()) / (Ya.std() + 1e-9)
                cand = xs[(xs >= xb - L) & (xs <= xb + L)]
                if len(cand) < 3:
                    cand = xs
                Kg = np.exp(-(cand[:, None] - Xa[None, :]) ** 2 / (2 * ls ** 2))
                Kxx = (np.exp(-(Xa[:, None] - Xa[None, :]) ** 2 / (2 * ls ** 2))
                       + (sn2 / (Ya.var() + 1e-9)) * np.eye(len(Xa)))
                Kgg = np.exp(-(cand[:, None] - cand[None, :]) ** 2 / (2 * ls ** 2))
                A = np.linalg.solve(Kxx, Ys)
                B = np.linalg.solve(Kxx, Kg.T)
                mu = Kg @ A
                S = Kgg - Kg @ B
                S = (S + S.T) / 2 + 1e-10 * np.eye(len(cand))
                xi = cand[np.argmax(mu + np.linalg.cholesky(S)
                                    @ rng.standard_normal(len(cand)))]
            y = _bo_step(xi, th, K, rng, p)
            K += y
            X.append(xi)
            Y.append(y)
            KT[r, t] = K
            if y >= yb:
                yb, xb = y, xi
                succs += 1
                fails = 0
            else:
                fails += 1
                succs = 0
            if succs >= 3:
                L = min(1.0, L * 1.6)
                succs = 0
            if fails >= 5:
                L *= 0.5
                fails = 0
            if L < 0.02:
                L = L0
        Kf[r] = K
    out = {"K_final": Kf.mean(), "K_se": Kf.std() / np.sqrt(R)}
    if ret_traj:
        out["K_mean"] = KT.mean(0)
    return out


def run_oracle_threshold(T=100, R=10_000, x=-1.5, seed0=0, par=None):
    """Oracle capability-frozen policy: best fixed threshold offset.

    An upper bound on threshold-selection policies that do not learn
    (Theorem 3(iii) of the paper).
    """
    p = par or Params()
    K = np.zeros(R)
    Km = np.zeros(T)
    for t in range(T):
        rng = np.random.default_rng(seed0 * 100003 + t)
        # vectorized equivalent of _bo_step at fixed x
        phi = Phi((Phiinv(p.alpha_0 * p.theta_0) - p.c_star - x) / p.sigma_eps)
        N = rng.binomial(p.M, phi, R)
        pv = np.minimum(p.p_max, p.alpha_0 * p.theta_0
                        + p.gamma_0 * K / (p.H + K))
        K += rng.binomial(N, pv)
        Km[t] = K.mean()
    return {"K_mean": Km, "K_final": Km[-1],
            "K_se": K.std() / np.sqrt(R)}
