"""
calibrate_alab.py — Calibrate S4AI parameters to the A-Lab campaign.

Fits (alpha_0, gamma_0, eta_0) by nonlinear least squares to the
round-level success-rate series in data/alab_success_rates.csv, with
leave-one-out cross-validation and a parametric (residual) bootstrap
(B = 1,000). The validation ceiling p_max is held fixed at 0.82: the
round-level series tops out near 0.71, below the ceiling, so p_max is
not separately identifiable from these data and is adopted from the
baseline parameterization of the paper.

DATA PROVENANCE (important): the published A-Lab record (Szymanski et al.,
Nature 2023) reports campaign aggregates — 58 targeted inorganic compounds,
41 XRD-confirmed successes, 17 days of continuous operation — but not
round-level logs. The CSV shipped here is a monotone reconstruction
constrained to those aggregates (see data/README.md and
reconstruct_alab.py) and is intended as A-Lab-consistent demonstration
data for the calibration pipeline. To calibrate to laboratory logs,
replace the CSV with your own round-level series (same columns:
round, success_rate).

Usage:
    python calibrate_alab.py --data data/alab_success_rates.csv --out results/

Paper values (Section 2.8) reproduced by this script on the shipped CSV:
    alpha_0 = 0.58, gamma_0 = 0.25, eta_0 = 0.024, R^2 = 0.74,
    LOOCV R^2 = 0.67; parametric-bootstrap 95% CIs:
    alpha_0 [0.41, 0.78], gamma_0 [0.01, 0.40], eta_0 [0.013, 0.042].
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

P_MAX_FIXED = 0.82  # validation ceiling (see module docstring)


def simulate_rates(params, n_rounds, theta_0=0.4, theta_bar=0.95,
                   M=50, H=50.0, p_max=P_MAX_FIXED):
    """Mean-field success-rate trajectory of the S4AI model."""
    alpha_0, gamma_0, eta_0 = params
    theta = theta_0
    K = 0.0
    rates = np.zeros(n_rounds)
    for t in range(n_rounds):
        p = min(p_max, alpha_0 * theta + gamma_0 * K / (H + K))
        N = M * 0.5  # mean checkpoint count at phi ~ 0.5 (see paper Eq. 2)
        dK = N * p
        rates[t] = p
        K += dK
        theta = min(theta_bar, theta + eta_0 * dK / N)
    return rates


def fit(observed, **kw):
    def resid(params):
        return simulate_rates(params, len(observed), **kw) - observed
    p0 = [0.6, 0.2, 0.03]
    bounds = ([0.1, 0.0, 0.001], [1.0, 0.5, 0.1])
    res = least_squares(resid, p0, bounds=bounds, method="trf", max_nfev=10000)
    pred = simulate_rates(res.x, len(observed), **kw)
    ss_res = float(np.sum((observed - pred) ** 2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    return res.x, 1 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/alab_success_rates.csv")
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    observed = df["success_rate"].to_numpy()
    n = len(observed)
    print(f"Loaded {n} rounds from {args.data}")
    print(f"(p_max held fixed at {P_MAX_FIXED}; series max "
          f"{observed.max():.3f} is below the ceiling, so p_max is not "
          f"separately identifiable)")

    params, r2 = fit(observed)
    names = ["alpha_0", "gamma_0", "eta_0"]
    print("\n=== Point estimates (nonlinear least squares) ===")
    for nm, v in zip(names, params):
        print(f"{nm:<8} {v:.3f}")
    print(f"R^2      {r2:.3f}")

    # LOOCV
    r2_cv = []
    for i in range(n):
        mask = np.ones(n, bool)
        mask[i] = False
        try:
            pi, _ = fit(observed[mask])
            pred_i = simulate_rates(pi, n)[i]
            r2_cv.append((observed[i] - pred_i) ** 2)
        except Exception:
            pass
    r2_cv = 1 - np.sum(r2_cv) / np.sum((observed - observed.mean()) ** 2)
    print(f"\nLOOCV cross-validated R^2 = {r2_cv:.3f}")

    # Parametric (residual) bootstrap: refit on prediction + resampled
    # residuals, preserving the round ordering of the series.
    pred = simulate_rates(params, n)
    resid = observed - pred
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(args.bootstrap):
        ob = np.clip(pred + rng.choice(resid, n, replace=True), 0.01, 0.99)
        try:
            boot.append(fit(ob)[0])
        except Exception:
            pass
    boot = np.array(boot)
    print(f"\n=== Parametric bootstrap (B={len(boot)}, residual resampling) ===")
    out = {"n_rounds": int(n), "p_max_fixed": P_MAX_FIXED,
           "r_squared": float(r2), "r2_loocv": float(r2_cv)}
    for j, nm in enumerate(names):
        lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
        se = float(boot[:, j].std())
        out[nm] = {"point": float(params[j]), "median": float(np.median(boot[:, j])),
                   "se": se, "ci95": [float(lo), float(hi)]}
        print(f"{nm:<8} point {params[j]:.3f}  median {np.median(boot[:, j]):.3f}  "
              f"SE {se:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    Path(args.out).mkdir(exist_ok=True)
    with open(Path(args.out) / "calibration_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved {args.out}/calibration_results.json")


if __name__ == "__main__":
    main()
