"""
simulate.py — Reproduce Table 2 (multi-baseline comparison) and the
forgetting / ablation headline numbers of Yan (2026), "The Knowledge Ratchet".

Usage:
    python simulate.py                 # full run (Table 2 + forgetting + ablations)
    python simulate.py --quick         # reduced run counts for a fast smoke test
    python simulate.py --out results/  # write JSON summary

Expected output (full run, paper values):
    S4AI (full model)        2,812  (SE 0.5)
    Oracle threshold policy  2,046
    TuRBO-1                  1,919  (SE 6.2)
    GP-UCB                   1,472  (SE 24.6)
    Periodic (tau=10)        1,275
    Online CL                1,239
    No learning              1,067
    Random                     650
"""

import argparse
import json
from pathlib import Path

import numpy as np

from knowledge_ratchet import (Params, run_s4ai, run_nolearn, run_random,
                               run_periodic, run_online_cl, run_gp_ucb,
                               run_turbo1, run_oracle_threshold)


def table2(quick=False):
    R = 1_000 if quick else 10_000
    Rbo = 50 if quick else 200
    rows = {}
    rows["S4AI"] = run_s4ai(R=R, ret_traj=True)
    rows["Oracle threshold"] = run_oracle_threshold(R=R)
    rows["TuRBO-1"] = run_turbo1(R=Rbo)
    rows["GP-UCB"] = run_gp_ucb(R=Rbo)
    rows["Periodic (tau=10)"] = run_periodic(R=R)
    rows["Online CL"] = run_online_cl(R=R)
    rows["No learning"] = run_nolearn(R=R)
    rows["Random"] = run_random(R=R)
    return rows


def forgetting(quick=False):
    T, R = (300, 500) if quick else (1000, 2000)
    base = run_s4ai(T=T, R=R)
    forg = run_s4ai(T=T, R=R, par=Params(lambda_CF=0.035, delta_CF=0.3))
    lo = slice(T // 2, T)
    s_b = np.polyfit(np.arange(T // 2, T), base["K_mean"][lo], 1)[0]
    s_f = np.polyfit(np.arange(T // 2, T), forg["K_mean"][lo], 1)[0]
    return {"slope_no_forgetting": float(s_b), "slope_forgetting": float(s_f),
            "degradation_pct": float(100 * (1 - s_f / s_b))}


def ablations(quick=False):
    R = 1_000 if quick else 10_000
    out = {}
    base = run_s4ai(R=R)["K_final"]
    out["baseline"] = base
    for kap in (0.1, 0.2):
        p = Params(zeta_0=0.05, kappa=kap)
        out[f"pruning_kappa_{kap}"] = run_s4ai(R=R, par=p)["K_final"]
    for rd in (0.01, 0.05, 0.10, 0.20):
        p = Params(rho_d=rd)
        out[f"reversible_rho_d_{rd}"] = run_s4ai(R=R, par=p)["K_final"]
    p = Params(lambda_CF=0.035, delta_CF=0.3, rho_d=0.05, rho_f=0.3)
    rev = run_s4ai(R=R, par=p)
    out["reversible_under_forgetting"] = rev["K_final"]
    # late-window net rates (paper Ablation 3: 28.0 vs 2.2 discoveries/round)
    irr = run_s4ai(R=R, par=Params(lambda_CF=0.035, delta_CF=0.3))
    lo = slice(50, 100)
    t = np.arange(50, 100)
    out["rate_irreversible_under_forgetting"] = float(
        np.polyfit(t, irr["K_mean"][lo], 1)[0])
    out["rate_reversible_under_forgetting"] = float(
        np.polyfit(t, rev["K_mean"][lo], 1)[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    print("=== Table 2: relative productivity (mean K_100) ===")
    rows = table2(args.quick)
    s4 = rows["S4AI"]["K_final"]
    for name, r in rows.items():
        se = r.get("K_se", float("nan"))
        print(f"{name:<22} {r['K_final']:>8.1f}  (SE {se:5.2f})  "
              f"S4AI advantage {s4 / r['K_final']:.2f}x")

    print("\n=== Forgetting (lambda_CF=0.035, delta_CF=0.3) ===")
    f = forgetting(args.quick)
    print(f"slope {f['slope_forgetting']:.2f} vs {f['slope_no_forgetting']:.2f} "
          f"discoveries/round ({f['degradation_pct']:.1f}% degradation)")

    print("\n=== Ablations (mean K_100) ===")
    ab = ablations(args.quick)
    for k, v in ab.items():
        print(f"{k:<36} {v:>8.1f}")

    if args.out:
        out = Path(args.out)
        out.mkdir(exist_ok=True)
        def slim(d):
            return {k: (float(v) if np.isscalar(v) else None)
                    for k, v in d.items()}
        payload = {
            "table2": {k: slim(v) for k, v in rows.items()},
            "forgetting": f,
            "ablations": ab,
        }
        with open(out / "simulation_results.json", "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nSaved {out / 'simulation_results.json'}")


if __name__ == "__main__":
    main()
