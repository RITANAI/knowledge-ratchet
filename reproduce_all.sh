#!/usr/bin/env bash
# reproduce_all.sh — one-command reproduction of all numeric results and figures
# for Yan (2026), "The Knowledge Ratchet".
set -euo pipefail

echo "== [1/4] Monte Carlo simulations (Table 2, forgetting, ablations) =="
python simulate.py --out results/

echo "== [2/4] Figures 1-6 =="
python figures.py --out figures/

echo "== [3/4] A-Lab calibration (NLS + LOOCV + bootstrap) =="
python reconstruct_alab.py
python calibrate_alab.py --data data/alab_success_rates.csv --out results/

echo "== [4/4] Molecular closed-loop demonstration (Section 2.8) =="
python validate_guacamol.py --synthetic --out results/

echo "Done. See results/ and figures/."
