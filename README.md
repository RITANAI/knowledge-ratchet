# The Knowledge Ratchet

**Companion code for:** Yan, Z. (2026). *The Knowledge Ratchet: Physical Feedback as an Endogenous Driver of AI-Driven Scientific Discovery — A Unified Stochastic Framework with Reinforcement Learning from Physical Evidence.* SSRN working paper, RiTan Regeneration (Beijing) Biotechnology Co., Ltd.

---

## What this is

Science-for-AI (S4AI) is a closed-loop paradigm in which physical experimental outcomes feed back into the hypothesis-generating model every round. This repository contains the complete, runnable implementation of the stochastic framework:

- the **S4AI/RLPE simulator** (vectorized Monte Carlo, 10,000 runs in seconds);
- all **seven baselines** (periodic retraining, online continual learning, TuRBO-1, GP-UCB, no-learning, random, and the oracle threshold policy);
- the **forgetting, pruning, and reversibility** experiments (Figs. 4–5);
- **figure scripts** reproducing Figures 1–6 of the paper;
- the **A-Lab calibration** pipeline (NLS + LOOCV + parametric bootstrap);
- the **closed-loop molecular demonstration** (synthetic mode reproduces §2.8 exactly; optional GuacaMol-backed mode).

## Quickstart

```bash
git clone https://github.com/RITANAI/knowledge-ratchet.git
cd knowledge-ratchet
pip install -r requirements.txt
python simulate.py --quick        # 30-second smoke test
python simulate.py                # full Table 2 (~1-2 min)
python figures.py                 # regenerate all six figures (figures/)
```

Or run the quickstart notebook directly in your browser:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RITANAI/knowledge-ratchet/blob/main/notebooks/quickstart.ipynb)

## Headline results (reproduced by `simulate.py`)

| Configuration | Mean K₁₀₀ | S4AI advantage |
|---|---:|---:|
| **S4AI (full model)** | **2,812** | — |
| Oracle threshold policy (best fixed offset) | 2,046 | 1.37× |
| TuRBO-1 Bayesian optimization | 1,919 | 1.47× |
| GP-UCB Bayesian optimization | 1,472 | 1.91× |
| Periodic retraining (τ = 10) | 1,275 | 2.21× |
| Online continual learning | 1,239 | 2.27× |
| No learning (fixed capability) | 1,067 | 2.64× |
| Random screening | 650 | 4.33× |

S4AI exceeds the *oracle ceiling* of every capability-frozen threshold-selection policy by 37%: selection-policy optimization cannot substitute for endogenous capability learning (Theorem 3(iii)).

## Repository map

| File | Reproduces |
|---|---|
| `knowledge_ratchet.py` | Core library: model, all baselines, ablation options |
| `simulate.py` | Table 2; forgetting slope degradation (−12.4%); Ablation 2/3 numbers |
| `figures.py` | Figures 1–6 (PDF, vector) |
| `calibrate_alab.py` | Section 2.8 calibration pipeline (NLS, LOOCV, bootstrap) |
| `reconstruct_alab.py` | Regenerates `data/alab_success_rates.csv` (see provenance note) |
| `validate_guacamol.py` | §2.8 molecular closed-loop demonstration (synthetic mode = paper numbers) |
| `data/alab_success_rates.csv` | Round-level series for calibration (see provenance note below) |
| `reproduce_all.sh` | One-command reproduction of all numeric results |

## Data provenance

The published A-Lab record (Szymanski et al., *Nature* 2023) reports campaign aggregates — 58 targets, 41 XRD-confirmed successes, 17 days — but not round-level logs. `data/alab_success_rates.csv` is a monotone reconstruction constrained to those aggregates (regenerate with `reconstruct_alab.py`), provided as A-Lab-consistent demonstration data for the calibration pipeline — not a digitization of laboratory logs. The paper's §2.8 reports exactly what `calibrate_alab.py` outputs on this file (α₀ = 0.58, γ₀ = 0.25, η₀ = 0.024, R² = 0.74, LOOCV R² = 0.67). To calibrate against laboratory logs, drop your own `round,success_rate` CSV into `data/` and rerun `calibrate_alab.py`.

## Model in one paragraph

Each round, the generative model (capability θ_t) proposes M hypotheses; a physical checkpoint passes N_t ~ Binomial(M, φ_t) of them, where φ_t = Φ((Φ⁻¹(α₀θ_t) − c*)/σ_ε); each passing hypothesis is validated with probability p_t = min{p_max, α₀θ_t + γ₀K_t/(H+K_t)}, yielding ΔK_t confirmed discoveries. The knowledge stock ratchets: K_{t+1} = K_t + ΔK_t (non-decreasing). RLPE closes the loop: θ_{t+1} = min{θ̄, θ_t + η₀ ΔK_t/N_t}. Catastrophic forgetting resets θ stochastically but cannot erode K.

## Citation

```bibtex
@misc{yan2026knowledgeratchet,
  author = {Yan, Zepeng},
  title  = {The Knowledge Ratchet: Physical Feedback as an Endogenous Driver
            of AI-Driven Scientific Discovery},
  year   = {2026},
  note   = {SSRN working paper},
  url    = {https://github.com/RITANAI/knowledge-ratchet}
}
```

## License

MIT (see `LICENSE`). Code only; the manuscript is © the author.
