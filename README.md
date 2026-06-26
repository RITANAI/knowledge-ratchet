# The Knowledge Ratchet: SAPE Simulation Code

[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/XXXX.XXXXX)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the simulation code, calibration procedures, validation scripts, and analysis notebooks for the paper:

> **The Knowledge Ratchet: A Physical-Constraint-Driven Learning Architecture from Epistemological First Principles**  
> Zepeng Yan, RiTan Regeneration (Beijing) Biotechnology Co., Ltd.  
> *Correspondence: zp@ritanai.com*

## Overview

The **Knowledge Ratchet** is an architectural principle for AI systems that interact with the physical world. It separates:

- **Ephemeral capability** (theta_t): a running statistic that may reset
- **Permanent knowledge** (K_t): an irreversible accumulation of validated discoveries

**SAPE** (Stochastic Approximation from Physical Evidence) is the scalar update mechanism that implements this principle.

> **⚠️ Disclaimer**: SAPE operates under a *known parametric structure* (Eq. 1 in the paper). The expectation advantage in Theorem 3 and performance comparisons assume this structural knowledge. SAPE is **not** designed to replace black-box Bayesian optimization in unstructured spaces; it is a structural complement.

## Repository Structure

```
.
├── sapelib.py              # Core SAPE simulation library
├── simulate.py             # Main simulation script (Figures 2-5)
├── calibrate.py            # A-Lab data calibration
├── validate.py             # GuacaMol validation
├── analyze.py              # Analysis and plotting utilities
├── requirements.txt        # Python dependencies
├── data/                   # Data directory
│   └── alab_success_rates.csv
├── notebooks/              # Jupyter notebooks
│   └── reproduction.ipynb
├── figures/                # Generated figures
└── README.md               # This file
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/RITANAI/knowledge-ratchet.git
cd knowledge-ratchet

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Simulations

```bash
# Generate all figures (10,000 Monte Carlo runs)
python simulate.py --figure all --n_runs 10000 --output_dir figures/

# Generate specific figure
python simulate.py --figure 2  # Parameter sensitivity
python simulate.py --figure 3  # Multi-baseline comparison
python simulate.py --figure 4  # Catastrophic forgetting
python simulate.py --figure 5  # Ablation studies
```

### Calibrate to A-Lab Data

```bash
# Using built-in illustrative data
python calibrate.py --output results/

# Using custom data
python calibrate.py --data data/alab_success_rates.csv --output results/
```

### Validate on GuacaMol

```bash
python validate.py --data data/guacamol_trajectory.csv --output results/
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha_0` | 0.65 | Capability scaling factor |
| `gamma_0` | 0.20 | Experience-driven learning rate |
| `eta_0` | 0.03 | SAPE learning rate |
| `theta_0` | 0.40 | Initial capability index |
| `theta_bar` | 0.95 | Physical ceiling |
| `p_max` | 0.80 | Maximum validation success probability |
| `M` | 50 | Batch size (hypotheses per round) |
| `H` | 50.0 | Knowledge saturation parameter |
| `lambda_CF` | 0.0 | Catastrophic forgetting probability |
| `delta_CF` | 0.0 | Forgetting loss fraction |

## Computational Requirements

- **OS**: Linux/macOS/Windows
- **Python**: 3.11+
- **RAM**: ~2GB for 10,000 MC runs
- **Time**: ~5 minutes for full Figure 3 (10,000 runs), ~30 seconds for other figures

## Reproducing Paper Results

| Figure | Command | Runtime |
|--------|---------|---------|
| Fig. 2 | `python simulate.py --figure 2 --n_runs 1000` | ~2 min |
| Fig. 3 | `python simulate.py --figure 3 --n_runs 10000` | ~5 min |
| Fig. 4 | `python simulate.py --figure 4 --n_runs 1000` | ~30 sec |
| Fig. 5 | `python simulate.py --figure 5 --n_runs 5000` | ~3 min |
| A-Lab | `python calibrate.py` | ~10 sec |

## Citation

```bibtex
@article{yan2026knowledge,
  title={The Knowledge Ratchet: A Physical-Constraint-Driven Learning Architecture from Epistemological First Principles},
  author={Yan, Zepeng},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

This research was funded by RiTan Regeneration (Beijing) Biotechnology Co., Ltd.

The A-Lab calibration data is derived from Szymanski et al., *Nature* 624, 86-91 (2023).
