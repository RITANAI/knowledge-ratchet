# Data provenance

## alab_success_rates.csv

Round-level success-rate series (41 rounds) for the A-Lab calibration
pipeline (`calibrate_alab.py`).

**Important:** the published A-Lab record (Szymanski et al., *Nature* 2023)
reports campaign aggregates — 58 targeted inorganic compounds, 41
XRD-confirmed successes, 17 days of continuous operation — but does not
publish round-level logs. This CSV is a monotone reconstruction constrained
to those aggregates (regenerate deterministically with
`reconstruct_alab.py`) and is provided as **A-Lab-consistent demonstration
data** so the calibration pipeline (nonlinear least squares, LOOCV,
parametric bootstrap) can be run end to end. It is not a digitization of
laboratory logs. To calibrate to real operations data, replace this file
with your own round-level series using the same columns (`round`,
`success_rate`).

## Molecular demonstration

`validate_guacamol.py` needs no data file: it either runs the synthetic
closed-loop demonstration (default; the mode reported in §2.8 of the paper)
or the GuacaMol-backed analogue (`--guacamol`, requires `rdkit`,
`guacamol`, `scikit-learn`), which fetches GuacaMol SMILES at runtime.
