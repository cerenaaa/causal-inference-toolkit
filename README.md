# Causal Inference Toolkit

[![CI](https://github.com/cerenaaa/causal-inference-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/cerenaaa/causal-inference-toolkit/actions)

Production-ready causal inference methods for observational data: Difference-in-Differences, Instrumental Variables, Propensity Score Matching, and Synthetic Control. Built for business analysts who need rigorous causal estimates, not just correlations.

## Methods

| Method | Use case | Key assumption |
|---|---|---|
| **DiD** | Policy/experiment rollouts with control group | Parallel trends |
| **IV** | Endogenous treatment with a valid instrument | Relevance + exclusion |
| **PSM** | Matching treated/control on observables | No unmeasured confounding |
| **Synthetic Control** | Single treated unit, many controls | Pre-treatment fit |
| **CUPED** | Variance reduction in A/B tests | Pre-experiment covariate |

## Structure
```
causal-inference-toolkit/
├── methods/
│   ├── did.py               # Difference-in-Differences + parallel trends test
│   ├── iv.py                # Two-Stage Least Squares (2SLS)
│   ├── psm.py               # Propensity Score Matching + balance diagnostics
│   └── synthetic_control.py # Synthetic Control Method
├── diagnostics/
│   └── balance.py           # Covariate balance, SMD, love plots
├── data/
│   └── synthetic_experiments.py
└── run_demo.py
```

## Quickstart
```bash
pip install -r requirements.txt
python run_demo.py
```
