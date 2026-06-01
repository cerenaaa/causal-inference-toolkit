"""Synthetic experiment datasets for causal inference demos."""
import numpy as np
import pandas as pd


def generate_panel_data(n_units: int = 200, n_periods: int = 10,
                        treatment_effect: float = 5.0, seed: int = 42) -> pd.DataFrame:
    """Panel dataset for DiD. Treatment occurs at period 5 for half the units."""
    rng = np.random.default_rng(seed)
    records = []
    treated_units = rng.choice(n_units, size=n_units // 2, replace=False)
    for unit in range(n_units):
        is_treated = unit in treated_units
        unit_fe = rng.normal(0, 2)
        for t in range(n_periods):
            post = int(t >= n_periods // 2)
            trend = 0.3 * t
            y = 10 + unit_fe + trend + rng.normal(0, 1)
            if is_treated and post:
                y += treatment_effect
            records.append(dict(unit_id=unit, period=t, treated=int(is_treated),
                                post=post, outcome=round(y, 3)))
    df = pd.DataFrame(records)
    print(f"Panel data: {n_units} units x {n_periods} periods | true ATT = {treatment_effect}")
    return df


def generate_observational_data(n: int = 2000, treatment_effect: float = 3.0,
                                 seed: int = 42) -> pd.DataFrame:
    """Observational data with confounding for PSM/IV demo."""
    rng = np.random.default_rng(seed)
    age = rng.integers(25, 65, n)
    income = rng.lognormal(10.5, 0.5, n)
    propensity = 1 / (1 + np.exp(-(0.02 * age + 0.0001 * income - 3)))
    treated = rng.binomial(1, propensity)
    y = 5 + 0.05 * age + 0.0002 * income + treatment_effect * treated + rng.normal(0, 2, n)
    df = pd.DataFrame(dict(age=age, income=income.round(0), treated=treated, outcome=y.round(3)))
    print(f"Obs data: n={n} | true ATT = {treatment_effect} | treatment rate = {treated.mean():.1%}")
    return df
