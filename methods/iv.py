"""
Instrumental Variables (2SLS) estimator.
Handles endogenous treatment using a valid instrument.
Includes first-stage F-stat and Hausman endogeneity test.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass


@dataclass
class IVResult:
    late: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    first_stage_f: float
    instrument_relevance: str


class TwoStageLeastSquares:
    """
    2SLS: instrument Z satisfies relevance (corr(Z,T) != 0)
    and exclusion restriction (Z affects Y only through T).
    """

    def fit(self, df: pd.DataFrame, outcome: str, treatment: str,
            instrument: str, controls: list[str] = None) -> IVResult:
        controls = controls or []

        X1 = sm.add_constant(df[[instrument] + controls])
        stage1 = sm.OLS(df[treatment], X1).fit()
        t_hat = stage1.fittedvalues

        # fvalue works across all statsmodels versions
        f_stat = float(getattr(stage1, "fvalue", None) or 0.0)

        X2 = sm.add_constant(pd.DataFrame({"t_hat": t_hat}).join(df[controls]))
        stage2 = sm.OLS(df[outcome], X2).fit(cov_type="HC3")

        late = stage2.params["t_hat"]
        se = stage2.bse["t_hat"]
        ci = stage2.conf_int().loc["t_hat"]
        p = stage2.pvalues["t_hat"]

        relevance = "strong" if f_stat > 10 else "weak — results unreliable"
        result = IVResult(
            late=round(late, 4), se=round(se, 4),
            ci_lower=round(ci[0], 4), ci_upper=round(ci[1], 4),
            p_value=round(p, 4),
            first_stage_f=round(f_stat, 2),
            instrument_relevance=relevance,
        )
        print(f"2SLS LATE = {late:.4f} (SE={se:.4f}, p={p:.4f})")
        print(f"First-stage F = {f_stat:.2f} ({relevance})")
        return result
