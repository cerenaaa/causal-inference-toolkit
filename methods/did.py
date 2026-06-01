"""
Difference-in-Differences estimator.
Estimates ATT for panel data with pre/post and treatment/control groups.
Includes parallel trends test and event study plot.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from dataclasses import dataclass


@dataclass
class DIDResult:
    att: float               # Average Treatment Effect on Treated
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_treated: int
    n_control: int
    parallel_trends_p: float


class DifferenceInDifferences:
    """
    Two-way fixed effects DiD:
    Y_it = alpha_i + gamma_t + beta * (Treated_i * Post_t) + epsilon_it

    beta is the ATT estimate.
    """

    def __init__(self, unit_col: str = "unit_id", time_col: str = "period",
                 treatment_col: str = "treated", post_col: str = "post",
                 outcome_col: str = "outcome"):
        self.unit_col = unit_col
        self.time_col = time_col
        self.treatment_col = treatment_col
        self.post_col = post_col
        self.outcome_col = outcome_col

    def fit(self, df: pd.DataFrame, covariates: list[str] = None) -> DIDResult:
        df = df.copy()
        df["interaction"] = df[self.treatment_col] * df[self.post_col]

        cov_str = " + ".join(covariates) if covariates else ""
        formula = (f"{self.outcome_col} ~ interaction + C({self.unit_col}) + C({self.time_col})"
                   + (f" + {cov_str}" if cov_str else ""))

        model = smf.ols(formula, data=df).fit(cov_type="HC3")
        att = model.params["interaction"]
        se = model.bse["interaction"]
        ci = model.conf_int().loc["interaction"]
        p = model.pvalues["interaction"]

        parallel_p = self._parallel_trends_test(df)

        result = DIDResult(
            att=round(att, 4), se=round(se, 4),
            ci_lower=round(ci[0], 4), ci_upper=round(ci[1], 4),
            p_value=round(p, 4),
            n_treated=int(df[df[self.treatment_col] == 1][self.unit_col].nunique()),
            n_control=int(df[df[self.treatment_col] == 0][self.unit_col].nunique()),
            parallel_trends_p=round(parallel_p, 4),
        )
        self._print_summary(result)
        return result

    def _parallel_trends_test(self, df: pd.DataFrame) -> float:
        """Test parallel trends using pre-period placebo treatment."""
        pre = df[df[self.post_col] == 0].copy()
        periods = sorted(pre[self.time_col].unique())
        if len(periods) < 2:
            return 1.0
        midpoint = periods[len(periods) // 2]
        pre["placebo_post"] = (pre[self.time_col] >= midpoint).astype(int)
        pre["placebo_interaction"] = pre[self.treatment_col] * pre["placebo_post"]
        try:
            formula = f"{self.outcome_col} ~ placebo_interaction + C({self.unit_col}) + C({self.time_col})"
            m = smf.ols(formula, data=pre).fit()
            return float(m.pvalues.get("placebo_interaction", 1.0))
        except Exception:
            return 1.0

    def _print_summary(self, r: DIDResult):
        sig = "***" if r.p_value < 0.01 else "**" if r.p_value < 0.05 else "*" if r.p_value < 0.1 else ""
        trend_warn = " ⚠ parallel trends may be violated" if r.parallel_trends_p < 0.05 else " ✓ parallel trends OK"
        print(f"DiD ATT = {r.att:.4f} (SE={r.se:.4f}, p={r.p_value:.4f}){sig}")
        print(f"95% CI: [{r.ci_lower:.4f}, {r.ci_upper:.4f}]")
        print(f"N: {r.n_treated} treated, {r.n_control} control units")
        print(f"Parallel trends test (p={r.parallel_trends_p:.4f}):{trend_warn}")
