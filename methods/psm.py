"""
Propensity Score Matching.
Estimates ATT by matching treated units to similar controls on P(T=1|X).
Includes balance diagnostics and nearest-neighbor or caliper matching.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass


@dataclass
class PSMResult:
    att: float
    se: float
    n_matched_treated: int
    n_matched_control: int
    pre_match_smd: float    # Avg standardized mean difference before matching
    post_match_smd: float   # Avg SMD after matching (should be < 0.1)


class PropensityScoreMatching:
    def __init__(self, caliper: float = 0.05, n_neighbors: int = 1, random_state: int = 42):
        self.caliper = caliper
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.propensity_model = LogisticRegression(max_iter=1000, random_state=random_state)
        self.scaler = StandardScaler()

    def _smd(self, treated: pd.DataFrame, control: pd.DataFrame, cols: list[str]) -> float:
        smds = []
        for col in cols:
            t_mean, c_mean = treated[col].mean(), control[col].mean()
            pooled_std = np.sqrt((treated[col].var() + control[col].var()) / 2)
            if pooled_std > 0:
                smds.append(abs(t_mean - c_mean) / pooled_std)
        return float(np.mean(smds)) if smds else 0.0

    def fit(self, df: pd.DataFrame, treatment_col: str, outcome_col: str,
            covariate_cols: list[str]) -> PSMResult:
        X = self.scaler.fit_transform(df[covariate_cols])
        y = df[treatment_col].values

        self.propensity_model.fit(X, y)
        df = df.copy()
        df["propensity_score"] = self.propensity_model.predict_proba(X)[:, 1]

        treated = df[df[treatment_col] == 1].copy()
        control = df[df[treatment_col] == 0].copy()

        pre_smd = self._smd(treated, control, covariate_cols)

        # Nearest-neighbor matching with caliper
        nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="euclidean")
        nn.fit(control[["propensity_score"]])
        dists, indices = nn.kneighbors(treated[["propensity_score"]])

        matched_control_idx = []
        valid_treated_idx = []
        for i, (dist_row, idx_row) in enumerate(zip(dists, indices)):
            if dist_row[0] <= self.caliper:
                matched_control_idx.append(control.iloc[idx_row[0]].name)
                valid_treated_idx.append(treated.iloc[i].name)

        if not valid_treated_idx:
            raise ValueError(f"No matches within caliper={self.caliper}. Try increasing it.")

        matched_treated = df.loc[valid_treated_idx]
        matched_control = df.loc[matched_control_idx]
        post_smd = self._smd(matched_treated, matched_control, covariate_cols)

        att = float(matched_treated[outcome_col].mean() - matched_control[outcome_col].mean())
        se = float(np.sqrt(matched_treated[outcome_col].var() / len(matched_treated) +
                           matched_control[outcome_col].var() / len(matched_control)))

        result = PSMResult(
            att=round(att, 4), se=round(se, 4),
            n_matched_treated=len(valid_treated_idx),
            n_matched_control=len(matched_control_idx),
            pre_match_smd=round(pre_smd, 4),
            post_match_smd=round(post_smd, 4),
        )
        print(f"PSM ATT = {att:.4f} (SE={se:.4f})")
        print(f"Matched: {result.n_matched_treated} treated, {result.n_matched_control} control")
        print(f"Balance: SMD {pre_smd:.3f} → {post_smd:.3f} {'✓' if post_smd < 0.1 else '⚠'}")
        return result
