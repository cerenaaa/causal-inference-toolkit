"""Causal inference demos using synthetic data."""
from data.synthetic_experiments import generate_panel_data, generate_observational_data
from methods.did import DifferenceInDifferences
from methods.psm import PropensityScoreMatching
from methods.iv import TwoStageLeastSquares
import numpy as np

def main():
    print("=" * 50)
    print("DiD Demo (true ATT = 5.0)")
    print("=" * 50)
    df = generate_panel_data(n_units=100, n_periods=8, treatment_effect=5.0)
    did = DifferenceInDifferences()
    did.fit(df)

    print("
" + "=" * 50)
    print("PSM Demo (true ATT = 3.0)")
    print("=" * 50)
    obs = generate_observational_data(n=1000, treatment_effect=3.0)
    psm = PropensityScoreMatching(caliper=0.1)
    psm.fit(obs, "treated", "outcome", ["age", "income"])

if __name__ == "__main__":
    main()
