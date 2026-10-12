"""Sanity tests: the estimators recover known truth on simulated data.
Run: pytest -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from frontier import dea_vrs_output_hull, dea_vrs_output_lp, sfa_efficiency, sfa_halfnormal  # noqa: E402


def test_sfa_recovers_parameters():
    rng = np.random.default_rng(1)
    n = 4000
    x = rng.normal(0, 0.5, n)
    X = np.column_stack([np.ones(n), x])
    v = rng.normal(0, 0.08, n)
    u = np.abs(rng.normal(0, 0.12, n))
    y = X @ np.array([-0.5, 0.15]) + v - u
    fit = sfa_halfnormal(y, X)
    assert abs(fit["beta"][1] - 0.15) < 0.02
    assert abs(fit["sigma_v"] - 0.08) < 0.015
    assert abs(fit["sigma_u"] - 0.12) < 0.02
    te, _ = sfa_efficiency(y, X, fit)
    assert np.corrcoef(te, np.exp(-u))[0, 1] > 0.5     # ranks true efficiency


def test_sfa_no_inefficiency_gives_small_sigma_u():
    rng = np.random.default_rng(2)
    n = 3000
    X = np.column_stack([np.ones(n), rng.normal(0, 0.5, n)])
    y = X @ np.array([0.0, 0.2]) + rng.normal(0, 0.1, n) * np.ones(n)
    y = y + 0.0 * rng.normal(size=n)
    fit = sfa_halfnormal(y, X)
    assert fit["sigma_u"] < 0.05


def test_dea_hull_matches_lp():
    rng = np.random.default_rng(3)
    x = rng.uniform(0.3, 2.5, 30)
    y = 0.35 + 0.1 * np.log(x) + rng.normal(0, 0.03, 30)
    assert np.allclose(dea_vrs_output_hull(x, y), dea_vrs_output_lp(x, y), atol=1e-8)
    e = dea_vrs_output_hull(x, y)
    assert e.max() <= 1 + 1e-12 and (e > 0).all() and np.isclose(e.max(), 1.0)
