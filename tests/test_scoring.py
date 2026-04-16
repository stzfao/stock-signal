"""Tests for scoring pipeline."""

import numpy as np
import pandas as pd

from stock_signal.scoring import (
    composite_score,
    cross_sectional_zscore,
    rank_top_decile,
    winsorize,
)


class TestWinsorize:
    def test_clips_outliers(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        result = winsorize(s)
        assert result.max() < 100

    def test_preserves_middle(self):
        s = pd.Series([1, 2, 3, 4, 5])
        result = winsorize(s)
        assert result.iloc[2] == 3


class TestZScore:
    def test_mean_near_zero(self):
        s = pd.Series([10, 20, 30, 40, 50])
        result = cross_sectional_zscore(s)
        assert abs(result.mean()) < 1e-10

    def test_std_near_one(self):
        s = pd.Series([10, 20, 30, 40, 50])
        result = cross_sectional_zscore(s)
        assert abs(result.std() - 1.0) < 0.1

    def test_handles_constant(self):
        s = pd.Series([5, 5, 5])
        result = cross_sectional_zscore(s)
        assert (result == 0).all()


class TestCompositeScore:
    def test_produces_series(self):
        factors = {
            "momentum": pd.Series({"A": 0.1, "B": -0.05, "C": 0.2}),
            "fscore": pd.Series({"A": 7, "B": 5, "C": 8}),
        }
        result = composite_score(factors)
        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_higher_factors_higher_score(self):
        factors = {
            "momentum": pd.Series({"GOOD": 0.5, "BAD": -0.5}),
            "fscore": pd.Series({"GOOD": 9, "BAD": 2}),
            "sue": pd.Series({"GOOD": 2.0, "BAD": -1.0}),
            "revisions": pd.Series({"GOOD": 1.0, "BAD": -1.0}),
            "gross_profitability": pd.Series({"GOOD": 0.5, "BAD": 0.1}),
            "proximity_52wk_high": pd.Series({"GOOD": 0.95, "BAD": 0.6}),
        }
        result = composite_score(factors)
        assert result["GOOD"] > result["BAD"]


class TestRankTopDecile:
    def test_returns_correct_count(self):
        scores = pd.Series({f"S{i}": np.random.randn() for i in range(100)})
        result = rank_top_decile(scores, top_pct=0.10)
        assert len(result) == 10

    def test_sorted_descending(self):
        scores = pd.Series({"A": 3.0, "B": 1.0, "C": 5.0, "D": 2.0})
        result = rank_top_decile(scores, top_pct=0.50)
        assert result["composite_score"].is_monotonic_decreasing

    def test_rank_column(self):
        scores = pd.Series({"A": 3.0, "B": 1.0, "C": 5.0})
        result = rank_top_decile(scores, top_pct=1.0)
        assert list(result["rank"]) == [1, 2, 3]
