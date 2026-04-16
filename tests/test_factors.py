"""Tests for factor computation modules."""

import pandas as pd

from stock_signal.factors.fscore import piotroski_fscore
from stock_signal.factors.momentum import momentum_12_1, proximity_52wk_high
from stock_signal.factors.quality import gross_profitability
from stock_signal.factors.revisions import eps_revision_breadth
from stock_signal.factors.sue import sue


class TestMomentum:
    def test_returns_series_indexed_by_symbol(self, sample_prices: pd.DataFrame):
        result = momentum_12_1(sample_prices)
        assert isinstance(result, pd.Series)
        assert len(result) > 0
        for sym in result.index:
            assert isinstance(sym, str)

    def test_values_are_finite(self, sample_prices: pd.DataFrame):
        result = momentum_12_1(sample_prices)
        assert result.notna().all()

    def test_proximity_52wk_high_bounded(self, sample_prices: pd.DataFrame):
        result = proximity_52wk_high(sample_prices)
        assert (result > 0).all()
        assert (result <= 1.0).all()


class TestFScore:
    def test_returns_integer_0_to_9(self, sample_financials: pd.DataFrame):
        result = piotroski_fscore(sample_financials)
        assert isinstance(result, pd.Series)
        assert result.dtype == int
        assert (result >= 0).all()
        assert (result <= 9).all()

    def test_all_symbols_scored(self, sample_financials: pd.DataFrame):
        result = piotroski_fscore(sample_financials)
        assert len(result) == 3


class TestSUE:
    def test_returns_series(self, sample_earnings: pd.DataFrame):
        result = sue(sample_earnings)
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_values_are_finite(self, sample_earnings: pd.DataFrame):
        result = sue(sample_earnings)
        assert result.notna().all()


class TestRevisions:
    def test_breadth_bounded(self, sample_estimates: pd.DataFrame):
        result = eps_revision_breadth(sample_estimates, window_days=365)
        assert isinstance(result, pd.Series)
        assert (result >= -1).all()
        assert (result <= 1).all()

    def test_upward_revisions_positive(self, sample_estimates: pd.DataFrame):
        # AAPL has upward drift in our fixture
        result = eps_revision_breadth(sample_estimates, window_days=365)
        if "AAPL" in result.index:
            assert result["AAPL"] > 0


class TestGrossProfitability:
    def test_returns_positive(self, sample_financials: pd.DataFrame):
        result = gross_profitability(sample_financials)
        assert isinstance(result, pd.Series)
        assert (result > 0).all()

    def test_all_symbols(self, sample_financials: pd.DataFrame):
        result = gross_profitability(sample_financials)
        assert len(result) == 3
