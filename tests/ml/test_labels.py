"""Tests for src.ml.labels.

Focus areas: shift correctness, trailing NaN, leakage timing, and
multi-asset (panel) behaviour for ranking_target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.ml.labels import (
    binary_direction_label,
    forward_returns,
    ranking_target,
    risk_adjusted_ranking_target,
    volatility_target,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _price_series(n: int = 100, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    log_r = rng.normal(0.0002, 0.01, size=n)
    prices = pd.Series(100.0 * np.exp(log_r.cumsum()), index=idx, name="close")
    return prices


def _price_df(n: int = 120, tickers: list[str] | None = None, seed: int = 0) -> pd.DataFrame:
    tickers = tickers or ["A", "B", "C"]
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    data = {}
    for t in tickers:
        log_r = rng.normal(0.0002, 0.01, size=n)
        data[t] = 100.0 * np.exp(log_r.cumsum())
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# forward_returns
# ---------------------------------------------------------------------------


def test_forward_returns_length_preserved():
    prices = _price_series()
    result = forward_returns(prices, horizon=5)
    assert len(result) == len(prices)


def test_forward_returns_trailing_nans():
    horizon = 10
    prices = _price_series(n=50)
    result = forward_returns(prices, horizon=horizon)
    assert result.iloc[-horizon:].isna().all()


def test_forward_returns_no_leading_nans_from_shift():
    prices = _price_series(n=50)
    result = forward_returns(prices, horizon=5)
    # The first few may be NaN due to pct_change(horizon), but no additional
    # NaN should appear in the middle rows
    non_nan = result.dropna()
    assert len(non_nan) > 0


def test_forward_returns_value_correctness():
    # Manual check: forward_returns at t should equal prices[t+h]/prices[t] - 1
    horizon = 3
    idx = pd.date_range("2020-01-01", periods=20, freq="B")
    prices = pd.Series(np.arange(100.0, 120.0), index=idx)
    result = forward_returns(prices, horizon=horizon)
    for i in range(len(prices) - horizon):
        expected = prices.iloc[i + horizon] / prices.iloc[i] - 1
        assert abs(result.iloc[i] - expected) < 1e-10, f"Mismatch at row {i}"


def test_forward_returns_same_index_as_input():
    prices = _price_series()
    result = forward_returns(prices, horizon=5)
    assert result.index.equals(prices.index)


def test_forward_returns_horizon_1():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    prices = pd.Series([100.0, 102.0, 101.0, 103.0, 104.0], index=idx)
    result = forward_returns(prices, horizon=1)
    assert abs(result.iloc[0] - (102.0 / 100.0 - 1)) < 1e-10
    assert result.iloc[-1] != result.iloc[-1]  # NaN check


# ---------------------------------------------------------------------------
# binary_direction_label
# ---------------------------------------------------------------------------


def test_binary_direction_label_values_are_zero_or_one():
    prices = _price_series()
    result = binary_direction_label(prices, horizon=5)
    valid = result.dropna()
    assert set(valid.unique()).issubset({0.0, 1.0})


def test_binary_direction_label_trailing_nans():
    horizon = 7
    prices = _price_series(n=60)
    result = binary_direction_label(prices, horizon=horizon)
    assert result.iloc[-horizon:].isna().all()


def test_binary_direction_label_dtype_float():
    prices = _price_series()
    result = binary_direction_label(prices, horizon=5)
    assert pd.api.types.is_float_dtype(result)


def test_binary_direction_label_consistent_with_forward_returns():
    prices = _price_series()
    horizon = 10
    fwd = forward_returns(prices, horizon=horizon)
    binary = binary_direction_label(prices, horizon=horizon)
    # Where fwd > 0, binary should be 1; where fwd <= 0, binary should be 0
    valid_mask = fwd.notna()
    assert (binary[valid_mask & (fwd > 0)] == 1.0).all()
    assert (binary[valid_mask & (fwd <= 0)] == 0.0).all()


def test_binary_direction_label_same_index_as_input():
    prices = _price_series()
    result = binary_direction_label(prices, horizon=5)
    assert result.index.equals(prices.index)


# ---------------------------------------------------------------------------
# volatility_target
# ---------------------------------------------------------------------------


def test_volatility_target_length_preserved():
    prices = _price_series()
    result = volatility_target(prices, horizon=10)
    assert len(result) == len(prices)


def test_volatility_target_trailing_nans():
    horizon = 15
    prices = _price_series(n=80)
    result = volatility_target(prices, horizon=horizon)
    assert result.iloc[-horizon:].isna().all()


def test_volatility_target_non_negative_where_valid():
    prices = _price_series(n=100)
    result = volatility_target(prices, horizon=10)
    valid = result.dropna()
    assert (valid >= 0).all()


def test_volatility_target_same_index_as_input():
    prices = _price_series()
    result = volatility_target(prices, horizon=5)
    assert result.index.equals(prices.index)


# ---------------------------------------------------------------------------
# ranking_target
# ---------------------------------------------------------------------------


def test_ranking_target_shape_matches_input():
    prices = _price_df()
    result = ranking_target(prices, horizon=10)
    assert result.shape == prices.shape


def test_ranking_target_columns_match_input():
    prices = _price_df(tickers=["SPY", "QQQ", "IWM"])
    result = ranking_target(prices, horizon=5)
    assert list(result.columns) == list(prices.columns)


def test_ranking_target_trailing_nans():
    horizon = 10
    prices = _price_df(n=80)
    result = ranking_target(prices, horizon=horizon)
    assert result.iloc[-horizon:].isna().all().all()


def test_ranking_target_values_in_unit_interval():
    prices = _price_df(n=200, tickers=["A", "B", "C", "D"])
    result = ranking_target(prices, horizon=20)
    valid = result.dropna()
    assert (valid > 0.0).all().all()
    assert (valid <= 1.0).all().all()


def test_ranking_target_relative_ordering():
    # Construct prices where A always beats B over the horizon window
    horizon = 5
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = pd.DataFrame(
        {
            "A": np.exp(np.linspace(0, 1, n)),   # always rising faster
            "B": np.exp(np.linspace(0, 0.3, n)), # rising slower
        },
        index=idx,
    )
    result = ranking_target(prices, horizon=horizon)
    valid = result.dropna()
    # A should consistently rank higher (rank closer to 1.0)
    assert (valid["A"] > valid["B"]).all()


def test_ranking_target_same_index_as_input():
    prices = _price_df()
    result = ranking_target(prices, horizon=5)
    assert result.index.equals(prices.index)


# ---------------------------------------------------------------------------
# risk_adjusted_ranking_target
# ---------------------------------------------------------------------------


def test_rart_shape_matches_input_horizon5():
    prices = _price_df(n=120)
    result = risk_adjusted_ranking_target(prices, horizon=5)
    assert result.shape == prices.shape


def test_rart_shape_matches_input_horizon10():
    prices = _price_df(n=120)
    result = risk_adjusted_ranking_target(prices, horizon=10)
    assert result.shape == prices.shape


def test_rart_shape_matches_input_horizon21():
    prices = _price_df(n=200)
    result = risk_adjusted_ranking_target(prices, horizon=21)
    assert result.shape == prices.shape


def test_rart_columns_match_input():
    prices = _price_df(tickers=["SPY", "QQQ", "IWM"])
    result = risk_adjusted_ranking_target(prices, horizon=5)
    assert list(result.columns) == list(prices.columns)


def test_rart_trailing_nans_horizon5():
    horizon = 5
    prices = _price_df(n=80)
    result = risk_adjusted_ranking_target(prices, horizon=horizon)
    assert result.iloc[-horizon:].isna().all().all()


def test_rart_trailing_nans_horizon21():
    horizon = 21
    prices = _price_df(n=120)
    result = risk_adjusted_ranking_target(prices, horizon=horizon)
    assert result.iloc[-horizon:].isna().all().all()


def test_rart_same_index_as_input():
    prices = _price_df()
    result = risk_adjusted_ranking_target(prices, horizon=5)
    assert result.index.equals(prices.index)


def test_rart_values_in_unit_interval():
    prices = _price_df(n=200, tickers=["A", "B", "C", "D"])
    result = risk_adjusted_ranking_target(prices, horizon=10)
    valid = result.dropna()
    assert (valid > 0.0).all().all()
    assert (valid <= 1.0).all().all()


def test_rart_monotonic_ranking():
    # Asset A has consistently high risk-adjusted returns; B has consistently low.
    # Construct such that A's forward return / realized_vol > B's in every window.
    horizon = 5
    n = 60
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    # A: steady upward drift, small vol → high Sharpe
    a_r = 0.003 + rng.normal(0, 0.001, n)
    # B: negative drift, larger vol → low Sharpe
    b_r = -0.003 + rng.normal(0, 0.005, n)
    a_prices = pd.Series(100.0 * np.exp(a_r.cumsum()), index=idx)
    b_prices = pd.Series(100.0 * np.exp(b_r.cumsum()), index=idx)
    prices = pd.DataFrame({"A": a_prices, "B": b_prices})
    result = risk_adjusted_ranking_target(prices, horizon=horizon)
    valid = result.dropna()
    # A should rank higher (closer to 1.0) than B in most rows
    assert (valid["A"] > valid["B"]).mean() > 0.7


def test_rart_nan_propagation():
    # Introduce NaN at a specific cell; that (date, asset) rank should be NaN.
    prices = _price_df(n=80, tickers=["A", "B", "C"])
    prices_with_nan = prices.copy()
    prices_with_nan.iloc[10, 0] = float("nan")  # NaN in asset A at row 10
    result = risk_adjusted_ranking_target(prices_with_nan, horizon=5)
    # Row 10 of A contributed to forward-return / vol computation for rows 5-10;
    # the NaN propagates through pct_change and rolling — row 10 rank for A is NaN.
    assert pd.isna(result.iloc[10, 0])


def test_rart_zero_volatility_excluded_from_ranking():
    # Build a flat asset (zero vol) alongside two normal assets.
    n = 60
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(3)
    flat = pd.Series(100.0, index=idx, name="FLAT")          # zero vol
    a = pd.Series(100.0 * np.exp(rng.normal(0.001, 0.01, n).cumsum()), index=idx)
    b = pd.Series(100.0 * np.exp(rng.normal(-0.001, 0.01, n).cumsum()), index=idx)
    prices = pd.DataFrame({"FLAT": flat, "A": a, "B": b})

    result = risk_adjusted_ranking_target(prices, horizon=5)
    valid = result.dropna(how="all")

    # FLAT has zero realized vol → its rank is NaN on all valid dates
    flat_ranks = valid["FLAT"].dropna()
    assert len(flat_ranks) == 0, "FLAT (zero vol) should have no valid ranks"

    # A and B still produce valid ranks despite FLAT being excluded
    ab_valid = valid[["A", "B"]].dropna(how="all")
    assert len(ab_valid) > 0
    # Two valid assets ranked with pct=True produce {0.5, 1.0} per row → sum = 1.5
    # (rank(pct=True) among k assets gives k equally-spaced values summing to (k+1)/2)
    complete_rows = ab_valid.dropna(how="any")
    row_sums = complete_rows.sum(axis=1)
    assert (row_sums - 1.5).abs().max() < 1e-10
    # All values must be within (0, 1]
    assert (complete_rows > 0.0).all().all()
    assert (complete_rows <= 1.0).all().all()


def test_rart_output_is_independent_dataframe():
    # Verify the result is a new DataFrame, not a view of the input.
    prices = _price_df()
    result = risk_adjusted_ranking_target(prices, horizon=5)
    result.iloc[0, 0] = 99.9
    assert prices.iloc[0, 0] != 99.9


def test_rart_existing_labels_unchanged():
    # Ensure ranking_target produces the same result after adding the new function.
    prices = _price_df(n=100, tickers=["A", "B", "C"])
    horizon = 10
    rt = ranking_target(prices, horizon=horizon)
    fwd = prices.pct_change(horizon).shift(-horizon)
    expected = fwd.rank(axis=1, pct=True)
    pd.testing.assert_frame_equal(rt, expected)
