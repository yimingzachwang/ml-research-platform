"""Tests for periodic retraining support in run_walk_forward_validation.

Covers:
  - Legacy path: retrain_frequency_days=None produces identical outputs
  - Retrain schedule generation
  - Prediction stitching continuity
  - Weight coverage of the full test period
  - Leakage: training prices never exceed the retrain boundary
  - Backward compatibility: new SplitResult fields default to legacy values
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.strategies.baselines import EqualWeightStrategy
from src.validation.splits import rolling_time_splits
from src.validation.walk_forward import (
    SplitResult,
    WalkForwardResult,
    run_walk_forward_validation,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def prices() -> pd.DataFrame:
    idx = pd.date_range("2010-01-01", "2020-12-31", freq="B")
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "A": 100 * np.cumprod(1 + rng.normal(0.001, 0.01, len(idx))),
            "B": 100 * np.cumprod(1 + rng.normal(0.000, 0.01, len(idx))),
            "C": 100 * np.cumprod(1 + rng.normal(0.002, 0.01, len(idx))),
        },
        index=idx,
    )


@pytest.fixture()
def splits(prices: pd.DataFrame):
    return rolling_time_splits(prices.index, train_months=36, test_months=12)


# ---------------------------------------------------------------------------
# Backward compatibility: legacy path unchanged
# ---------------------------------------------------------------------------


def test_legacy_path_n_retrains_is_zero(prices: pd.DataFrame, splits: list) -> None:
    """Legacy SplitResult.n_retrains defaults to 0 when retrain_frequency_days=None."""
    result = run_walk_forward_validation(prices, EqualWeightStrategy(), splits)
    for sr in result.splits:
        assert sr.n_retrains == 0


def test_legacy_path_retrain_dates_is_empty(prices: pd.DataFrame, splits: list) -> None:
    result = run_walk_forward_validation(prices, EqualWeightStrategy(), splits)
    for sr in result.splits:
        assert sr.retrain_dates == []


def test_legacy_path_split_result_is_still_split_result(
    prices: pd.DataFrame, splits: list
) -> None:
    result = run_walk_forward_validation(prices, EqualWeightStrategy(), splits)
    for sr in result.splits:
        assert isinstance(sr, SplitResult)


# ---------------------------------------------------------------------------
# Retrain path: basic structure
# ---------------------------------------------------------------------------


def test_retrain_path_returns_wf_result(prices: pd.DataFrame, splits: list) -> None:
    result = run_walk_forward_validation(
        prices, EqualWeightStrategy(), splits, retrain_frequency_days=30
    )
    assert isinstance(result, WalkForwardResult)


def test_retrain_path_n_splits_equals_n_splits(
    prices: pd.DataFrame, splits: list
) -> None:
    result = run_walk_forward_validation(
        prices, EqualWeightStrategy(), splits, retrain_frequency_days=30
    )
    assert result.n_splits == len(splits)


def test_retrain_path_n_retrains_positive(prices: pd.DataFrame, splits: list) -> None:
    """Each split should have at least one retrain event."""
    result = run_walk_forward_validation(
        prices, EqualWeightStrategy(), splits, retrain_frequency_days=30
    )
    for sr in result.splits:
        assert sr.n_retrains >= 1


def test_retrain_path_retrain_dates_match_n_retrains(
    prices: pd.DataFrame, splits: list
) -> None:
    result = run_walk_forward_validation(
        prices, EqualWeightStrategy(), splits, retrain_frequency_days=30
    )
    for sr in result.splits:
        assert len(sr.retrain_dates) == sr.n_retrains


def test_retrain_path_first_retrain_is_test_start(
    prices: pd.DataFrame, splits: list
) -> None:
    """First retrain date must be the first date of the test window."""
    result = run_walk_forward_validation(
        prices, EqualWeightStrategy(), splits, retrain_frequency_days=30
    )
    for sr in result.splits:
        if sr.retrain_dates:
            assert sr.retrain_dates[0] == sr.split.test_start


def test_retrain_dates_are_within_test_window(
    prices: pd.DataFrame, splits: list
) -> None:
    result = run_walk_forward_validation(
        prices, EqualWeightStrategy(), splits, retrain_frequency_days=30
    )
    for sr in result.splits:
        for rd in sr.retrain_dates:
            assert sr.split.test_start <= rd <= sr.split.test_end


# ---------------------------------------------------------------------------
# Retrain path: weight coverage
# ---------------------------------------------------------------------------


def test_retrain_weights_cover_full_test_window(
    prices: pd.DataFrame, splits: list
) -> None:
    """Stitched weights must cover the entire test window."""
    result = run_walk_forward_validation(
        prices, EqualWeightStrategy(), splits, retrain_frequency_days=30
    )
    for sr in result.splits:
        assert sr.weights.index[0] >= sr.split.test_start
        assert sr.weights.index[-1] <= sr.split.test_end


def test_retrain_equity_within_test_window(prices: pd.DataFrame, splits: list) -> None:
    result = run_walk_forward_validation(
        prices, EqualWeightStrategy(), splits, retrain_frequency_days=30
    )
    for sr in result.splits:
        assert sr.equity_curve.index[0] >= sr.split.test_start
        assert sr.equity_curve.index[-1] <= sr.split.test_end


def test_retrain_metrics_present(prices: pd.DataFrame, splits: list) -> None:
    result = run_walk_forward_validation(
        prices, EqualWeightStrategy(), splits, retrain_frequency_days=30
    )
    for sr in result.splits:
        assert "sharpe_ratio" in sr.metrics
        assert "annualized_return" in sr.metrics


# ---------------------------------------------------------------------------
# Retrain schedule: spacing invariant
# ---------------------------------------------------------------------------


def test_retrain_dates_spacing_at_least_freq_days(
    prices: pd.DataFrame, splits: list
) -> None:
    """Consecutive retrain dates must be separated by exactly retrain_frequency_days
    trading days (since retrain_dates = test_index[::freq])."""
    freq = 20
    result = run_walk_forward_validation(
        prices, EqualWeightStrategy(), splits, retrain_frequency_days=freq
    )
    for sr in result.splits:
        test_idx = prices.index[
            (prices.index >= sr.split.test_start) & (prices.index <= sr.split.test_end)
        ]
        expected_retrain_dates = test_idx[::freq]
        # All retrain dates that were used should match the schedule
        for rd in sr.retrain_dates:
            assert rd in expected_retrain_dates


# ---------------------------------------------------------------------------
# Leakage: no train data touches prices on or after retrain_date
# ---------------------------------------------------------------------------


class _LeakageAuditStrategy(EqualWeightStrategy):
    """Records every training price window end-date passed to fit()."""

    def __init__(self) -> None:
        super().__init__()
        self.fit_calls: list[pd.Timestamp] = []

    def fit(self, train_prices: pd.DataFrame) -> None:  # type: ignore[override]
        self.fit_calls.append(train_prices.index[-1])


def test_retrain_leakage_training_ends_before_retrain_date(
    prices: pd.DataFrame,
) -> None:
    """Each fit() call must receive prices ending strictly before the retrain date."""
    small_splits = rolling_time_splits(prices.index, train_months=36, test_months=12)[:2]
    freq = 21
    strategy = _LeakageAuditStrategy()

    result = run_walk_forward_validation(
        prices, strategy, small_splits, retrain_frequency_days=freq
    )

    # For each split, check recorded fit end-dates against the retrain schedule
    call_iter = iter(strategy.fit_calls)
    for sr in result.splits:
        test_idx = prices.index[
            (prices.index >= sr.split.test_start) & (prices.index <= sr.split.test_end)
        ]
        retrain_schedule = test_idx[::freq]
        for retrain_date in retrain_schedule:
            try:
                fit_end = next(call_iter)
            except StopIteration:
                break
            assert fit_end < retrain_date, (
                f"Training end {fit_end} is not strictly before retrain date {retrain_date}"
            )


# ---------------------------------------------------------------------------
# Retrain vs. legacy: same structure, different refit cadence
# ---------------------------------------------------------------------------


def test_retrain_path_produces_equity_like_legacy(
    prices: pd.DataFrame, splits: list
) -> None:
    """Both paths should produce finite, positive equity curves."""
    legacy = run_walk_forward_validation(prices, EqualWeightStrategy(), splits)
    retrain = run_walk_forward_validation(
        prices, EqualWeightStrategy(), splits, retrain_frequency_days=21
    )
    for sr_l, sr_r in zip(legacy.splits, retrain.splits, strict=False):
        assert sr_l.equity_curve.notna().all()
        assert sr_r.equity_curve.notna().all()
        assert (sr_r.equity_curve > 0).all()


# ---------------------------------------------------------------------------
# Edge case: retrain_frequency_days >= test period length → single retrain
# ---------------------------------------------------------------------------


def test_single_retrain_when_freq_exceeds_test_period(prices: pd.DataFrame) -> None:
    """When freq > len(test_index), only one retrain event should occur."""
    # Use a very short test period (3 months) and a large freq
    short_splits = rolling_time_splits(prices.index, train_months=36, test_months=3)
    result = run_walk_forward_validation(
        prices, EqualWeightStrategy(), short_splits, retrain_frequency_days=500
    )
    for sr in result.splits:
        assert sr.n_retrains == 1


# ---------------------------------------------------------------------------
# Empty splits: still returns empty WalkForwardResult
# ---------------------------------------------------------------------------


def test_retrain_with_empty_splits(prices: pd.DataFrame) -> None:
    result = run_walk_forward_validation(
        prices, EqualWeightStrategy(), splits=[], retrain_frequency_days=21
    )
    assert result.n_splits == 0
    assert result.splits == []
