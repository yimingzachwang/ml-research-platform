"""Walk-forward validation runner.

Evaluates a strategy on each test window of a set of time splits without
any look-ahead leakage.

Leakage prevention contract
----------------------------
For each split the strategy receives prices up to and including
``split.test_end`` — this is the maximum amount of data consistent with
when the evaluation period ends.  Critically, *no prices beyond test_end*
are ever passed, so future returns cannot influence weight generation.

The ``run_portfolio_backtest`` function (and therefore ``run_strategy``)
applies ``weights.shift(1)`` before multiplying returns, so the position
entered on day ``t`` only uses information available at close of day ``t-1``.

Future ML compatibility
-----------------------
Strategies may optionally implement a ``fit(train_data)`` method.  If
present, it is called on the training slice before ``generate_weights`` is
called on the full window.  Traditional strategies (no ``fit``) work without
any changes.

Periodic retraining
-------------------
When ``retrain_frequency_days`` is supplied to ``run_walk_forward_validation``,
the model is refitted every N trading days inside each outer split's test
period using a rolling training window matching the outer split width.
Sub-period predictions are stitched before the backtest runs.  The legacy
single-fit path is preserved exactly when ``retrain_frequency_days=None``.

Leakage guarantee for periodic retraining: at each retrain date ``R``, the
training slice ends strictly before ``R``.  Label construction uses only
``prices[:R-1]``, so forward returns at row ``t`` can only reference prices
up to ``R-1+horizon``, all of which precede ``R+horizon`` — no test-period
prices ever enter the training labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.backtesting.metrics import compute_metrics
from src.strategies.base import Strategy
from src.strategies.runner import run_strategy
from src.validation.splits import TimeSplit

# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------


@dataclass
class SplitResult:
    """Performance on one test window.

    Attributes:
        split: The TimeSplit that defines the train/test dates.
        strategy_name: Canonical name of the strategy used.
        metrics: Standard metrics computed over the test period.
        equity_curve: Cumulative return starting at 1.0 over test period.
        weights: Daily weights during test period.
        n_retrains: Number of periodic retrain events within this split.
            Zero for the legacy single-fit path.
        retrain_dates: Ordered list of retrain timestamps.  Empty for legacy.
    """

    split: TimeSplit
    strategy_name: str
    metrics: dict[str, float]
    equity_curve: pd.Series
    weights: pd.DataFrame
    n_retrains: int = 0
    retrain_dates: list = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """Aggregated result of a full walk-forward validation run.

    Attributes:
        strategy_name: Canonical name of the strategy.
        splits: Ordered list of per-split results.
    """

    strategy_name: str
    splits: list[SplitResult] = field(default_factory=list)

    @property
    def n_splits(self) -> int:
        return len(self.splits)


# ---------------------------------------------------------------------------
# Internal helper: periodic retraining within one outer split
# ---------------------------------------------------------------------------


def _run_split_with_retraining(
    prices: pd.DataFrame,
    strategy: Strategy,
    split: TimeSplit,
    transaction_cost_bps: float,
    retrain_frequency_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Run one outer split with periodic model retraining.

    Retrains the model every ``retrain_frequency_days`` trading days inside the
    test period.  Each retrain uses a rolling training window whose calendar
    width matches the outer split's training window.  Sub-period weights are
    stitched into a continuous daily matrix before the portfolio backtest runs.

    Leakage guarantee: each retrain receives ``prices[:retrain_date - 1 day]``.
    Label construction inside ``strategy.fit()`` naturally produces NaN on the
    last ``horizon`` rows of that slice, which are then dropped by
    ``align_features_and_labels`` — no future prices ever enter the labels.

    Args:
        prices: Full price history available at the time of this split.
        strategy: Strategy instance with an optional ``fit()`` method.
        split: Outer TimeSplit defining train/test boundaries.
        transaction_cost_bps: One-way cost in basis points.
        retrain_frequency_days: Retrain every N trading days in the test period.

    Returns:
        Tuple of (test_backtest, test_weights, retrain_dates_used).
        test_backtest and test_weights cover [split.test_start, split.test_end].
        retrain_dates_used is the ordered list of retrain timestamps applied.
    """
    from src.portfolio.panel import universe_returns
    from src.portfolio.portfolio_backtest import run_portfolio_backtest

    # Price window visible at this split — no leakage beyond test_end
    prices_window = prices.loc[:split.test_end]

    # Test-period index
    test_index = prices_window.index[prices_window.index >= split.test_start]
    if len(test_index) == 0:
        return pd.DataFrame(dtype=float), pd.DataFrame(dtype=float), []

    # Retrain schedule: first retrain at test_start, then every N days
    retrain_dates = test_index[::retrain_frequency_days]

    # Calendar width of the outer training window — preserved for each retrain
    train_window_td = split.train_end - split.train_start

    sub_weights_parts: list[pd.DataFrame] = []
    retrain_dates_used: list[pd.Timestamp] = []

    for i, retrain_date in enumerate(retrain_dates):
        # Rolling training window ends strictly before retrain_date
        pre_retrain_idx = prices_window.index[prices_window.index < retrain_date]
        retrain_train_end = pre_retrain_idx[-1] if len(pre_retrain_idx) > 0 else split.train_end

        # Slide the window back by the outer split's training width
        retrain_train_start_ts = retrain_train_end - train_window_td
        avail_from = prices.index[prices.index >= retrain_train_start_ts]
        retrain_train_start = avail_from[0] if len(avail_from) > 0 else prices.index[0]

        retrain_train_prices = prices.loc[retrain_train_start:retrain_train_end]

        if hasattr(strategy, "fit") and not retrain_train_prices.empty:
            strategy.fit(retrain_train_prices)  # type: ignore[attr-defined]

        # Sub-period: [retrain_date, next_retrain - 1] (or test_end for the last)
        if i + 1 < len(retrain_dates):
            pre_next_idx = prices_window.index[prices_window.index < retrain_dates[i + 1]]
            sub_end = pre_next_idx[-1] if len(pre_next_idx) > 0 else split.test_end
        else:
            sub_end = split.test_end

        # Generate weights on the full window for correct feature warm-up
        weights_full = strategy.generate_weights(prices_window)

        # Extract sub-period slice
        weights_sub = weights_full.loc[
            (weights_full.index >= retrain_date) & (weights_full.index <= sub_end)
        ]

        if not weights_sub.empty:
            sub_weights_parts.append(weights_sub)
            retrain_dates_used.append(retrain_date)

    if not sub_weights_parts:
        # No valid retrain events — fall back to legacy behaviour for this split
        if hasattr(strategy, "fit"):
            strategy.fit(prices.loc[split.train_start : split.train_end])  # type: ignore[attr-defined]
        result = run_strategy(prices_window, strategy, transaction_cost_bps=transaction_cost_bps)
        test_backtest = result.backtest.loc[split.test_start : split.test_end]
        test_weights = result.weights.loc[split.test_start : split.test_end]
        return test_backtest, test_weights, []

    # Stitch sub-period weights into a continuous daily matrix
    stitched_weights = pd.concat(sub_weights_parts).sort_index()
    stitched_weights = stitched_weights[~stitched_weights.index.duplicated(keep="first")]

    # Build full-window weight matrix; pre-test dates are flat (0)
    all_tickers = stitched_weights.columns
    full_weights = pd.DataFrame(0.0, index=prices_window.index, columns=all_tickers)
    full_weights.update(stitched_weights)

    # Portfolio backtest on stitched weights (engine unchanged)
    returns = universe_returns(prices_window)
    bt_result = run_portfolio_backtest(
        returns=returns,
        weights=full_weights,
        transaction_cost_bps=transaction_cost_bps,
    )

    test_backtest = bt_result.backtest.loc[split.test_start : split.test_end]
    test_weights = bt_result.weights.loc[split.test_start : split.test_end]

    return test_backtest, test_weights, retrain_dates_used


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_walk_forward_validation(
    prices: pd.DataFrame,
    strategy: Strategy,
    splits: list[TimeSplit],
    transaction_cost_bps: float = 0.0,
    retrain_frequency_days: int | None = None,
) -> WalkForwardResult:
    """Run walk-forward validation of a strategy over a set of time splits.

    For each split in the **legacy path** (``retrain_frequency_days=None``):
    1. Optionally calls ``strategy.fit(train_prices)`` if the method exists.
    2. Runs the strategy on ``prices[:test_end]`` to generate weights.
    3. Extracts the test-period slice of the backtest.
    4. Computes metrics solely on the test window.

    For the **periodic retraining path** (``retrain_frequency_days`` is set):
    1. Within each test window, retrains every N trading days using a rolling
       training window matching the outer split width.
    2. Sub-period weights are stitched into a continuous daily matrix.
    3. Portfolio backtest runs on the stitched weights.
    4. Metrics computed solely on the test window — identical consumer interface.

    No data from after ``split.test_end`` is ever visible to the strategy.

    Args:
        prices: Date × Asset price DataFrame.
        strategy: Strategy instance implementing ``generate_weights()``.
        splits: Time splits from ``rolling_time_splits`` or
            ``expanding_time_splits``.
        transaction_cost_bps: One-way transaction cost in basis points.
        retrain_frequency_days: Refit the model every N trading days inside
            each test window.  ``None`` (default) preserves the legacy
            single-fit-per-split behaviour exactly.

    Returns:
        WalkForwardResult with per-split metrics, equity curves, and weights.
    """
    split_results: list[SplitResult] = []

    for split in splits:
        if retrain_frequency_days is not None:
            # Periodic retraining path
            test_backtest, test_weights, retrain_dates_used = _run_split_with_retraining(
                prices=prices,
                strategy=strategy,
                split=split,
                transaction_cost_bps=transaction_cost_bps,
                retrain_frequency_days=retrain_frequency_days,
            )
            if test_backtest.empty or "net_return" not in test_backtest.columns:
                continue

            net_returns = test_backtest["net_return"]
            equity_curve = (1.0 + net_returns).cumprod()
            equity_curve.name = "equity"
            test_metrics = compute_metrics(net_returns)

            split_results.append(
                SplitResult(
                    split=split,
                    strategy_name=strategy.name,
                    metrics=test_metrics,
                    equity_curve=equity_curve,
                    weights=test_weights,
                    n_retrains=len(retrain_dates_used),
                    retrain_dates=list(retrain_dates_used),
                )
            )

        else:
            # Legacy path — unchanged behaviour
            prices_window = prices.loc[:split.test_end]

            if hasattr(strategy, "fit"):
                train_prices = prices.loc[split.train_start : split.train_end]
                strategy.fit(train_prices)  # type: ignore[attr-defined]

            result = run_strategy(
                prices_window, strategy, transaction_cost_bps=transaction_cost_bps
            )

            test_backtest = result.backtest.loc[split.test_start : split.test_end]
            test_weights = result.weights.loc[split.test_start : split.test_end]

            net_returns = test_backtest["net_return"]
            equity_curve = (1.0 + net_returns).cumprod()
            equity_curve.name = "equity"
            test_metrics = compute_metrics(net_returns)

            split_results.append(
                SplitResult(
                    split=split,
                    strategy_name=strategy.name,
                    metrics=test_metrics,
                    equity_curve=equity_curve,
                    weights=test_weights,
                )
            )

    return WalkForwardResult(
        strategy_name=strategy.name,
        splits=split_results,
    )
