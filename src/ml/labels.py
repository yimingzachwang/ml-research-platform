"""Label generation for supervised ML research.

All labels use shift(-horizon) — lookahead is explicit and visible in every
function.  No hidden forward windows.  Each function documents the exact
horizon assumption in its docstring.

Labels are purely functional: prices in, targets out.  No state, no fitting.
No label is valid on the last `horizon` rows of the input — those rows will
contain NaN because the future has not arrived.  This is correct and expected;
align_features_and_labels() will remove those rows automatically.
"""

from __future__ import annotations

import pandas as pd


def forward_returns(prices: pd.Series, horizon: int) -> pd.Series:
    """Forward simple return over horizon trading days.

    Label at index t = prices[t + horizon] / prices[t] - 1.

    Implemented as pct_change(horizon).shift(-horizon), which is equivalent
    to the ratio definition above and avoids floating-point accumulation.

    The last `horizon` rows will be NaN — no future data available.

    Args:
        prices: Price series with DatetimeIndex.
        horizon: Prediction horizon in trading days (>= 1).

    Returns:
        Forward return series with the same index as prices.
    """
    return prices.pct_change(horizon).shift(-horizon)


def binary_direction_label(prices: pd.Series, horizon: int) -> pd.Series:
    """Binary direction label: 1.0 if forward return > 0, 0.0 otherwise.

    Label at index t = 1.0 if prices[t + horizon] > prices[t] else 0.0.
    NaN is preserved wherever the forward return is NaN (last horizon rows).

    Stored as float64 (not int) so that NaN rows survive without dtype
    promotion surprises.

    Args:
        prices: Price series with DatetimeIndex.
        horizon: Prediction horizon in trading days (>= 1).

    Returns:
        Float series of 0.0 / 1.0 / NaN with the same index as prices.
    """
    fwd = forward_returns(prices, horizon)
    return (fwd > 0).astype("float64").where(fwd.notna(), other=float("nan"))


def volatility_target(prices: pd.Series, horizon: int) -> pd.Series:
    """Realized return volatility over the next horizon trading days.

    Label at index t = std(daily_returns[t+1 : t+horizon+1]).

    Computed as rolling(horizon).std() of daily returns shifted back by
    horizon periods so the label sits on the row where the prediction would
    be made.  The last `horizon` rows will be NaN.

    Args:
        prices: Price series with DatetimeIndex.
        horizon: Prediction horizon in trading days (>= 1).

    Returns:
        Forward realized volatility series with the same index as prices.
    """
    daily_returns = prices.pct_change()
    return daily_returns.rolling(horizon).std().shift(-horizon)


def ranking_target(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Cross-sectional percentile rank of forward returns within the universe.

    At each row t, assets are ranked by their horizon-period forward return.
    Rank values are in (0, 1]; a higher rank means a better forward return.
    Ties are broken by averaging (pandas default for pct=True).

    The last `horizon` rows will be all NaN.

    Args:
        prices: Price DataFrame (DatetimeIndex × assets).
        horizon: Prediction horizon in trading days (>= 1).

    Returns:
        DataFrame of the same shape as prices with percentile ranks in (0, 1].
    """
    fwd = prices.pct_change(horizon).shift(-horizon)
    return fwd.rank(axis=1, pct=True)


def risk_adjusted_ranking_target(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Cross-sectional percentile rank of risk-adjusted forward returns.

    At each row t, assets are ranked by their horizon-period Sharpe-like ratio:

        risk_adjusted_return = forward_return / realized_vol

    where:
        forward_return = (price[t+h] / price[t]) - 1
        realized_vol   = std(daily_returns[t+1 : t+h+1])   (ddof=1)

    Rank values are in (0, 1]; a higher rank means a better risk-adjusted
    forward return. Ties are broken by averaging.

    The last `horizon` rows will be all NaN (no future data available).

    Edge cases:
        Zero realized volatility: assets with realized_vol == 0 are assigned
            NaN for that date and excluded from cross-sectional ranking on that
            date. Ranks for remaining assets are recomputed on the available
            subset. This is the correct financial interpretation: a Sharpe ratio
            is undefined when there is no price movement over the horizon.
        NaN prices: propagated as NaN through forward return and volatility
            computation; those (date, asset) cells are excluded from ranking.
        horizon == 1: rolling(1).std() is NaN under ddof=1 (undefined standard
            deviation of a single value). All ranks will be NaN at horizon=1.
            The minimum meaningful horizon for this label is 2.

    Args:
        prices: Price DataFrame (DatetimeIndex × assets).
        horizon: Prediction horizon in trading days (>= 2 for meaningful output).

    Returns:
        DataFrame of the same shape as prices with percentile ranks in (0, 1].
    """
    fwd = prices.pct_change(horizon).shift(-horizon)

    daily_returns = prices.pct_change()
    vol = daily_returns.rolling(horizon).std().shift(-horizon)

    # Zero volatility → NaN, not ±inf; excluded from cross-sectional ranking
    vol_safe = vol.where(vol > 0, other=float("nan"))

    ra_return = fwd / vol_safe
    return ra_return.rank(axis=1, pct=True)
