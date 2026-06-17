"""Deterministic compatibility checker.

Implements hardcoded checks A–K from the Phase 2 spec plus the rules
declared in config/compatibility/rules.yaml (loaded as YAML metadata but
enforced via code here).

Accepts both ResolvedResearchSpec and ResearchSpec (checked by duck-typing
— only present fields are validated; None fields are skipped).
"""

from __future__ import annotations

from typing import Any

from src.agent.critic.compatibility_result import CompatibilityIssue, CompatibilityResult
from src.agent.registry.registry_schema import ResearchConfigRegistry


def check_compatibility(
    spec: Any,  # ResearchSpec | ResolvedResearchSpec
    registry: ResearchConfigRegistry,
) -> CompatibilityResult:
    """Run all compatibility checks against a spec.

    Fields that are None are silently skipped — partial specs only fail on
    fields that ARE set but are incompatible or invalid.

    Args:
        spec:     A ResearchSpec or ResolvedResearchSpec instance.
        registry: The loaded ResearchConfigRegistry.

    Returns:
        CompatibilityResult with errors, warnings, and info issues.
    """
    result = CompatibilityResult()

    _check_registry_membership(spec, registry, result)
    _check_panel_signal_universe(spec, registry, result)
    _check_panel_label_universe(spec, registry, result)
    _check_date_order(spec, result)
    _check_transaction_cost(spec, result)
    _check_rolling_beta_market_ticker(spec, registry, result)
    _check_vol_compression_windows(spec, registry, result)
    _check_logistic_label(spec, result)
    _check_weighting_universe(spec, registry, result)
    _check_lasso_max_iter(spec, result)
    _check_high_turnover_cost(spec, registry, result)
    _check_rebalance_vs_horizon(spec, result)

    return result


# ---------------------------------------------------------------------------
# A / B / C  Registry membership checks
# ---------------------------------------------------------------------------

def _check_registry_membership(
    spec: Any,
    registry: ResearchConfigRegistry,
    result: CompatibilityResult,
) -> None:
    """A–H. Unknown model / features / signal / label / validation are errors."""

    model = _get(spec, "model")
    if model is not None and model not in registry.model_names():
        result.errors.append(CompatibilityIssue(
            rule_id="CHECK_A",
            severity="error",
            message=f"Model '{model}' is not in the registry. Valid models: {sorted(registry.model_names())}",
            field="model",
        ))

    features = _get(spec, "features") or []
    for ft in features:
        if ft not in registry.feature_names():
            result.errors.append(CompatibilityIssue(
                rule_id="CHECK_B",
                severity="error",
                message=f"Feature type '{ft}' is not in the registry. Valid types: {sorted(registry.feature_names())}",
                field="features",
            ))

    signal = _get(spec, "signal")
    if signal is not None and signal not in registry.signal_names():
        result.errors.append(CompatibilityIssue(
            rule_id="CHECK_C",
            severity="error",
            message=f"Signal '{signal}' is not in the registry. Valid signals: {sorted(registry.signal_names())}",
            field="signal",
        ))

    label = _get(spec, "label")
    if label is not None and label not in registry.label_names():
        result.errors.append(CompatibilityIssue(
            rule_id="CHECK_D",
            severity="error",
            message=f"Label '{label}' is not in the registry. Valid labels: {sorted(registry.label_names())}",
            field="label",
        ))

    val = _get(spec, "validation_protocol")
    if val is not None and val not in registry.validation_names():
        result.errors.append(CompatibilityIssue(
            rule_id="CHECK_E",
            severity="error",
            message=(
                f"Validation protocol '{val}' is not in the registry. "
                f"Valid protocols: {sorted(registry.validation_names())}"
            ),
            field="validation_protocol",
        ))


# ---------------------------------------------------------------------------
# COMPAT_001 / COMPAT_002 / COMPAT_003  Panel signal / label + universe
# ---------------------------------------------------------------------------

def _check_panel_signal_universe(
    spec: Any,
    registry: ResearchConfigRegistry,
    result: CompatibilityResult,
) -> None:
    """COMPAT_001 / COMPAT_003: Panel signals require multi-asset universe."""
    signal = _get(spec, "signal")
    if signal is None:
        return
    try:
        sig_entry = registry.signal(signal)
    except KeyError:
        return  # already caught by registry membership check

    if not sig_entry.requires_multi_asset:
        return

    n_tickers = _n_tickers(spec, registry)
    if n_tickers is None:
        return  # universe not yet resolved; cannot check

    if n_tickers <= 1:
        result.errors.append(CompatibilityIssue(
            rule_id="COMPAT_001",
            severity="error",
            message=(
                f"Signal '{signal}' requires a multi-asset universe (>= 2 tickers) "
                f"but the selected universe has {n_tickers} ticker(s). "
                "Use 'sign' or 'threshold' for single-asset experiments."
            ),
            field="signal",
        ))
    elif n_tickers > 1:
        # Multi-asset + high-turnover: advisory if transaction cost is zero or very low
        pass  # handled by _check_high_turnover_cost


def _check_panel_label_universe(
    spec: Any,
    registry: ResearchConfigRegistry,
    result: CompatibilityResult,
) -> None:
    """COMPAT_002: ranking_target and other panel labels require multi-asset universe."""
    label = _get(spec, "label")
    if label is None:
        return
    try:
        lbl_entry = registry.label(label)
    except KeyError:
        return

    if not lbl_entry.requires_multi_asset:
        return

    n_tickers = _n_tickers(spec, registry)
    if n_tickers is None:
        return

    if n_tickers <= 1:
        result.errors.append(CompatibilityIssue(
            rule_id="COMPAT_002",
            severity="error",
            message=(
                f"Label '{label}' requires a multi-asset universe (>= 2 tickers) "
                f"but the selected universe has {n_tickers} ticker(s). "
                "Use 'forward_returns' or 'binary_direction' for single-asset experiments."
            ),
            field="label",
        ))


# ---------------------------------------------------------------------------
# I  Date order
# ---------------------------------------------------------------------------

def _check_date_order(spec: Any, result: CompatibilityResult) -> None:
    """I. start_date must be before end_date."""
    start = _get(spec, "start_date")
    end = _get(spec, "end_date")
    if start is None or end is None:
        return
    if str(start) >= str(end):
        result.errors.append(CompatibilityIssue(
            rule_id="CHECK_I",
            severity="error",
            message=f"start_date '{start}' must be before end_date '{end}'.",
            field="start_date",
        ))


# ---------------------------------------------------------------------------
# J  Transaction cost non-negative
# ---------------------------------------------------------------------------

def _check_transaction_cost(spec: Any, result: CompatibilityResult) -> None:
    """J. Transaction costs must be non-negative."""
    tc = _get(spec, "transaction_cost_bps")
    if tc is None:
        return
    if float(tc) < 0:
        result.errors.append(CompatibilityIssue(
            rule_id="CHECK_J",
            severity="error",
            message=f"transaction_cost_bps={tc} is negative. Must be >= 0.",
            field="transaction_cost_bps",
        ))


# ---------------------------------------------------------------------------
# COMPAT_004 / K  rolling_beta market_ticker
# ---------------------------------------------------------------------------

def _check_rolling_beta_market_ticker(
    spec: Any,
    registry: ResearchConfigRegistry,
    result: CompatibilityResult,
) -> None:
    """COMPAT_004 / K. rolling_beta without explicit market_ticker produces a warning."""
    features = _get(spec, "features") or []
    if "rolling_beta" not in features:
        return

    # Check feature_specs for an explicit market_ticker param
    feature_specs = _get(spec, "feature_specs") or []
    has_explicit_market_ticker = any(
        (isinstance(fs, dict) and fs.get("type") == "rolling_beta"
         and "market_ticker" in (fs.get("params") or {}))
        for fs in feature_specs
    )

    if not has_explicit_market_ticker:
        result.warnings.append(CompatibilityIssue(
            rule_id="COMPAT_004",
            severity="warning",
            message=(
                "Feature 'rolling_beta' is used but market_ticker is not set explicitly. "
                "The engine defaults to market_ticker='SPY' (ml_factory.py:178). "
                "For non-US-equity universes, set market_ticker explicitly in feature params."
            ),
            field="features",
        ))


# ---------------------------------------------------------------------------
# COMPAT_009  vol_compression window ordering
# ---------------------------------------------------------------------------

def _check_vol_compression_windows(
    spec: Any,
    registry: ResearchConfigRegistry,
    result: CompatibilityResult,
) -> None:
    """COMPAT_009. vol_compression: short_window must be < long_window."""
    features = _get(spec, "features") or []
    if "vol_compression" not in features:
        return

    feature_specs = _get(spec, "feature_specs") or []
    for fs in feature_specs:
        if not isinstance(fs, dict) or fs.get("type") != "vol_compression":
            continue
        params = fs.get("params") or {}
        short = params.get("short_window")
        long_ = params.get("long_window")
        if short is not None and long_ is not None:
            if int(short) >= int(long_):
                result.warnings.append(CompatibilityIssue(
                    rule_id="COMPAT_009",
                    severity="warning",
                    message=(
                        f"vol_compression: short_window={short} must be < long_window={long_}. "
                        "The engine may raise ValueError if this invariant is violated."
                    ),
                    field="features",
                ))


# ---------------------------------------------------------------------------
# COMPAT_007  LogisticRegression + label
# ---------------------------------------------------------------------------

def _check_logistic_label(spec: Any, result: CompatibilityResult) -> None:
    """COMPAT_007. LogisticRegression is most meaningful with binary_direction labels."""
    model = _get(spec, "model")
    label = _get(spec, "label")
    if model == "LogisticRegression" and label is not None and label != "binary_direction":
        result.info.append(CompatibilityIssue(
            rule_id="COMPAT_007",
            severity="info",
            message=(
                f"LogisticRegression is designed for binary classification. "
                f"Using it with label='{label}' is technically valid but the model output "
                "is a class probability, not a continuous return forecast. "
                "Consider using binary_direction label for classification experiments."
            ),
            field="model",
        ))


# ---------------------------------------------------------------------------
# COMPAT_008  Weighting scheme + universe size
# ---------------------------------------------------------------------------

def _check_weighting_universe(
    spec: Any,
    registry: ResearchConfigRegistry,
    result: CompatibilityResult,
) -> None:
    """COMPAT_008. confidence_weighted / zscore_softmax on single-asset is advisory."""
    # Weighting scheme is not a direct ResearchSpec field yet, but may be in
    # ResolvedResearchSpec or a dict field. Skip gracefully if not available.
    pass  # Phase 3 will expose weighting_scheme as a top-level spec field


# ---------------------------------------------------------------------------
# COMPAT_010  Lasso / ElasticNet convergence
# ---------------------------------------------------------------------------

def _check_lasso_max_iter(spec: Any, result: CompatibilityResult) -> None:
    """COMPAT_010. Lasso/ElasticNet may need higher max_iter."""
    model = _get(spec, "model")
    if model not in ("LassoRegression", "ElasticNetRegression"):
        return

    model_params = _get(spec, "model_params") or {}
    max_iter = model_params.get("max_iter", 1000)
    if int(max_iter) < 2000:
        result.info.append(CompatibilityIssue(
            rule_id="COMPAT_010",
            severity="info",
            message=(
                f"{model} max_iter={max_iter} may not be sufficient for convergence "
                "on large feature sets or high regularisation. "
                "Increase to 5000+ if ConvergenceWarning appears during execution."
            ),
            field="model_params",
        ))


# ---------------------------------------------------------------------------
# COMPAT_006  High-turnover transaction costs
# ---------------------------------------------------------------------------

def _check_high_turnover_cost(
    spec: Any,
    registry: ResearchConfigRegistry,
    result: CompatibilityResult,
) -> None:
    """COMPAT_006. Panel signals with zero/very low transaction costs produce a warning."""
    signal = _get(spec, "signal")
    if signal is None:
        return
    try:
        sig_entry = registry.signal(signal)
    except KeyError:
        return
    if not sig_entry.requires_multi_asset:
        return

    tc = _get(spec, "transaction_cost_bps")
    if tc is not None and float(tc) == 0.0:
        result.warnings.append(CompatibilityIssue(
            rule_id="COMPAT_006",
            severity="warning",
            message=(
                f"Signal '{signal}' produces high portfolio turnover. "
                "transaction_cost_bps=0 may overstate net returns. "
                "Consider setting >= 5 bps for liquid ETFs."
            ),
            field="transaction_cost_bps",
        ))


# ---------------------------------------------------------------------------
# COMPAT_011  Rebalance frequency > horizon
# ---------------------------------------------------------------------------

def _check_rebalance_vs_horizon(spec: Any, result: CompatibilityResult) -> None:
    """COMPAT_011. rebalance_frequency_days > horizon is likely a misconfiguration.

    When the rebalance period exceeds the prediction horizon, the model is
    making decisions on stale predictions — the forward window used to train
    the label has already elapsed before the next rebalance. This is not a
    hard error (there may be deliberate reasons), but it warrants review.
    """
    rebalance_days = _get(spec, "rebalance_frequency_days")
    horizon = _get(spec, "horizon")
    if rebalance_days is None or horizon is None:
        return
    try:
        r = int(rebalance_days)
        h = int(horizon)
    except (TypeError, ValueError):
        return
    if r > h:
        result.warnings.append(CompatibilityIssue(
            rule_id="COMPAT_011",
            severity="warning",
            message=(
                f"rebalance_frequency_days={r} > horizon={h}. "
                "The model will rebalance less frequently than its prediction horizon, "
                "meaning positions are held past the window the label was trained on. "
                "Verify this is intentional."
            ),
            field="rebalance_frequency_days",
        ))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get(spec: Any, attr: str) -> Any:
    """Safe attribute access that works for both ResearchSpec and ResolvedResearchSpec."""
    return getattr(spec, attr, None)


def _n_tickers(spec: Any, registry: ResearchConfigRegistry) -> int | None:
    """Return the number of tickers for the spec's universe, or None if unknown."""
    # Prefer spec.tickers if already populated
    tickers = _get(spec, "tickers")
    if tickers:
        return len(tickers)

    universe = _get(spec, "universe")
    if universe is None:
        return None
    try:
        return len(registry.universe(universe).tickers)
    except KeyError:
        return None
