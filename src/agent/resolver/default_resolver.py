"""Default resolver — preset expansion, field defaults, and options.

Three resolution modes:
  strict      Fill only safe_default_allowed fields.
  recommended Fill safe and recommended_default_allowed fields + context derivations.
  demo        Fill everything; no user interaction required.

All filled fields are recorded in DefaultResolutionResult.defaulted_fields.
No field is ever silently defaulted.
User-provided fields are NEVER overwritten.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from src.agent.intent.research_spec import ResearchSpec
from src.agent.registry.registry_schema import ResearchConfigRegistry

_VALID_MODES = frozenset({"strict", "recommended", "demo"})


@dataclass
class DefaultResolutionResult:
    """Output of apply_defaults().

    spec:            A NEW ResearchSpec (the input is never mutated).
    feature_specs:   Full feature configurations (type + params) inferred during
                     resolution. Empty if features were not defaulted.
    preset_fields:   Fields that were applied from the preset (field → value).
    defaulted_fields: Fields filled by default policy (field → value).
    assumptions:     Human-readable description of every decision made.
    warnings:        Non-blocking advisory messages.
    requires_confirmation: True when opinionated (recommended) defaults were applied.
    """

    spec: ResearchSpec
    feature_specs: list[dict[str, Any]] = field(default_factory=list)
    preset_fields: dict[str, Any] = field(default_factory=dict)
    defaulted_fields: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    requires_confirmation: bool = False


# ---------------------------------------------------------------------------
# Preset expansion
# ---------------------------------------------------------------------------

def expand_preset(
    spec: ResearchSpec,
    registry: ResearchConfigRegistry,
) -> tuple[ResearchSpec, dict[str, Any], list[dict[str, Any]]]:
    """Apply preset fields to a ResearchSpec, respecting user overrides.

    Returns:
        (new_spec, preset_fields_applied, feature_specs_from_preset)

    preset_fields_applied maps field_name → value that came from the preset.
    User-provided fields (non-None in spec) always win over preset values.

    Raises:
        KeyError: if spec.preset names a preset that does not exist in the registry.
    """
    if spec.preset is None:
        return spec, {}, []

    preset = registry.preset(spec.preset)
    applied: dict[str, Any] = {}
    feature_specs: list[dict[str, Any]] = []

    # We build a dict of values to fill, then call dataclasses.replace once.
    fills: dict[str, Any] = {}

    def _fill(attr: str, value: Any) -> None:
        """Set attr on the working spec only if user has not already set it."""
        if getattr(spec, attr) is None and value is not None:
            fills[attr] = value
            applied[attr] = value

    # Universe and related
    _fill("universe", preset.universe if preset.universe else None)
    _fill("frequency", preset.frequency if preset.frequency else None)
    _fill("asset_class", preset.asset_class if preset.asset_class not in ("custom", "multi_asset") else None)
    _fill("benchmark", preset.benchmark if preset.benchmark else None)
    _fill("transaction_cost_bps", preset.transaction_cost_bps if preset.transaction_cost_bps else None)

    # Features: extract type names; preserve full specs for ResolvedResearchSpec
    if spec.features is None and preset.features:
        feature_type_names = [
            (f["type"] if isinstance(f, dict) else str(f))
            for f in preset.features
        ]
        feature_specs = [
            (dict(f) if isinstance(f, dict) else {"type": str(f), "params": {}})
            for f in preset.features
        ]
        fills["features"] = feature_type_names
        applied["features"] = feature_type_names

    # Label
    if spec.label is None and preset.labels and "type" in preset.labels:
        fills["label"] = preset.labels["type"]
        applied["label"] = preset.labels["type"]
        if spec.label_horizon is None and "params" in preset.labels:
            horizon = preset.labels["params"].get("horizon")
            if horizon is not None:
                fills["label_horizon"] = int(horizon)
                applied["label_horizon"] = int(horizon)

    # Model
    if spec.model is None and preset.model and "type" in preset.model:
        fills["model"] = preset.model["type"]
        applied["model"] = preset.model["type"]
        if spec.model_params is None and preset.model.get("params"):
            fills["model_params"] = dict(preset.model["params"])
            applied["model_params"] = dict(preset.model["params"])

    # Signal
    if spec.signal is None and preset.signal and "type" in preset.signal:
        fills["signal"] = preset.signal["type"]
        applied["signal"] = preset.signal["type"]
        if spec.signal_params is None and preset.signal.get("params"):
            fills["signal_params"] = dict(preset.signal["params"])
            applied["signal_params"] = dict(preset.signal["params"])

    # Validation
    if spec.validation_protocol is None and preset.validation and "type" in preset.validation:
        fills["validation_protocol"] = preset.validation["type"]
        applied["validation_protocol"] = preset.validation["type"]
        params = preset.validation.get("params", {})
        if spec.train_window is None and params.get("train_months") is not None:
            fills["train_window"] = f"{params['train_months']}m"
            applied["train_window"] = f"{params['train_months']}m"
        if spec.test_window is None and params.get("test_months") is not None:
            fills["test_window"] = f"{params['test_months']}m"
            applied["test_window"] = f"{params['test_months']}m"

    # Diagnostics
    if spec.diagnostics is None and preset.diagnostics:
        diag = list(preset.diagnostics)
        fills["diagnostics"] = diag
        applied["diagnostics"] = diag

    new_spec = dataclasses.replace(spec, **fills)
    return new_spec, applied, feature_specs


# ---------------------------------------------------------------------------
# Default resolver
# ---------------------------------------------------------------------------

def apply_defaults(
    spec: ResearchSpec,
    registry: ResearchConfigRegistry,
    mode: str = "recommended",
) -> DefaultResolutionResult:
    """Fill missing fields using the registry default policy.

    Never mutates spec. Returns a new ResearchSpec copy with filled fields.
    Records every decision in DefaultResolutionResult.

    Mode semantics:
      strict:       Only fill safe_default_allowed fields.
      recommended:  Fill safe + recommended defaults + context derivations.
      demo:         Fill everything, including derived fields; no confirmation needed.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown resolution mode '{mode}'. Valid: {sorted(_VALID_MODES)}")

    policy_data = registry.default_policy().field_policies
    fills: dict[str, Any] = {}
    defaulted: dict[str, Any] = {}
    assumptions: list[str] = []
    warnings: list[str] = []
    feature_specs_from_default: list[dict[str, Any]] = []
    requires_confirmation = False

    def _add(attr: str, value: Any, reason: str, is_recommended: bool = False) -> None:
        """Queue a field fill only if not already set by the user."""
        if getattr(spec, attr) is None and value is not None:
            fills[attr] = value
            defaulted[attr] = value
            assumptions.append(reason)
            nonlocal requires_confirmation
            if is_recommended:
                requires_confirmation = True

    # ------------------------------------------------------------------
    # safe_default_allowed fields (filled in ALL modes)
    # ------------------------------------------------------------------

    _apply_safe_defaults(spec, policy_data, _add, registry)

    # ------------------------------------------------------------------
    # recommended_default_allowed fields (filled in recommended + demo)
    # ------------------------------------------------------------------

    if mode in ("recommended", "demo"):
        _apply_recommended_defaults(
            spec, policy_data, _add, registry,
            feature_specs_container=feature_specs_from_default,
        )

    # ------------------------------------------------------------------
    # Context-derived fields (filled in recommended + demo)
    # ------------------------------------------------------------------

    if mode in ("recommended", "demo"):
        _apply_context_derivations(spec, fills, _add, registry)

    # ------------------------------------------------------------------
    # demo-only: fill asset_class if still missing (aggressive completion)
    # ------------------------------------------------------------------

    if mode == "demo" and spec.asset_class is None and "asset_class" not in fills:
        # Derive from universe if universe was just filled
        resolved_universe = fills.get("universe") or spec.universe
        if resolved_universe and resolved_universe != "custom_ticker_list":
            try:
                u = registry.universe(resolved_universe)
                _add(
                    "asset_class",
                    u.asset_class,
                    f"Demo: derived asset_class='{u.asset_class}' from universe '{resolved_universe}'",
                    is_recommended=False,
                )
            except KeyError:
                pass

    new_spec = dataclasses.replace(spec, **fills)

    return DefaultResolutionResult(
        spec=new_spec,
        feature_specs=feature_specs_from_default,
        preset_fields={},
        defaulted_fields=defaulted,
        assumptions=assumptions,
        warnings=warnings,
        requires_confirmation=requires_confirmation,
    )


def _apply_safe_defaults(
    spec: ResearchSpec,
    policy_data: dict[str, Any],
    _add,
    registry: ResearchConfigRegistry,
) -> None:
    """Fill safe_default_allowed fields. Called in all resolution modes."""

    # validation_protocol
    p_val = policy_data.get("validation", {})
    if p_val.get("policy") == "safe_default_allowed" and spec.validation_protocol is None:
        default = p_val.get("safe_default", "rolling")
        _add("validation_protocol", default,
             f"Defaulted validation_protocol='{default}' (safe default)", is_recommended=False)
        params = p_val.get("safe_default_params", {})
        if spec.train_window is None and params.get("train_months") is not None:
            _add("train_window", f"{params['train_months']}m",
                 f"Defaulted train_window='{params['train_months']}m' (safe default)", is_recommended=False)
        if spec.test_window is None and params.get("test_months") is not None:
            _add("test_window", f"{params['test_months']}m",
                 f"Defaulted test_window='{params['test_months']}m' (safe default)", is_recommended=False)

    # benchmark
    p_bench = policy_data.get("benchmark", {})
    if p_bench.get("policy") == "safe_default_allowed" and spec.benchmark is None:
        default = p_bench.get("safe_default", "SPY")
        _add("benchmark", default,
             f"Defaulted benchmark='{default}' (safe default)", is_recommended=False)

    # diagnostics
    p_diag = policy_data.get("diagnostics", {})
    if p_diag.get("policy") == "safe_default_allowed" and spec.diagnostics is None:
        default = p_diag.get("safe_default", ["split_metrics"])
        _add("diagnostics", list(default),
             f"Defaulted diagnostics={default} (safe default)", is_recommended=False)


def _apply_recommended_defaults(
    spec: ResearchSpec,
    policy_data: dict[str, Any],
    _add,
    registry: ResearchConfigRegistry,
    feature_specs_container: list[dict[str, Any]],
) -> None:
    """Fill recommended_default_allowed fields. Called in recommended + demo modes."""

    # universe
    p_univ = policy_data.get("universe", {})
    if p_univ.get("policy") == "recommended_default_allowed" and spec.universe is None:
        default = p_univ.get("recommended_default", "canonical_multi_asset_etfs")
        _add("universe", default,
             f"Defaulted universe='{default}' (recommended default)", is_recommended=True)

    # model
    p_model = policy_data.get("model", {})
    if p_model.get("policy") == "recommended_default_allowed" and spec.model is None:
        default = p_model.get("recommended_default", "RidgeRegression")
        _add("model", default,
             f"Defaulted model='{default}' (recommended default)", is_recommended=True)
        if spec.model_params is None:
            params = p_model.get("recommended_params", {})
            if params:
                _add("model_params", dict(params),
                     f"Defaulted model_params={params} (recommended default)", is_recommended=True)

    # label
    p_label = policy_data.get("label", {})
    if p_label.get("policy") == "recommended_default_allowed" and spec.label is None:
        default = p_label.get("recommended_default", "forward_returns")
        _add("label", default,
             f"Defaulted label='{default}' (recommended default)", is_recommended=True)
        # Default horizon from registry
        if spec.label_horizon is None:
            try:
                entry = registry.label(default)
                horizon = entry.default_params.get("horizon")
                if horizon is not None:
                    _add("label_horizon", int(horizon),
                         f"Defaulted label_horizon={horizon} from registry default for '{default}'",
                         is_recommended=True)
            except KeyError:
                pass

    # features
    p_feat = policy_data.get("features", {})
    if p_feat.get("policy") == "recommended_default_allowed":
        if spec.features is None and spec.feature_families is None:
            default_set = p_feat.get("recommended_default_set", [])
            if default_set:
                names = [d["type"] for d in default_set if isinstance(d, dict)]
                _add("features", names,
                     f"Defaulted features={names} (recommended default set)", is_recommended=True)
                feature_specs_container.extend(
                    {"type": d["type"], "params": dict(d.get("params", {}))}
                    for d in default_set if isinstance(d, dict)
                )

    # timeframe — start_date
    p_time = policy_data.get("timeframe", {})
    if p_time.get("policy") == "recommended_default_allowed" and spec.start_date is None:
        default_start = p_time.get("recommended_start")
        if default_start:
            _add("start_date", str(default_start),
                 f"Defaulted start_date='{default_start}' (recommended default)", is_recommended=True)

    # transaction_cost_bps
    p_tc = policy_data.get("transaction_cost_bps", {})
    if p_tc.get("policy") == "recommended_default_allowed" and spec.transaction_cost_bps is None:
        default = p_tc.get("recommended_default", 10)
        _add("transaction_cost_bps", float(default),
             f"Defaulted transaction_cost_bps={default} (recommended default)", is_recommended=True)


def _apply_context_derivations(
    spec: ResearchSpec,
    fills: dict[str, Any],
    _add,
    registry: ResearchConfigRegistry,
) -> None:
    """Derive fields that can be unambiguously inferred from other resolved fields."""

    resolved_universe = fills.get("universe") or spec.universe

    # tickers from named universe
    if resolved_universe and (spec.tickers is None or spec.tickers == []):
        try:
            u = registry.universe(resolved_universe)
            if u.asset_class != "custom" and u.tickers:
                _add("tickers", list(u.tickers),
                     f"Derived tickers from universe '{resolved_universe}' ({len(u.tickers)} assets)",
                     is_recommended=False)
        except KeyError:
            pass

    # asset_class from universe
    if spec.asset_class is None and resolved_universe:
        try:
            u = registry.universe(resolved_universe)
            if u.asset_class not in ("custom",):
                _add("asset_class", u.asset_class,
                     f"Derived asset_class='{u.asset_class}' from universe '{resolved_universe}'",
                     is_recommended=False)
        except KeyError:
            pass

    # frequency from universe
    if spec.frequency is None and resolved_universe:
        try:
            u = registry.universe(resolved_universe)
            _add("frequency", u.frequency,
                 f"Derived frequency='{u.frequency}' from universe '{resolved_universe}'",
                 is_recommended=False)
        except KeyError:
            pass

    # features from feature_families expansion
    if spec.features is None and spec.feature_families and "features" not in fills:
        expanded = _expand_feature_families(spec.feature_families, registry)
        if expanded:
            _add("features", expanded,
                 f"Expanded feature_families {spec.feature_families} to {expanded}",
                 is_recommended=False)

    # signal derived from universe size + label
    resolved_signal = fills.get("signal") or spec.signal
    resolved_label = fills.get("label") or spec.label
    if resolved_signal is None and resolved_universe:
        derived_signal = _derive_signal(resolved_universe, resolved_label, fills, spec, registry)
        if derived_signal:
            _add("signal", derived_signal,
                 f"Derived signal='{derived_signal}' from universe and label context",
                 is_recommended=False)


def _expand_feature_families(
    families: list[str],
    registry: ResearchConfigRegistry,
) -> list[str]:
    """Return all feature type names belonging to the given family names."""
    families_normalised = {f.lower().strip() for f in families}
    result = []
    for fname in sorted(registry.feature_names()):
        entry = registry.feature(fname)
        if entry.family and entry.family.lower() in families_normalised:
            result.append(fname)
    return result


def _derive_signal(
    universe_name: str,
    label: str | None,
    fills: dict[str, Any],
    spec: ResearchSpec,
    registry: ResearchConfigRegistry,
) -> str | None:
    """Derive a sensible signal type from the universe size and label."""
    try:
        u = registry.universe(universe_name)
        n_tickers = len(u.tickers)
    except KeyError:
        return None

    if n_tickers <= 1:
        return "sign"
    # Multi-asset: use top_n for both ranking and return-prediction labels
    return "top_n"


# ---------------------------------------------------------------------------
# Options provider
# ---------------------------------------------------------------------------

def options_for_field(
    field_name: str,
    registry: ResearchConfigRegistry,
    spec: ResearchSpec | None = None,
) -> list[Any]:
    """Return valid registry choices for a missing field.

    For universe, filters by asset_class if spec.asset_class is set.
    For signal, filters by multi-asset requirement if universe is set.
    """
    if field_name == "asset_class":
        base = sorted(registry.asset_class_names()) + ["multi_asset", "custom"]
        return sorted(set(base))

    if field_name == "universe":
        all_universes = sorted(registry.universe_names())
        if spec and spec.asset_class:
            # Filter to universes whose asset_class matches
            ac = spec.asset_class
            filtered = [
                n for n in all_universes
                if registry.universe(n).asset_class in (ac, "custom", "multi_asset")
            ]
            return filtered or all_universes
        return all_universes

    if field_name == "frequency":
        return ["monthly"]  # only supported frequency in Phase 2

    if field_name == "features":
        return sorted(registry.feature_names())

    if field_name == "feature_families":
        families = sorted({
            e.family for e in (registry.feature(n) for n in registry.feature_names())
            if e.family is not None
        })
        return families

    if field_name == "label":
        return sorted(registry.label_names())

    if field_name == "model":
        return sorted(registry.model_names())

    if field_name == "signal":
        all_signals = sorted(registry.signal_names())
        if spec and spec.universe:
            try:
                u = registry.universe(spec.universe)
                if len(u.tickers) <= 1:
                    # Single-asset: exclude panel-only signals
                    return [s for s in all_signals
                            if not registry.signal(s).requires_multi_asset]
            except KeyError:
                pass
        return all_signals

    if field_name == "validation_protocol":
        return sorted(registry.validation_names())

    if field_name == "transaction_cost_bps":
        return [0, 5, 10, 15, 20, 25, 50]

    if field_name == "diagnostics":
        return sorted(registry.diagnostic_names())

    if field_name == "preset":
        return sorted(registry.preset_names())

    if field_name == "benchmark":
        # Expose common benchmark tickers
        return ["SPY", "QQQ", "IWM", "TLT", "GLD", "EEM"]

    return []
