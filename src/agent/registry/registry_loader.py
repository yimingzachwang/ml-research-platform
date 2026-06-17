"""Registry loader for the research config registry.

Loads all config/registry/*.yaml files into a ResearchConfigRegistry object.
Fails loudly on missing files, malformed YAML, or duplicate names.
Does not mutate the returned object after construction.

Usage:
    from src.agent.registry import load_registry
    registry = load_registry()          # uses default config/registry/ path
    registry = load_registry(base_path) # uses a custom base directory
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .registry_schema import (
    AssetClassEntry,
    ChangePathEntry,
    CompatibilityRule,
    DefaultPolicy,
    DiagnosticRegistryEntry,
    FeatureRegistryEntry,
    LabelRegistryEntry,
    ModelRegistryEntry,
    PresetEntry,
    PredictionNormalizationEntry,
    ResearchConfigRegistry,
    SignalRegistryEntry,
    UniverseRegistryEntry,
    ValidationRegistryEntry,
    WeightingSchemeEntry,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_REGISTRY_PATH = _REPO_ROOT / "config" / "registry"
_DEFAULT_PRESETS_PATH = _REPO_ROOT / "config" / "presets"
_DEFAULT_DEFAULTS_PATH = _REPO_ROOT / "config" / "defaults"
_DEFAULT_COMPAT_PATH = _REPO_ROOT / "config" / "compatibility"


def load_registry(base_path: Path | None = None) -> ResearchConfigRegistry:
    """Load all registry YAML files and return a ResearchConfigRegistry.

    Args:
        base_path: Optional override for the config/registry/ directory.
                   If None, uses the project-default path.

    Returns:
        Fully populated ResearchConfigRegistry.

    Raises:
        FileNotFoundError: If a required YAML file is missing.
        ValueError: If YAML is malformed, names are duplicated, or entries
                    are structurally invalid.
    """
    reg_path = Path(base_path) if base_path is not None else _DEFAULT_REGISTRY_PATH
    presets_path = reg_path.parent / "presets"
    defaults_path = reg_path.parent / "defaults"
    compat_path = reg_path.parent / "compatibility"

    registry = ResearchConfigRegistry()
    registry._models = _load_models(reg_path / "models.yaml")
    registry._features = _load_features(reg_path / "features.yaml")
    registry._signals = _load_signals(reg_path / "signals.yaml")
    registry._labels = _load_labels(reg_path / "labels.yaml")
    registry._universes = _load_universes(reg_path / "universes.yaml")
    registry._asset_classes = _load_asset_classes(reg_path / "asset_classes.yaml")
    registry._validations = _load_validations(reg_path / "validation.yaml")

    pc_data = _load_yaml_required(reg_path / "portfolio_construction.yaml")
    registry._weighting_schemes = _load_weighting_schemes(pc_data)
    registry._prediction_normalizations = _load_prediction_normalizations(pc_data)

    registry._diagnostics = _load_diagnostics(reg_path / "diagnostics.yaml")
    registry._change_paths = _load_change_paths(reg_path / "change_paths.yaml")
    registry._presets = _load_presets(presets_path)
    registry._default_policy = _load_default_policy(defaults_path / "default_policy.yaml")
    registry._compatibility_rules = _load_compatibility_rules(compat_path / "rules.yaml")

    return registry


# ---------------------------------------------------------------------------
# Internal loaders
# ---------------------------------------------------------------------------

def _load_yaml_required(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Registry file not found: {path}")
    try:
        with path.open() as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML in {path}: {exc}") from exc
    if data is None:
        raise ValueError(f"Registry file is empty: {path}")
    return data


def _check_no_duplicates(names: list[str], source: Path) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"Duplicate entry '{name}' in {source}")
        seen.add(name)


def _load_models(path: Path) -> dict[str, ModelRegistryEntry]:
    raw: dict[str, Any] = _load_yaml_required(path)
    _check_no_duplicates(list(raw), path)
    result: dict[str, ModelRegistryEntry] = {}
    for name, entry in raw.items():
        result[name] = ModelRegistryEntry(
            name=name,
            display_name=str(entry.get("display_name", name)),
            description=str(entry.get("description", "")),
            supported=bool(entry.get("supported", True)),
            default_params=dict(entry.get("default_params") or {}),
            tunable_params=dict(entry.get("tunable_params") or {}),
            notes=str(entry.get("notes", "")),
        )
    return result


def _load_features(path: Path) -> dict[str, FeatureRegistryEntry]:
    raw: dict[str, Any] = _load_yaml_required(path)
    _check_no_duplicates(list(raw), path)
    result: dict[str, FeatureRegistryEntry] = {}
    for name, entry in raw.items():
        rp = entry.get("required_params") or []
        result[name] = FeatureRegistryEntry(
            name=name,
            display_name=str(entry.get("display_name", name)),
            family=entry.get("family"),
            description=str(entry.get("description", "")),
            required_params=frozenset(rp),
            optional_params=dict(entry.get("optional_params") or {}),
            default_params=dict(entry.get("default_params") or {}),
            data_requirements=tuple(entry.get("data_requirements") or []),
            notes=str(entry.get("notes", "")),
        )
    return result


def _load_signals(path: Path) -> dict[str, SignalRegistryEntry]:
    raw: dict[str, Any] = _load_yaml_required(path)
    _check_no_duplicates(list(raw), path)
    result: dict[str, SignalRegistryEntry] = {}
    for name, entry in raw.items():
        rp = entry.get("required_params") or []
        result[name] = SignalRegistryEntry(
            name=name,
            display_name=str(entry.get("display_name", name)),
            description=str(entry.get("description", "")),
            required_params=frozenset(rp),
            optional_params=dict(entry.get("optional_params") or {}),
            default_params=dict(entry.get("default_params") or {}),
            requires_multi_asset=bool(entry.get("requires_multi_asset", False)),
            notes=str(entry.get("notes", "")),
        )
    return result


def _load_labels(path: Path) -> dict[str, LabelRegistryEntry]:
    raw: dict[str, Any] = _load_yaml_required(path)
    _check_no_duplicates(list(raw), path)
    result: dict[str, LabelRegistryEntry] = {}
    for name, entry in raw.items():
        rp = entry.get("required_params") or []
        result[name] = LabelRegistryEntry(
            name=name,
            display_name=str(entry.get("display_name", name)),
            description=str(entry.get("description", "")),
            required_params=frozenset(rp),
            optional_params=dict(entry.get("optional_params") or {}),
            default_params=dict(entry.get("default_params") or {}),
            requires_multi_asset=bool(entry.get("requires_multi_asset", False)),
            notes=str(entry.get("notes", "")),
        )
    return result


def _load_universes(path: Path) -> dict[str, UniverseRegistryEntry]:
    raw: dict[str, Any] = _load_yaml_required(path)
    _check_no_duplicates(list(raw), path)
    result: dict[str, UniverseRegistryEntry] = {}
    for name, entry in raw.items():
        tickers = entry.get("tickers") or []
        result[name] = UniverseRegistryEntry(
            name=name,
            display_name=str(entry.get("display_name", name)),
            asset_class=str(entry.get("asset_class", "custom")),
            tickers=tuple(str(t).split()[0] for t in tickers),  # strip inline comments
            frequency=str(entry.get("frequency", "monthly")),
            description=str(entry.get("description", "")),
            source=str(entry.get("source", "")),
            notes=str(entry.get("notes", "")),
        )
    return result


def _load_asset_classes(path: Path) -> dict[str, AssetClassEntry]:
    raw: dict[str, Any] = _load_yaml_required(path)
    _check_no_duplicates(list(raw), path)
    result: dict[str, AssetClassEntry] = {}
    for name, entry in raw.items():
        tickers = entry.get("tickers") or []
        result[name] = AssetClassEntry(
            name=name,
            display_name=str(entry.get("display_name", name)),
            description=str(entry.get("description", "")),
            tickers=frozenset(str(t).split()[0] for t in tickers),
        )
    return result


def _load_validations(path: Path) -> dict[str, ValidationRegistryEntry]:
    raw: dict[str, Any] = _load_yaml_required(path)
    _check_no_duplicates(list(raw), path)
    result: dict[str, ValidationRegistryEntry] = {}
    for name, entry in raw.items():
        rp = entry.get("required_params") or []
        result[name] = ValidationRegistryEntry(
            name=name,
            display_name=str(entry.get("display_name", name)),
            description=str(entry.get("description", "")),
            required_params=frozenset(rp),
            optional_params=dict(entry.get("optional_params") or {}),
            default_params=dict(entry.get("default_params") or {}),
            minimum_data_guidance=str(entry.get("minimum_data_guidance", "")),
            notes=str(entry.get("notes", "")),
        )
    return result


def _load_weighting_schemes(pc_data: dict[str, Any]) -> dict[str, WeightingSchemeEntry]:
    raw = pc_data.get("weighting_schemes") or {}
    _check_no_duplicates(list(raw), Path("portfolio_construction.yaml#weighting_schemes"))
    result: dict[str, WeightingSchemeEntry] = {}
    for name, entry in raw.items():
        rp = entry.get("required_params") or []
        result[name] = WeightingSchemeEntry(
            name=name,
            display_name=str(entry.get("display_name", name)),
            description=str(entry.get("description", "")),
            required_params=frozenset(rp),
            optional_params=dict(entry.get("optional_params") or {}),
            default_params=dict(entry.get("default_params") or {}),
            notes=str(entry.get("notes", "")),
        )
    return result


def _load_prediction_normalizations(pc_data: dict[str, Any]) -> dict[str, PredictionNormalizationEntry]:
    raw = pc_data.get("prediction_normalizations") or {}
    _check_no_duplicates(list(raw), Path("portfolio_construction.yaml#prediction_normalizations"))
    result: dict[str, PredictionNormalizationEntry] = {}
    for name, entry in raw.items():
        result[name] = PredictionNormalizationEntry(
            name=name,
            display_name=str(entry.get("display_name", name)),
            description=str(entry.get("description", "")),
            notes=str(entry.get("notes", "")),
        )
    return result


def _load_diagnostics(path: Path) -> dict[str, DiagnosticRegistryEntry]:
    raw: dict[str, Any] = _load_yaml_required(path)
    _check_no_duplicates(list(raw), path)
    result: dict[str, DiagnosticRegistryEntry] = {}
    for name, entry in raw.items():
        result[name] = DiagnosticRegistryEntry(
            name=name,
            display_name=str(entry.get("display_name", name)),
            description=str(entry.get("description", "")),
            artefact_path=str(entry.get("artefact_path", "")),
            required_for_complete_artefact=bool(entry.get("required_for_complete_artefact", False)),
            notes=str(entry.get("notes", "")),
        )
    return result


def _load_change_paths(path: Path) -> tuple[ChangePathEntry, ...]:
    raw: list[Any] = _load_yaml_required(path)
    if not isinstance(raw, list):
        raise ValueError(f"change_paths.yaml must be a YAML list, got {type(raw).__name__}")
    seen: set[str] = set()
    entries: list[ChangePathEntry] = []
    for item in raw:
        dotted = str(item["dotted_path"])
        if dotted in seen:
            raise ValueError(f"Duplicate change path '{dotted}' in {path}")
        seen.add(dotted)
        entries.append(ChangePathEntry(
            section=str(item["section"]),
            field=str(item["field"]),
            dotted_path=dotted,
            description=str(item.get("description", "")),
        ))
    return tuple(entries)


def _load_presets(presets_path: Path) -> dict[str, PresetEntry]:
    if not presets_path.exists():
        return {}
    result: dict[str, PresetEntry] = {}
    for yaml_path in sorted(presets_path.glob("*.yaml")):
        raw: dict[str, Any] = _load_yaml_required(yaml_path)
        name = str(raw.get("name", yaml_path.stem))
        if name in result:
            raise ValueError(f"Duplicate preset name '{name}' (from {yaml_path})")
        features_raw = raw.get("features") or []
        diag_raw = raw.get("diagnostics") or []
        result[name] = PresetEntry(
            name=name,
            display_name=str(raw.get("display_name", name)),
            description=str(raw.get("description", "")),
            asset_class=str(raw.get("asset_class", "custom")),
            universe=str(raw.get("universe", "")),
            frequency=str(raw.get("frequency", "monthly")),
            features=tuple(features_raw),
            labels=dict(raw.get("labels") or {}),
            model=dict(raw.get("model") or {}),
            signal=dict(raw.get("signal") or {}),
            validation=dict(raw.get("validation") or {}),
            portfolio_construction=dict(raw.get("portfolio_construction") or {}),
            transaction_cost_bps=float(raw.get("transaction_cost_bps", 0)),
            benchmark=str(raw.get("benchmark", "")),
            diagnostics=tuple(diag_raw),
            notes=str(raw.get("notes", "")),
        )
    return result


def _load_default_policy(path: Path) -> DefaultPolicy:
    raw: dict[str, Any] = _load_yaml_required(path)
    return DefaultPolicy(
        default_mode=str(raw.get("default_mode", "recommended")),
        field_policies=dict(raw.get("field_policies") or {}),
    )


def _load_compatibility_rules(path: Path) -> tuple[CompatibilityRule, ...]:
    raw: list[Any] = _load_yaml_required(path)
    if not isinstance(raw, list):
        raise ValueError(f"compatibility/rules.yaml must be a YAML list, got {type(raw).__name__}")
    seen_ids: set[str] = set()
    rules: list[CompatibilityRule] = []
    for item in raw:
        rid = str(item["rule_id"])
        if rid in seen_ids:
            raise ValueError(f"Duplicate rule_id '{rid}' in {path}")
        seen_ids.add(rid)
        rules.append(CompatibilityRule(
            rule_id=rid,
            severity=str(item["severity"]),
            description=str(item.get("description", "")),
            applies_when=dict(item.get("applies_when") or {}),
            message=str(item.get("message", "")),
            future_checker=str(item.get("future_checker", "")),
        ))
    return tuple(rules)
