"""Typed dataclasses for the research config registry.

Each dataclass maps to one entry type in config/registry/*.yaml.
The registry_loader.py deserialises raw YAML dicts into these objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelRegistryEntry:
    name: str
    display_name: str
    description: str
    supported: bool
    default_params: dict[str, Any]
    tunable_params: dict[str, Any]
    notes: str


@dataclass(frozen=True)
class FeatureRegistryEntry:
    name: str
    display_name: str
    family: str | None
    description: str
    required_params: frozenset[str]
    optional_params: dict[str, Any]
    default_params: dict[str, Any]
    data_requirements: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class SignalRegistryEntry:
    name: str
    display_name: str
    description: str
    required_params: frozenset[str]
    optional_params: dict[str, Any]
    default_params: dict[str, Any]
    requires_multi_asset: bool
    notes: str


@dataclass(frozen=True)
class LabelRegistryEntry:
    name: str
    display_name: str
    description: str
    required_params: frozenset[str]
    optional_params: dict[str, Any]
    default_params: dict[str, Any]
    requires_multi_asset: bool
    notes: str


@dataclass(frozen=True)
class UniverseRegistryEntry:
    name: str
    display_name: str
    asset_class: str
    tickers: tuple[str, ...]
    frequency: str
    description: str
    source: str
    notes: str


@dataclass(frozen=True)
class AssetClassEntry:
    name: str
    display_name: str
    description: str
    tickers: frozenset[str]


@dataclass(frozen=True)
class ValidationRegistryEntry:
    name: str
    display_name: str
    description: str
    required_params: frozenset[str]
    optional_params: dict[str, Any]
    default_params: dict[str, Any]
    minimum_data_guidance: str
    notes: str


@dataclass(frozen=True)
class WeightingSchemeEntry:
    name: str
    display_name: str
    description: str
    required_params: frozenset[str]
    optional_params: dict[str, Any]
    default_params: dict[str, Any]
    notes: str


@dataclass(frozen=True)
class PredictionNormalizationEntry:
    name: str
    display_name: str
    description: str
    notes: str


@dataclass(frozen=True)
class DiagnosticRegistryEntry:
    name: str
    display_name: str
    description: str
    artefact_path: str
    required_for_complete_artefact: bool
    notes: str


@dataclass(frozen=True)
class ChangePathEntry:
    section: str
    field: str
    dotted_path: str
    description: str


@dataclass(frozen=True)
class PresetEntry:
    name: str
    display_name: str
    description: str
    asset_class: str
    universe: str
    frequency: str
    features: tuple[dict[str, Any], ...]
    labels: dict[str, Any]
    model: dict[str, Any]
    signal: dict[str, Any]
    validation: dict[str, Any]
    portfolio_construction: dict[str, Any]
    transaction_cost_bps: float
    benchmark: str
    diagnostics: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class DefaultPolicy:
    default_mode: str
    field_policies: dict[str, Any]


@dataclass(frozen=True)
class CompatibilityRule:
    rule_id: str
    severity: str
    description: str
    applies_when: dict[str, Any]
    message: str
    future_checker: str


@dataclass
class ResearchConfigRegistry:
    """Top-level registry object. Holds all loaded registry entries.

    Instantiated by registry_loader.load_registry(). Exposes typed read-only
    accessors. All name lookups raise KeyError with a descriptive message on
    unknown keys.
    """

    _models: dict[str, ModelRegistryEntry] = field(default_factory=dict)
    _features: dict[str, FeatureRegistryEntry] = field(default_factory=dict)
    _signals: dict[str, SignalRegistryEntry] = field(default_factory=dict)
    _labels: dict[str, LabelRegistryEntry] = field(default_factory=dict)
    _universes: dict[str, UniverseRegistryEntry] = field(default_factory=dict)
    _asset_classes: dict[str, AssetClassEntry] = field(default_factory=dict)
    _validations: dict[str, ValidationRegistryEntry] = field(default_factory=dict)
    _weighting_schemes: dict[str, WeightingSchemeEntry] = field(default_factory=dict)
    _prediction_normalizations: dict[str, PredictionNormalizationEntry] = field(default_factory=dict)
    _diagnostics: dict[str, DiagnosticRegistryEntry] = field(default_factory=dict)
    _change_paths: tuple[ChangePathEntry, ...] = field(default_factory=tuple)
    _presets: dict[str, PresetEntry] = field(default_factory=dict)
    _default_policy: DefaultPolicy | None = None
    _compatibility_rules: tuple[CompatibilityRule, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------ names

    def model_names(self) -> frozenset[str]:
        return frozenset(self._models)

    def feature_names(self) -> frozenset[str]:
        return frozenset(self._features)

    def feature_family_names(self) -> frozenset[str]:
        return frozenset(
            entry.family
            for entry in self._features.values()
            if entry.family
        )

    def signal_names(self) -> frozenset[str]:
        return frozenset(self._signals)

    def label_names(self) -> frozenset[str]:
        return frozenset(self._labels)

    def universe_names(self) -> frozenset[str]:
        return frozenset(self._universes)

    def asset_class_names(self) -> frozenset[str]:
        return frozenset(self._asset_classes)

    def validation_names(self) -> frozenset[str]:
        return frozenset(self._validations)

    def weighting_scheme_names(self) -> frozenset[str]:
        return frozenset(self._weighting_schemes)

    def prediction_normalization_names(self) -> frozenset[str]:
        return frozenset(self._prediction_normalizations)

    def diagnostic_names(self) -> frozenset[str]:
        return frozenset(self._diagnostics)

    def preset_names(self) -> frozenset[str]:
        return frozenset(self._presets)

    def changeable_paths(self) -> frozenset[str]:
        return frozenset(cp.dotted_path for cp in self._change_paths)

    # ---------------------------------------------------------------- lookups

    def model(self, name: str) -> ModelRegistryEntry:
        if name not in self._models:
            raise KeyError(f"Unknown model '{name}'. Available: {sorted(self._models)}")
        return self._models[name]

    def feature(self, name: str) -> FeatureRegistryEntry:
        if name not in self._features:
            raise KeyError(f"Unknown feature '{name}'. Available: {sorted(self._features)}")
        return self._features[name]

    def signal(self, name: str) -> SignalRegistryEntry:
        if name not in self._signals:
            raise KeyError(f"Unknown signal '{name}'. Available: {sorted(self._signals)}")
        return self._signals[name]

    def label(self, name: str) -> LabelRegistryEntry:
        if name not in self._labels:
            raise KeyError(f"Unknown label '{name}'. Available: {sorted(self._labels)}")
        return self._labels[name]

    def universe(self, name: str) -> UniverseRegistryEntry:
        if name not in self._universes:
            raise KeyError(f"Unknown universe '{name}'. Available: {sorted(self._universes)}")
        return self._universes[name]

    def asset_class(self, name: str) -> AssetClassEntry:
        if name not in self._asset_classes:
            raise KeyError(f"Unknown asset class '{name}'. Available: {sorted(self._asset_classes)}")
        return self._asset_classes[name]

    def validation(self, name: str) -> ValidationRegistryEntry:
        if name not in self._validations:
            raise KeyError(f"Unknown validation '{name}'. Available: {sorted(self._validations)}")
        return self._validations[name]

    def weighting_scheme(self, name: str) -> WeightingSchemeEntry:
        if name not in self._weighting_schemes:
            raise KeyError(f"Unknown weighting scheme '{name}'. Available: {sorted(self._weighting_schemes)}")
        return self._weighting_schemes[name]

    def preset(self, name: str) -> PresetEntry:
        if name not in self._presets:
            raise KeyError(f"Unknown preset '{name}'. Available: {sorted(self._presets)}")
        return self._presets[name]

    def default_policy(self) -> DefaultPolicy:
        if self._default_policy is None:
            raise RuntimeError("Default policy was not loaded.")
        return self._default_policy

    def compatibility_rules(self) -> tuple[CompatibilityRule, ...]:
        return self._compatibility_rules

    def change_path_entries(self) -> tuple[ChangePathEntry, ...]:
        return self._change_paths
