"""Registry context builder for the intent parser prompt.

Converts a ResearchConfigRegistry into a compact, prompt-embeddable dict
of valid registry values.  Only registry-supported values are included —
this context is what the LLM is allowed to reference.
"""

from __future__ import annotations

from typing import Any

from src.agent.registry.registry_schema import ResearchConfigRegistry


def build_registry_context(
    registry: ResearchConfigRegistry,
    compact: bool = True,
) -> dict[str, Any]:
    """Build a prompt-embeddable dict of valid registry values.

    Args:
        registry: The loaded ResearchConfigRegistry.
        compact:  If True (default), omit verbose descriptions and include
                  only names/values needed for constraint guidance.  Compact
                  context is deterministic and safe to embed in an LLM prompt.

    Returns:
        A plain dict with keys:
            presets, asset_classes, universes, models, features,
            feature_families, labels, signals, validation_protocols,
            diagnostics, default_mode.
    """
    if compact:
        return _build_compact(registry)
    return _build_full(registry)


def _build_compact(registry: ResearchConfigRegistry) -> dict[str, Any]:
    presets = sorted(registry.preset_names())
    asset_classes = sorted(registry.asset_class_names())
    universes = sorted(registry.universe_names())
    models = sorted(registry.model_names())
    features = sorted(registry.feature_names())
    feature_families = _extract_feature_families(registry)
    labels = sorted(registry.label_names())
    signals = sorted(registry.signal_names())
    validation_protocols = sorted(registry.validation_names())
    diagnostics = sorted(registry.diagnostic_names())

    try:
        default_mode = registry.default_policy().default_mode
    except Exception:
        default_mode = "recommended"

    return {
        "presets": presets,
        "asset_classes": asset_classes,
        "universes": universes,
        "models": models,
        "features": features,
        "feature_families": feature_families,
        "labels": labels,
        "signals": signals,
        "validation_protocols": validation_protocols,
        "diagnostics": diagnostics,
        "default_mode": default_mode,
    }


def _build_full(registry: ResearchConfigRegistry) -> dict[str, Any]:
    """Full context with display names and descriptions."""
    compact = _build_compact(registry)

    model_details = []
    for name in sorted(registry.model_names()):
        try:
            e = registry.model(name)
            model_details.append({
                "name": e.name,
                "display_name": e.display_name,
                "description": e.description,
            })
        except Exception:
            model_details.append({"name": name})

    preset_details = []
    for name in sorted(registry.preset_names()):
        try:
            e = registry.preset(name)
            preset_details.append({
                "name": e.name,
                "display_name": e.display_name,
                "description": e.description,
                "asset_class": e.asset_class,
                "universe": e.universe,
            })
        except Exception:
            preset_details.append({"name": name})

    compact["model_details"] = model_details
    compact["preset_details"] = preset_details
    return compact


def _extract_feature_families(registry: ResearchConfigRegistry) -> list[str]:
    """Collect unique family names from all FeatureRegistryEntries."""
    families: set[str] = set()
    for name in registry.feature_names():
        try:
            entry = registry.feature(name)
            if entry.family:
                families.add(entry.family)
        except Exception:
            pass
    return sorted(families)
