"""Phase 6A: Candidate config generator.

Generates YAML config dicts from SearchCandidate specs and writes them to
configs/experiments/search/.  Produces the search manifest (manifest.json).

Governance:
  - All candidate configs are derived from the registry-supported base config.
  - No LLM, no invented fields, no execution.
  - Configs land in configs/experiments/search/ (separate from live configs).
  - Baseline candidate uses the original config path unchanged.
"""

from __future__ import annotations

import copy
import datetime
import json
from pathlib import Path
from typing import Any

from .search_space import (
    BASE_CONFIG_PATH,
    BASE_EXPERIMENT,
    SearchCandidate,
    SearchSpace,
)

# Output dirs (relative to project root; caller can override)
_DEFAULT_SEARCH_CONFIGS_DIR = Path("configs/experiments/search")
_DEFAULT_OUTPUT_DIR = Path("results/strategy_search")


# ---------------------------------------------------------------------------
# Base config loading
# ---------------------------------------------------------------------------


def load_base_config(base_config_path: Path | str | None = None) -> dict[str, Any]:
    """Load the base experiment YAML as a dict (requires PyYAML)."""
    import yaml  # optional runtime dep — only needed for config generation

    path = Path(base_config_path) if base_config_path else Path(BASE_CONFIG_PATH)
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------


def build_candidate_config(
    candidate: SearchCandidate,
    base_config: dict[str, Any],
) -> dict[str, Any]:
    """Build a full YAML config dict for a candidate.

    Deep-copies the base config and applies model / features / cost overrides.
    Baseline candidate returns a copy of the base config with no overrides
    (name and metadata updated only).
    """
    cfg = copy.deepcopy(base_config)

    cfg["name"] = candidate.candidate_id

    if candidate.is_baseline:
        # Baseline: preserve everything; just stamp metadata
        cfg.setdefault("metadata", {}).update(_candidate_metadata(candidate))
        cfg["tags"] = _candidate_tags(candidate, existing=cfg.get("tags", []))
        return cfg

    # --- Model override ---
    cfg["model"] = {
        "type": candidate.model.type,
        "params": copy.deepcopy(candidate.model.params),
    }

    # --- Feature override ---
    cfg.setdefault("features", {})["entries"] = [
        e.to_dict() for e in candidate.feature_set.entries
    ]
    # Keep the ticker field from base (needed for template resolution)
    if "ticker" not in cfg["features"] and "ticker" in base_config.get("features", {}):
        cfg["features"]["ticker"] = base_config["features"]["ticker"]

    # --- Cost override ---
    cfg.setdefault("execution", {})["transaction_cost_bps"] = float(candidate.cost_bps)

    # --- Metadata ---
    cfg.setdefault("metadata", {}).update(_candidate_metadata(candidate))
    cfg["tags"] = _candidate_tags(candidate)

    return cfg


def _candidate_metadata(c: SearchCandidate) -> dict[str, Any]:
    return {
        "search_phase": "6A",
        "candidate_id": c.candidate_id,
        "base_experiment": c.base_experiment,
        "model_id": c.model.id,
        "feature_set_id": c.feature_set.id,
        "cost_bps": c.cost_bps,
        "is_baseline": c.is_baseline,
    }


def _candidate_tags(c: SearchCandidate, existing: list[str] | None = None) -> list[str]:
    if c.is_baseline:
        tags = list(existing or [])
        for t in ("search_6a", "baseline"):
            if t not in tags:
                tags.append(t)
        return tags
    return [
        "search_6a",
        f"model_{c.model.id}",
        f"features_{c.feature_set.id}",
        f"cost_{c.cost_bps}bps",
        c.model.type.lower(),
    ]


# ---------------------------------------------------------------------------
# File writing
# ---------------------------------------------------------------------------


def write_candidate_config(
    candidate: SearchCandidate,
    cfg: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Write candidate config YAML to output_dir / {candidate_id}.yaml."""
    import yaml

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{candidate.candidate_id}.yaml"
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return out_path


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_manifest(
    candidates: list[SearchCandidate],
    config_paths: dict[str, Path],  # candidate_id → written config path
    results_base: Path | str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the search manifest dict."""
    rb = Path(results_base) if results_base else _DEFAULT_OUTPUT_DIR
    ts = generated_at or datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    candidate_entries = []
    for c in candidates:
        cfg_path = config_paths.get(c.candidate_id)
        entry = c.to_dict()
        entry["config_path"] = str(cfg_path) if cfg_path else None
        entry["result_path"] = str(rb / "experiments" / c.candidate_id)
        candidate_entries.append(entry)

    from .search_space import SearchSpace

    return {
        "search_phase": "6A",
        "generated_at": ts,
        "base_experiment": BASE_EXPERIMENT,
        "base_config_path": BASE_CONFIG_PATH,
        "n_candidates": len(candidates),
        "n_generated": sum(1 for c in candidates if not c.is_baseline),
        "search_space": SearchSpace.summary(),
        "candidates": candidate_entries,
    }


def write_manifest(manifest: dict[str, Any], output_dir: Path) -> Path:
    """Write manifest.json to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return path


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def generate_all_configs(
    configs_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    base_config_path: Path | str | None = None,
    dry_run: bool = False,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Generate all 46 candidate configs + manifest.

    Returns (manifest, config_paths) where config_paths maps candidate_id → Path.
    With dry_run=True nothing is written to disk.
    """
    configs_out = Path(configs_dir) if configs_dir else _DEFAULT_SEARCH_CONFIGS_DIR
    results_out = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_DIR

    base_cfg = load_base_config(base_config_path)
    candidates = SearchSpace.all_candidates_with_baseline()

    config_paths: dict[str, Path] = {}
    for c in candidates:
        if c.is_baseline:
            # Baseline points to the existing config file
            config_paths[c.candidate_id] = Path(BASE_CONFIG_PATH)
            continue
        cfg = build_candidate_config(c, base_cfg)
        if not dry_run:
            path = write_candidate_config(c, cfg, configs_out)
        else:
            path = configs_out / f"{c.candidate_id}.yaml"
        config_paths[c.candidate_id] = path

    manifest = build_manifest(candidates, config_paths, results_base=results_out)
    if not dry_run:
        write_manifest(manifest, results_out)

    return manifest, config_paths
