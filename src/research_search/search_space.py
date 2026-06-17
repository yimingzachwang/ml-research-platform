"""Phase 6A: Governed Strategy Search — deterministic search space definition.

All search dimensions are defined here as frozen module-level constants.
``SearchSpace.all_candidates()`` is fully deterministic: same list, same order,
every call.  No LLM, no randomness, no disk access.

Search space:
  Models  : RidgeRegression, LassoRegression, ElasticNetRegression
  Features: baseline (15), trend_only (10), trend_vol (12), trend_mr (12), trend_vol_mktstr (13)
  Costs   : 5, 10, 20 bps
  Universe: canonical_ml_multi_asset (unchanged)
  Total   : 3 × 5 × 3 = 45 generated candidates + 1 baseline = 46 rows
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Primitive specs
# ---------------------------------------------------------------------------


@dataclass
class FeatureEntry:
    name: str
    type: str
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.type, "params": dict(self.params)}


@dataclass
class ModelSpec:
    id: str            # slug used in candidate_id
    type: str          # RidgeRegression / LassoRegression / ElasticNetRegression
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "params": dict(self.params)}


@dataclass
class FeatureSetSpec:
    id: str             # slug used in candidate_id
    name: str           # human-readable label
    family_description: str
    entries: list[FeatureEntry]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "family_description": self.family_description,
            "n_features": len(self.entries),
            "feature_types": sorted({e.type for e in self.entries}),
        }


@dataclass
class SearchCandidate:
    candidate_id: str
    model: ModelSpec
    feature_set: FeatureSetSpec
    cost_bps: int
    base_experiment: str
    is_baseline: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "model_id": self.model.id,
            "model_type": self.model.type,
            "feature_set_id": self.feature_set.id,
            "feature_set_name": self.feature_set.name,
            "cost_bps": self.cost_bps,
            "base_experiment": self.base_experiment,
            "is_baseline": self.is_baseline,
        }


# ---------------------------------------------------------------------------
# Feature entries (sourced from canonical_ml_multi_asset.yaml)
# ---------------------------------------------------------------------------

# Trend family features
_F_MOM_5     = FeatureEntry("mom_5",     "momentum",             {"lookback": 5})
_F_MOM_20    = FeatureEntry("mom_20",    "momentum",             {"lookback": 20})
_F_MOM_60    = FeatureEntry("mom_60",    "momentum",             {"lookback": 60})
_F_MOM_252   = FeatureEntry("mom_252",   "momentum",             {"lookback": 252})
_F_TREND_20  = FeatureEntry("trend_20",  "trend_strength",       {"window": 20})
_F_TPERS_20  = FeatureEntry("trend_persist_20d", "trend_persistence", {"window": 20})
_F_BRK_63    = FeatureEntry("breakout_63d", "breakout_strength", {"window": 63})
_F_SHMOM_63  = FeatureEntry("sharpe_mom_63d",  "risk_adjusted_momentum",
                             {"mom_window": 63,  "vol_window": 63})
_F_SHMOM_126 = FeatureEntry("sharpe_mom_126d", "risk_adjusted_momentum",
                             {"mom_window": 126, "vol_window": 63})
_F_SHMOM     = FeatureEntry("sharpe_mom_252d", "risk_adjusted_momentum",
                             {"mom_window": 252, "vol_window": 63})

# Volatility family features
_F_VOL_21    = FeatureEntry("vol_21",    "rolling_volatility",   {"window": 21})
_F_VOLCMP    = FeatureEntry("vol_compress_21_63", "vol_compression",
                             {"short_window": 21, "long_window": 63})

# Mean-Reversion family features
_F_ZSC_20    = FeatureEntry("zscore_20", "rolling_zscore",       {"window": 20})
_F_DDDIST    = FeatureEntry("drawdown_dist_252d", "drawdown_distance", {"window": 252})

# Market Structure family features
_F_BETA_60   = FeatureEntry("beta_60d",  "rolling_beta",
                             {"window": 60, "market_ticker": "SPY"})

# Trend-only group (10 features)
_TREND_ENTRIES: list[FeatureEntry] = [
    _F_MOM_5, _F_MOM_20, _F_MOM_60, _F_MOM_252,
    _F_TREND_20, _F_TPERS_20, _F_BRK_63,
    _F_SHMOM_63, _F_SHMOM_126, _F_SHMOM,
]


# ---------------------------------------------------------------------------
# Registry-supported model types
# ---------------------------------------------------------------------------

SUPPORTED_MODEL_TYPES: frozenset[str] = frozenset({
    "RidgeRegression",
    "LassoRegression",
    "ElasticNetRegression",
})

# ---------------------------------------------------------------------------
# Registry-supported feature types
# ---------------------------------------------------------------------------

SUPPORTED_FEATURE_TYPES: frozenset[str] = frozenset({
    "momentum",
    "trend_strength",
    "trend_persistence",
    "breakout_strength",
    "risk_adjusted_momentum",
    "rolling_volatility",
    "vol_compression",
    "rolling_zscore",
    "drawdown_distance",
    "rolling_beta",
})


# ---------------------------------------------------------------------------
# Frozen search dimensions
# ---------------------------------------------------------------------------

MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("ridge",       "RidgeRegression",      {"alpha": 1.0}),
    ModelSpec("lasso",       "LassoRegression",      {"alpha": 0.1}),
    ModelSpec("elasticnet",  "ElasticNetRegression", {"alpha": 0.1, "l1_ratio": 0.5}),
)

FEATURE_SETS: tuple[FeatureSetSpec, ...] = (
    FeatureSetSpec(
        id="baseline",
        name="Baseline canonical",
        family_description="Trend + Volatility + Mean-Reversion + Market Structure",
        entries=[
            _F_MOM_5, _F_MOM_20, _F_MOM_60, _F_MOM_252,
            _F_VOL_21,
            _F_ZSC_20,
            _F_TREND_20,
            _F_TPERS_20, _F_BRK_63, _F_DDDIST,
            _F_VOLCMP,
            _F_BETA_60,
            _F_SHMOM_63, _F_SHMOM_126, _F_SHMOM,
        ],
    ),
    FeatureSetSpec(
        id="trend_only",
        name="Trend only",
        family_description="Trend",
        entries=list(_TREND_ENTRIES),
    ),
    FeatureSetSpec(
        id="trend_vol",
        name="Trend + Volatility",
        family_description="Trend + Volatility",
        entries=list(_TREND_ENTRIES) + [_F_VOL_21, _F_VOLCMP],
    ),
    FeatureSetSpec(
        id="trend_mr",
        name="Trend + Mean-Reversion",
        family_description="Trend + Mean-Reversion",
        entries=list(_TREND_ENTRIES) + [_F_ZSC_20, _F_DDDIST],
    ),
    FeatureSetSpec(
        id="trend_vol_mktstr",
        name="Trend + Volatility + Market Structure",
        family_description="Trend + Volatility + Market Structure",
        entries=list(_TREND_ENTRIES) + [_F_VOL_21, _F_VOLCMP, _F_BETA_60],
    ),
)

COST_BPS: tuple[int, ...] = (5, 10, 20)

BASE_EXPERIMENT: str = "canonical_ml_multi_asset"
BASE_CONFIG_PATH: str = f"configs/experiments/{BASE_EXPERIMENT}.yaml"

# Total generated candidates (excludes baseline)
N_GENERATED: int = len(MODELS) * len(FEATURE_SETS) * len(COST_BPS)
# Total leaderboard rows (includes baseline)
N_TOTAL: int = N_GENERATED + 1


# ---------------------------------------------------------------------------
# Candidate enumeration
# ---------------------------------------------------------------------------


def _make_candidate_id(model: ModelSpec, fs: FeatureSetSpec, cost: int) -> str:
    return f"search_{model.id}_{fs.id}_{cost}bps"


class SearchSpace:
    """Deterministic, registry-bounded search space for Phase 6A.

    All methods are pure functions of the module-level constants above.
    Calling ``all_candidates()`` twice returns identical lists.
    """

    models: tuple[ModelSpec, ...] = MODELS
    feature_sets: tuple[FeatureSetSpec, ...] = FEATURE_SETS
    cost_bps: tuple[int, ...] = COST_BPS
    base_experiment: str = BASE_EXPERIMENT

    @classmethod
    def all_candidates(cls) -> list[SearchCandidate]:
        """Return all 45 generated candidates in deterministic order.

        Order: models × feature_sets × costs (outer to inner).
        Does NOT include the baseline row (use ``baseline_candidate()``).
        """
        candidates = []
        for model in cls.models:
            for fs in cls.feature_sets:
                for cost in cls.cost_bps:
                    candidates.append(SearchCandidate(
                        candidate_id=_make_candidate_id(model, fs, cost),
                        model=model,
                        feature_set=fs,
                        cost_bps=cost,
                        base_experiment=cls.base_experiment,
                        is_baseline=False,
                    ))
        return candidates

    @classmethod
    def baseline_candidate(cls) -> SearchCandidate:
        """Return the baseline candidate (canonical_ml_multi_asset, unchanged)."""
        baseline_fs = cls.feature_sets[0]  # "baseline" feature set
        baseline_model = cls.models[0]     # "ridge"
        return SearchCandidate(
            candidate_id=BASE_EXPERIMENT,
            model=baseline_model,
            feature_set=baseline_fs,
            cost_bps=5,
            base_experiment=BASE_EXPERIMENT,
            is_baseline=True,
        )

    @classmethod
    def all_candidates_with_baseline(cls) -> list[SearchCandidate]:
        """Return baseline (index 0) followed by all 45 generated candidates."""
        return [cls.baseline_candidate()] + cls.all_candidates()

    @classmethod
    def candidate_by_id(cls, candidate_id: str) -> SearchCandidate | None:
        """Look up a candidate by ID. Returns None if not found."""
        for c in cls.all_candidates_with_baseline():
            if c.candidate_id == candidate_id:
                return c
        return None

    @classmethod
    def summary(cls) -> dict[str, Any]:
        return {
            "models": [m.to_dict() for m in cls.models],
            "feature_sets": [fs.to_dict() for fs in cls.feature_sets],
            "cost_bps": list(cls.cost_bps),
            "base_experiment": cls.base_experiment,
            "n_generated": N_GENERATED,
            "n_total_with_baseline": N_TOTAL,
        }
