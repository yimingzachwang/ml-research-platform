"""Deterministic intervention taxonomy classifier.

Classifies config changes or free-text ``tested_change`` descriptions into
standardised intervention types.  No LLM, no disk access, no side effects.

Confidence levels
-----------------
``high``   Structured field_path/operation match, or unambiguous keyword.
``medium`` Keyword match with possible ambiguity.
``low``    Weak signal; conservative fallback to ``unknown_change``.
"""

from __future__ import annotations

__all__ = [
    "INTERVENTION_TYPES",
    "classify_structured",
    "classify_text",
    "classify",
]

INTERVENTION_TYPES: tuple[str, ...] = (
    "regularisation_change",
    "feature_addition",
    "feature_removal",
    "feature_replacement",
    "model_change",
    "validation_change",
    "signal_change",
    "execution_cost_change",
    "portfolio_construction_change",
    "unknown_change",
)


# ---------------------------------------------------------------------------
# Structured classification: (field_path, operation) → intervention_type
# ---------------------------------------------------------------------------

_EXACT: dict[tuple[str, str], str] = {
    ("model.type", "set"): "model_change",
    ("features.entries", "add"): "feature_addition",
    ("features.entries", "remove"): "feature_removal",
    ("features.entries", "replace"): "feature_replacement",
}

# Longest-prefix-first; first match wins
_PREFIX_RULES: list[tuple[str, str]] = [
    ("model.params.alpha", "regularisation_change"),
    ("model.params.l1_ratio", "regularisation_change"),
    ("model.params.lambda", "regularisation_change"),
    ("model.params.penalty", "regularisation_change"),
    ("model.params.c", "regularisation_change"),      # sklearn-style C param (lowercased)
    ("model.params.reg", "regularisation_change"),    # reg_strength, reg_alpha, …
    ("model.params.", "regularisation_change"),        # any other model param → regularisation
    ("model.type", "model_change"),
    ("model.", "model_change"),
    ("features.", "feature_replacement"),              # catch-all for features.*
    ("validation.", "validation_change"),
    ("execution.", "execution_cost_change"),
    ("portfolio.", "portfolio_construction_change"),
    ("signal.", "signal_change"),
]


def classify_structured(field_path: str, operation: str) -> dict[str, str]:
    """Classify from a (field_path, operation) pair.

    Always returns ``intervention_confidence='high'`` because the structured
    source is unambiguous.
    """
    fp = field_path.lower().strip()
    op = operation.lower().strip()

    exact = _EXACT.get((fp, op))
    if exact:
        return {"intervention_type": exact, "intervention_confidence": "high"}

    for prefix, itype in _PREFIX_RULES:
        if fp.startswith(prefix):
            return {"intervention_type": itype, "intervention_confidence": "high"}

    return {"intervention_type": "unknown_change", "intervention_confidence": "high"}


# ---------------------------------------------------------------------------
# Text classification: free-text tested_change → intervention_type
# ---------------------------------------------------------------------------

# Priority-ordered rules; first match wins.
# Each entry: (tuple_of_lower_keywords, intervention_type, confidence)
_TEXT_RULES: list[tuple[tuple[str, ...], str, str]] = [
    # Explicit field-path patterns
    (("features.entries add", "feature entries add"), "feature_addition", "high"),
    (("features.entries remove", "feature entries remove"), "feature_removal", "high"),
    (("features.entries replace", "feature entries replace"), "feature_replacement", "high"),
    # Feature verbs — unambiguous in context
    (
        ("added ", "add new feature", "adding feature", "added feature", "introduce feature"),
        "feature_addition",
        "high",
    ),
    (
        ("removed ", "remove feature", "removing feature", "dropped feature", "drop feature"),
        "feature_removal",
        "high",
    ),
    (
        ("replaced ", "replace feature", "replacing feature", "swap feature", "swapped "),
        "feature_replacement",
        "high",
    ),
    # Explicit regularisation param names
    (
        ("model.params.alpha", ".alpha", "set alpha", "alpha ->", "alpha=", "alpha =", "alpha to"),
        "regularisation_change",
        "high",
    ),
    (("l1_ratio", "lambda =", "lambda=", "set lambda", "penalty =", "penalty="), "regularisation_change", "high"),
    (("regularisation", "regularization"), "regularisation_change", "high"),
    # Model type
    (
        ("model.type", "model type", "changed model type", "switch model", "model changed"),
        "model_change",
        "high",
    ),
    # Validation
    (("n_splits", "walk-forward split", "validation split", "split count"), "validation_change", "high"),
    # Execution costs
    (("transaction cost", "slippage", "commission", "execution cost"), "execution_cost_change", "high"),
    # Portfolio construction
    (
        ("portfolio construction", "position sizing", "position size", "allocation weight"),
        "portfolio_construction_change",
        "high",
    ),
    # Medium-confidence — model family names (may be param or type change)
    (("lasso", "elasticnet", "elastic net", "ridge"), "regularisation_change", "medium"),
    # Medium-confidence — signal/momentum keywords
    (
        ("momentum", "mean reversion", "carry factor", "reversal factor", "signal type"),
        "signal_change",
        "medium",
    ),
    # Medium-confidence — generic validation keywords
    (("validation", "window"), "validation_change", "medium"),
    # Medium-confidence — portfolio
    (("portfolio", "allocation"), "portfolio_construction_change", "medium"),
]


def classify_text(tested_change: str) -> dict[str, str]:
    """Classify from a free-text description; confidence reflects match quality."""
    if not tested_change or not tested_change.strip():
        return {"intervention_type": "unknown_change", "intervention_confidence": "low"}

    lower = tested_change.lower()
    for keywords, itype, conf in _TEXT_RULES:
        if any(kw in lower for kw in keywords):
            return {"intervention_type": itype, "intervention_confidence": conf}

    return {"intervention_type": "unknown_change", "intervention_confidence": "low"}


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------


def classify(
    tested_change: str | None = None,
    field_path: str | None = None,
    operation: str | None = None,
) -> dict[str, str]:
    """Classify using structured info when available, else fall back to text.

    Structured classification (field_path + operation) always yields
    ``confidence='high'``.  Free-text classification yields ``high``,
    ``medium``, or ``low`` depending on keyword strength.

    Returns
    -------
    dict with keys ``intervention_type`` and ``intervention_confidence``.
    """
    if field_path and operation:
        return classify_structured(field_path, operation)
    if tested_change:
        return classify_text(tested_change)
    return {"intervention_type": "unknown_change", "intervention_confidence": "low"}
