"""Phase 6A: Governed Strategy Search and Robustness Leaderboard.

Public surface:
    SearchSpace          — deterministic search space (models × features × costs)
    SearchCandidate      — one candidate in the sweep
    generate_all_configs — write candidate YAMLs + manifest
    build_leaderboard    — build ranked leaderboard from artefacts
    compute_robustness_score — Phase 6A scoring formula
    build_full_report    — full markdown report
    build_website_summary — concise website-ready summary
    generate_all_plots   — all 5 required plots
"""

from .search_space import SearchSpace, SearchCandidate
from .candidate_generator import generate_all_configs, build_candidate_config
from .robustness_score import compute_robustness_score, RobustnessComponents, FORMULA_DESCRIPTION
from .leaderboard import build_leaderboard, LeaderboardRow, leaderboard_to_dict
from .report import build_full_report, build_website_summary, write_full_report, write_website_summary
from .plots import generate_all_plots

__all__ = [
    "SearchSpace",
    "SearchCandidate",
    "generate_all_configs",
    "build_candidate_config",
    "compute_robustness_score",
    "RobustnessComponents",
    "FORMULA_DESCRIPTION",
    "build_leaderboard",
    "LeaderboardRow",
    "leaderboard_to_dict",
    "build_full_report",
    "build_website_summary",
    "write_full_report",
    "write_website_summary",
    "generate_all_plots",
]
