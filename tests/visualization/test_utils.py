"""Tests for src/visualization/utils.py — save_figure()."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from src.visualization.utils import save_figure


def _make_fig() -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    return fig


def test_save_figure_writes_file(tmp_path: Path) -> None:
    fig = _make_fig()
    dest = save_figure(fig, tmp_path / "out.png")
    assert dest.exists()


def test_save_figure_closes_by_default(tmp_path: Path) -> None:
    fig = _make_fig()
    num = fig.number
    save_figure(fig, tmp_path / "out.png")
    assert num not in plt.get_fignums()


def test_save_figure_close_false_keeps_open(tmp_path: Path) -> None:
    fig = _make_fig()
    num = fig.number
    try:
        save_figure(fig, tmp_path / "out.png", close=False)
        assert num in plt.get_fignums()
    finally:
        plt.close(fig)


def test_save_figure_forwards_savefig_kwargs(tmp_path: Path) -> None:
    fig = _make_fig()
    with patch.object(fig, "savefig") as mock_savefig:
        save_figure(fig, tmp_path / "out.png", close=False, bbox_inches="tight", dpi=150)
    mock_savefig.assert_called_once()
    _, kwargs = mock_savefig.call_args
    assert kwargs["bbox_inches"] == "tight"
    assert kwargs["dpi"] == 150
    plt.close(fig)


def test_save_figure_no_kwargs_still_works(tmp_path: Path) -> None:
    fig = _make_fig()
    dest = save_figure(fig, tmp_path / "plain.png")
    assert dest.exists()
