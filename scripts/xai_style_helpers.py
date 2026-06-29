"""Reusable helpers for ML-SHAP reporting.

The functions are intentionally small and dependency-light so they can be
copied into analysis notebooks or imported from the skill directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


SUBSCRIPT_TRANSLATION = str.maketrans(
    {
        "0": "\u2080",
        "1": "\u2081",
        "2": "\u2082",
        "3": "\u2083",
        "4": "\u2084",
        "5": "\u2085",
        "6": "\u2086",
        "7": "\u2087",
        "8": "\u2088",
        "9": "\u2089",
        "+": "\u208a",
        "-": "\u208b",
        "=": "\u208c",
        "(": "\u208d",
        ")": "\u208e",
    }
)


def chem_sub(label: object) -> str:
    """Return a display label with digits rendered as Unicode subscripts."""
    return str(label).translate(SUBSCRIPT_TRANSLATION)


def display_feature_names(names: Sequence[object]) -> list[str]:
    """Return figure-ready feature names while preserving input order."""
    return [chem_sub(name) for name in names]


def safe_console(text: object) -> str:
    """Return a Windows-console-safe representation for logs and progress text."""
    return str(text).encode("gbk", errors="backslashreplace").decode("gbk")


def sanitize_svg_u2212(svg_path: str | Path) -> int:
    """Replace U+2212 MINUS SIGN with ASCII '-' in an SVG file.

    Returns the number of replacements made. Missing files raise FileNotFoundError
    so broken chart paths fail visibly.
    """
    path = Path(svg_path)
    text = path.read_text(encoding="utf-8")
    count = text.count("\u2212")
    if count:
        path.write_text(text.replace("\u2212", "-"), encoding="utf-8")
    return count


def setup_plot_style(prefer_times: bool = True, figure_facecolor: str = "#f8f7f4") -> str:
    """Configure matplotlib/seaborn for publication-style SHAP figures.

    Returns the selected font family. Import matplotlib lazily so non-plotting
    workflows can still use the string helpers without extra dependencies.
    """
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("white")
    available = {font.name for font in fm.fontManager.ttflist}
    font_family = "Times New Roman" if prefer_times and "Times New Roman" in available else "DejaVu Sans"

    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.sans-serif": [font_family],
            "axes.unicode_minus": False,
            "svg.fonttype": "path",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": figure_facecolor,
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "grid.alpha": 0.25,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )
    return font_family


def save_chart(fig, output_base: str | Path, dpi: int = 300, facecolor: str | None = None) -> tuple[Path, Path]:
    """Save a matplotlib figure as SVG and PNG, then sanitize SVG minus signs."""
    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    svg_path = base.with_suffix(".svg")
    png_path = base.with_suffix(".png")

    save_kwargs = {"bbox_inches": "tight", "dpi": dpi}
    if facecolor is not None:
        save_kwargs["facecolor"] = facecolor

    fig.savefig(svg_path, format="svg", **save_kwargs)
    fig.savefig(png_path, format="png", **save_kwargs)
    sanitize_svg_u2212(svg_path)
    return svg_path, png_path


def assert_no_u2212(paths: Iterable[str | Path]) -> None:
    """Raise AssertionError if any SVG still contains U+2212."""
    offenders: list[str] = []
    for item in paths:
        path = Path(item)
        if path.suffix.lower() == ".svg" and "\u2212" in path.read_text(encoding="utf-8"):
            offenders.append(str(path))
    if offenders:
        raise AssertionError("SVG files still contain U+2212: " + ", ".join(offenders))
