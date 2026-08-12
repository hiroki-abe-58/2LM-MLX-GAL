"""記事用グラフの共通設定 (配色と日本語フォント).

matplotlib の既定フォント (DejaVu Sans) には日本語の字が無く、
ラベルに日本語を入れると警告を出したうえで豆腐になる。
環境ごとに入っているフォントが違うので、候補から先に見つかったものを使う。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

BACKGROUND = "#0b1024"
SERIES_COLORS = ("#6ea8ff", "#ff7ba8", "#8ce99a", "#ffd43b")

_JP_FONT_CANDIDATES = (
    "Hiragino Sans",  # macOS
    "Yu Gothic",  # Windows
    "Meiryo",  # Windows
    "Noto Sans CJK JP",  # Linux
    "IPAexGothic",
    "Arial Unicode MS",
)


def japanese_font() -> str | None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    return next((name for name in _JP_FONT_CANDIDATES if name in available), None)


def apply_style() -> None:
    plt.style.use("dark_background")
    font = japanese_font()
    if font:
        plt.rcParams["font.family"] = font
    else:
        print("警告: 日本語フォントが見つかりません。ラベルが豆腐になります。")
    # マイナス記号だけは日本語フォントに無いことがあるので ASCII のハイフンにする。
    plt.rcParams["axes.unicode_minus"] = False


def style_axes(ax) -> None:
    ax.set_facecolor(BACKGROUND)
    ax.grid(alpha=0.15)
    for spine in ax.spines.values():
        spine.set_alpha(0.25)


def dark_figure(nrows: int = 1, figsize: tuple[float, float] = (8, 4.2), dpi: int = 160):
    apply_style()
    fig, axes = plt.subplots(nrows, 1, figsize=figsize, dpi=dpi, sharex=True)
    fig.patch.set_facecolor(BACKGROUND)
    axes = [axes] if nrows == 1 else list(axes)
    for ax in axes:
        style_axes(ax)
    return fig, axes


def dark_axes(figsize: tuple[float, float] = (8, 4.2), dpi: int = 160):
    fig, axes = dark_figure(1, figsize, dpi)
    return fig, axes[0]
