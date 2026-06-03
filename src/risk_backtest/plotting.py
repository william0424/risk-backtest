"""
Visualization helpers for VaR backtests.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter, MultipleLocator
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "matplotlib is required for risk_backtest.plotting. "
        "Install with: pip install matplotlib"
    ) from exc


# ---------------------------------------------------------------------------
# Style palette (matches the reference report chart)
# ---------------------------------------------------------------------------
_BAR_COLOR        = "#23404A"   # dark teal/slate for daily-return bars
_BAR_OVERSHOOT    = "#B0203D"   # dark red for overshoot bars
_UPPER_LINE       = "#A66BC9"   # soft purple (upper band, +VaR)
_LOWER_LINE       = "#7A1F4A"   # dark maroon (lower band, -VaR)
_GRID_COLOR       = "#D9D9D9"
_TEXT_COLOR       = "#4A4A4A"
_FONT_FAMILY      = "DejaVu Sans"


def plot_var_vs_returns(
    returns: Sequence[float],
    var: Sequence[float],
    dates: Optional[Sequence] = None,
    *,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    figsize: tuple = (16, 4),
    annotate_overshoots: bool = True,
    y_step: Optional[float] = 0.005,
) -> Figure:
    """
    Plot daily returns as thin bars with upper (+VaR) and lower (-VaR) bands.

    Styled to match the reference report chart: clean white background, minimal
    spines, dotted light-grey gridlines, percentage y-axis, thin dark bars,
    purple upper band, maroon lower band, dark-red overshoot bars and optional
    breach annotations.

    Parameters
    ----------
    returns : array-like
        Daily returns in decimal (e.g. 0.01 = 1%).
    var : array-like
        Forecast VaR as a positive loss magnitude (loss threshold = ``-var``).
    dates : array-like, optional
        X-axis values; defaults to integer index.
    title : str, optional
        Plot title.
    ax : matplotlib Axes, optional
        Existing axes to draw on.
    figsize : tuple
        Figure size when ``ax`` is None.
    annotate_overshoots : bool
        If True, label each overshoot with its return value in percent.
    y_step : float, optional
        Tick spacing on the y-axis in decimal (default 0.005 = 0.5%).

    Returns
    -------
    matplotlib.figure.Figure
    """
    returns = np.asarray(returns, dtype=float)
    var = np.asarray(var, dtype=float)
    if returns.shape != var.shape:
        raise ValueError(
            f"returns and var must have same shape, got {returns.shape} vs {var.shape}"
        )

    use_dates = dates is not None
    if use_dates:
        x = np.asarray(dates)
        if len(x) != len(returns):
            raise ValueError("dates length must match returns length")
        try:
            x_num = mdates.date2num(x)
        except Exception:
            x_num = np.arange(len(returns), dtype=float)
            use_dates = False
    else:
        x_num = np.arange(len(returns), dtype=float)

    overshoot_mask = returns < -var

    # ------------------------------------------------------------------
    # Figure / axes
    # ------------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Bars (single dark color, like the reference); overshoot bars in dark red
    bar_width = float(np.median(np.diff(x_num))) * 0.55 if len(x_num) > 1 else 0.55
    ax.bar(
        x_num[~overshoot_mask],
        returns[~overshoot_mask],
        width=bar_width,
        color=_BAR_COLOR,
        linewidth=0,
        zorder=2,
    )
    if overshoot_mask.any():
        ax.bar(
            x_num[overshoot_mask],
            returns[overshoot_mask],
            width=bar_width * 1.4,
            color=_BAR_OVERSHOOT,
            linewidth=0,
            zorder=3,
        )

    # Upper band (+VaR) and lower band (-VaR)
    ax.plot(x_num, var, color=_UPPER_LINE, linewidth=1.6, zorder=4)
    ax.plot(x_num, -var, color=_LOWER_LINE, linewidth=1.6, zorder=4)

    # ------------------------------------------------------------------
    # Overshoot annotations
    # ------------------------------------------------------------------
    if annotate_overshoots and overshoot_mask.any():
        for xi, ri in zip(x_num[overshoot_mask], returns[overshoot_mask]):
            ax.annotate(
                f"{ri * 100:.2f}%",
                xy=(xi, ri),
                xytext=(0, -10),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8,
                color=_TEXT_COLOR,
                fontfamily=_FONT_FAMILY,
            )
            ax.plot(xi, ri, marker="o", markersize=4,
                    color=_BAR_COLOR, zorder=5)

    # ------------------------------------------------------------------
    # Axes cosmetics
    # ------------------------------------------------------------------
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_GRID_COLOR)
    ax.spines["bottom"].set_linewidth(0.8)

    # Y-axis as percentage with sparse 0.5% gridlines
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.1f}%"))
    if y_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_step))
    ax.tick_params(axis="y", colors=_TEXT_COLOR, length=0, labelsize=10)
    ax.tick_params(axis="x", colors=_TEXT_COLOR, length=0, labelsize=10)

    # Light dotted gridlines
    ax.grid(True, which="major", axis="both",
            color=_GRID_COLOR, linestyle=":", linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)

    # X-axis date formatting — sparse month labels
    if use_dates:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # No axis labels in the reference; keep optional title
    ax.set_xlabel("")
    ax.set_ylabel("")
    if title:
        ax.set_title(title, color=_TEXT_COLOR, fontsize=12, loc="left", pad=10,
                     fontfamily=_FONT_FAMILY)

    ax.margins(x=0.005)
    fig.tight_layout()
    return fig
