import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hUtils.tools import check_and_makedirs

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # make compatible with negative or minus sign
plt.rcParams["xtick.direction"] = "in"  # maker ticker direction of x-axis to inner
plt.rcParams["ytick.direction"] = "in"  # maker ticker direction of y-axis to inner


def plot_lines(
    data: pd.DataFrame,
    figsize: tuple[int, int] = (16, 9),
    line_width: float = 2,
    line_style: list | None = None,
    line_color: list | None = None,
    colormap: str = "jet",
    fig_name: str = "fig_name",
    fig_save_type: str = "jpg",
    fig_save_dir: str = ".",
):
    check_and_makedirs(fig_save_dir)
    fig0, ax0 = plt.subplots(figsize=figsize)
    if line_color:
        data.plot.line(ax=ax0, lw=line_width, style=line_style if line_style else "-", color=line_color)
    elif colormap:
        data.plot.line(ax=ax0, lw=line_width, style=line_style if line_style else "-", colormap=colormap)

    xticks = np.arange(0, len(data), len(data) / 10)
    ax0.set_xticks(xticks)
    xticklabels = data.index[xticks.astype(int)]
    ax0.set_xticklabels(xticklabels)

    fig0_name = f"{fig_name}.{fig_save_type}"
    fig0_path = os.path.join(fig_save_dir, fig0_name)
    fig0.savefig(fig0_path, bbox_inches="tight")
    plt.close(fig0)
    return 0
