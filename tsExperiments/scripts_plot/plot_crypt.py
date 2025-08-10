#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid-of-plots comparing four probabilistic-forecast models (Tactis2,
Tempflow, TimeGrad, TimeMCL) on four crypto pairs.  Each panel shows the
K = 4 head trajectories, their mean, and the observations.

Changes v2
──────────
•  removed the “if p > 0.05” guard → all heads are drawn;
•  introduced min_alpha / max_alpha ramp for readability.
"""
# —————————————————————————————————————————
# Imports & global MPL settings
# —————————————————————————————————————————
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import matplotlib.dates  as mdates
import matplotlib       as mpl
from matplotlib.lines import Line2D
import os
import sys
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(os.path.dirname(os.environ["PROJECT_ROOT"]))

plt.rcParams["figure.dpi"]   = 300
plt.rcParams["savefig.dpi"]  = 300

mpl.rcParams["text.usetex"]        = True
mpl.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath}"
    # r"\usepackage[T1]{fontenc}"
)

import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')

# —————————————————————————————————————————
# Paths, models, dimensions
# —————————————————————————————————————————
base_path = os.environ["PROJECT_ROOT"] + "/tsExperiments/scripts_plot"
base_path = Path(base_path)
models       = ["timeMCL"]  # Only plot TimeMCL
pretty       = {"tactis2": "Tactis2", "transformer_tempflow": "Trf.Tempflow",
                "timeGrad": "TimeGrad", "timeMCL": "TimeMCL"}
# dimensions   = [1, 2, 5, 13]
# dimensions   = [1, 2, 5, 10]
dimensions = [1, 2, 4, 5]
crypto_names = {
    0: "ADA-USD",
    1: "BCH-USD",
    2: "BTC-USD",
    3: "DOGE-USD",
    4: "EOS-USD",
    5: "ETC-USD",
    6: "ETH-USD",
    7: "LINK-USD",
    8: "LTC-USD",
    9: "MATIC-USD",
    10: "SOL-USD",
    11: "VET-USD",
    12: "XLM-USD",
    13: "XMR-USD",
    14: "XRP-USD",
}
date = "2025-01-04"

pkl_files = {(m, d): base_path / m / f"{d}" / f"{d}.pkl"
             for m in models for d in dimensions}

colors = {
    "tactis2":  "#7E3FBE",   # vivid purple      (was #C5B4E3)
    "tempflow": "#D79E00",   # dark golden-yellow (was #FCD299)
    "transformer_tempflow": "#D79E00",   # dark golden-yellow (was #FCD299)
    "timeGrad": "lightcoral",
    "timeMCL":  "blue",
}

# opacity range for hypotheses
min_alpha, max_alpha = 0.25, 0.90

# y-axis limits (+10 %)
manual_limits = {
    1: {"min": 450,   "max": 490},
    2: {"min": 96000, "max": 101000},
    4: { "min": 0.81, "max": 0.98},
    5: {"min": 26.5,  "max": 29},
    10:{"min": 206,   "max": 225},
    11:{"min": 0.0475,   "max": 0.0545},
    13:{"min": 190,   "max": 208},
    14:{"min": 2.36,  "max": 2.55},
}
dimension_limits = {
    d: {"min": v["min"] - 0.10*(v["max"]-v["min"]),
        "max": v["max"] + 0.10*(v["max"]-v["min"])}
    for d, v in manual_limits.items()
}

# —————————————————————————————————————————
# FIGURE & GRID
# —————————————————————————————————————————
fig, axs = plt.subplots(
    1, 4, figsize=(30, 7), facecolor="white",
    gridspec_kw={"wspace": 0.28, "hspace": 0.05},  # Increased wspace for more space between plots
)

for j, dim in enumerate(dimensions):  # Only one row, so j is the column
    ax = axs[j]
    model = "timeMCL"
    # ── load data ───────────────────────────────────────────────
    with open(pkl_files[(model, dim)], "rb") as f:
        data = pickle.load(f)
    # ── panel style ─────────────────────────────────────────────
    ax.set_facecolor("lightgrey")
    ax.grid(True, which="both", linewidth=2.0, alpha=1.0, color="white")
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)
    # context & forecast windows
    ax.axvspan(data["historical_dates"][0], data["start_date"],
               color="gray", alpha=0.1, zorder=0)
    ax.axvspan(data["start_date"], data["forecast_dates"][-1],
               color="lightyellow", alpha=0.5, zorder=0)
    # observations
    ax.plot(data["historical_dates"], data["historical_values"],
            color="black", linewidth=2, zorder=2)
    # ── plot hypotheses (all heads) ─────────────────────────────
    if data["probabilities"] is not None:
        probs = data["probabilities"][:, 0]
        p_max = probs.max() if probs.max() > 0 else 1.0  # avoid /0
        for k, p in enumerate(probs):
            alpha = min_alpha + (max_alpha - min_alpha) * p / p_max
            ax.plot(data["forecast_dates"], data["forecast_array"][k],
                    color=colors[model], alpha=alpha,
                    linewidth=2, zorder=3)
    else:  # equal opacity if probs not provided
        for k in range(data["forecast_array"].shape[0]):
            ax.plot(data["forecast_dates"], data["forecast_array"][k],
                    color=colors[model], alpha=0.6,
                    linewidth=2, zorder=3)
    # mean forecast (only this line for TimeMCL)
    ax.plot(data["forecast_dates"], data["mean_forecast"],
            color=colors[model], linewidth=3,
            linestyle="--", zorder=4)
    # limits
    ax.set_ylim(dimension_limits[dim]["min"], dimension_limits[dim]["max"])
    # x-axis ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(axis="x", rotation=45, labelsize=18)
    ticks = ax.get_xticks()
    if len(ticks) > 0:
        ax.set_xticks(ticks[::2])
    # y-axis ticks & row labels
    ax.set_ylabel(crypto_names[dim], fontsize=22)
    ax.tick_params(axis="y", labelsize=18)
    # column titles
    # ax.set_title(rf"\texttt{{{pretty[model]}}}",
    #              fontsize=25, pad=20, fontweight="bold")  # Remove individual subplot titles

# —————————————————————————————————————————
# SINGLE (GLOBAL) LEGEND
# —————————————————————————————————————————
fig.suptitle(r"\texttt{TimeMCL} forecasts", fontsize=32, y=0.98)  # Global title
handles = [
    Line2D([0], [0], color="black", linewidth=2, label="Observations"),
    Line2D([0], [0], color=colors["timeMCL"], linewidth=3, linestyle="--",
           label=r"\texttt{TimeMCL} mean"),
    Line2D([0], [0], color=colors["timeMCL"], linewidth=2, linestyle="-",
           label=r"\texttt{TimeMCL} hypotheses"),
]
fig.legend(
    handles=handles,
    loc="center",
    bbox_to_anchor=(0.5, 1.04),  # Move legend above the title
    ncol=3,
    fontsize=28,  # Larger legend font
    frameon=False
)

# —————————————————————————————————————————
# layout, save, show
# —————————————————————————————————————————
plt.tight_layout(rect=[0, 0, 1, 0.96])         # leave space for legend
save_path = base_path / f"comparison_single_timeMCL_{date}.png"
plt.savefig(save_path, dpi=300,
            bbox_inches="tight", facecolor="white")
print(f"Figure saved to {save_path}")
plt.show()