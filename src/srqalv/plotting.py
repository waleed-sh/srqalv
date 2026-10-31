from __future__ import annotations

import math

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

from srqalv.observables import LambdaChoice

PALETTE = ["#f9dbbd", "#ffa5ab", "#da627d", "#a53860", "#450920"]
EXACT_COLOR = "#1f1f1f"


def configure_matplotlib(*, compact_kernel_labels: bool = False) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 1.25,
            "axes.labelsize": 14.5 if compact_kernel_labels else 17,
            "axes.titlesize": 17,
            "xtick.labelsize": 12 if compact_kernel_labels else 13,
            "ytick.labelsize": 12 if compact_kernel_labels else 13,
            "legend.fontsize": 9.8 if compact_kernel_labels else 11.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axes(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", length=5.5, width=1.1)
    ax.tick_params(axis="both", which="minor", length=3.0, width=0.85)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color(EXACT_COLOR)


def vertex_color(vertex: int) -> str:
    return PALETTE[int(vertex) % len(PALETTE)]


def make_vertex_legend(ax: plt.Axes, *, n_vertices: int = 5) -> None:
    vertex_handles = [
        Line2D(
            [0],
            [0],
            color=vertex_color(vertex),
            marker="o",
            linestyle="-",
            linewidth=2.1,
            markersize=6.0,
            markeredgecolor=EXACT_COLOR,
            markeredgewidth=0.45,
            label=fr"$v_{vertex}$",
        )
        for vertex in range(n_vertices)
    ]
    style_handles = [
        Line2D([0], [0], color=EXACT_COLOR, marker="o", linestyle="None", markersize=6.5, label="exact"),
        Line2D([0], [0], color=EXACT_COLOR, linestyle="-", linewidth=2.4, label="SRQ"),
    ]
    first = ax.legend(
        handles=vertex_handles,
        loc="upper left",
        bbox_to_anchor=(1.015, 1.0),
        borderaxespad=0.0,
        frameon=False,
        ncol=1,
    )
    ax.add_artist(first)
    ax.legend(
        handles=style_handles,
        loc="lower left",
        bbox_to_anchor=(1.015, 0.0),
        borderaxespad=0.0,
        frameon=False,
    )


def plot_iteration_scan(
    *,
    cutoff: int,
    exact_values: dict[int, float],
    rows: list[dict[str, Any]],
    orders: tuple[int, ...],
    out_dir: Path,
    dpi: int,
    filename: str | None = None,
    xlabel: str = r"number of shifted systems $M$",
) -> Path:

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(8.1, 5.35))
    fig.subplots_adjust(right=0.79, bottom=0.16, top=0.97)
    xs = np.asarray(orders, dtype=float)

    for vertex in sorted(exact_values):
        color = vertex_color(vertex)
        y_exact = exact_values[vertex]
        y_srq = np.asarray(
            [
                float(
                    next(
                        row["srq_value"]
                        for row in rows
                        if row["cutoff"] == cutoff and row["vertex"] == vertex and row["M"] == order
                    )
                )
                for order in orders
            ],
            dtype=float,
        )
        ax.plot(
            xs,
            np.full_like(xs, y_exact),
            color=color,
            linestyle="None",
            marker="o",
            markersize=6.2,
            markerfacecolor=color,
            markeredgecolor=EXACT_COLOR,
            markeredgewidth=0.45,
        )
        ax.plot(xs, y_srq, color=color, linestyle="-", linewidth=2.25, solid_capstyle="round")

    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(x)) for x in xs])
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\langle V_v\rangle_\psi$")
    style_axes(ax)
    make_vertex_legend(ax, n_vertices=len(exact_values))

    path = out_dir / (filename or f"k5_cutoff{cutoff}_vertex_volume_vs_M.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_lambda_scan(
    *,
    cutoff: int,
    exact_values: dict[int, float],
    rows: list[dict[str, Any]],
    choices: tuple[LambdaChoice, ...],
    out_dir: Path,
    dpi: int,
    filename: str | None = None,
) -> Path:

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(8.9, 5.55))
    fig.subplots_adjust(right=0.79, bottom=0.25, top=0.97)
    xs = np.arange(len(choices), dtype=float)

    for vertex in sorted(exact_values):
        color = vertex_color(vertex)
        y_exact = exact_values[vertex]
        y_srq = np.asarray(
            [
                float(
                    next(
                        row["srq_value"]
                        for row in rows
                        if row["cutoff"] == cutoff
                        and row["vertex"] == vertex
                        and row["lambda_key"] == choice.key
                    )
                )
                for choice in choices
            ],
            dtype=float,
        )
        ax.plot(
            xs,
            np.full_like(xs, y_exact),
            color=color,
            linestyle="None",
            marker="o",
            markersize=6.2,
            markerfacecolor=color,
            markeredgecolor=EXACT_COLOR,
            markeredgewidth=0.45,
        )
        ax.plot(xs, y_srq, color=color, linestyle="-", linewidth=2.25, solid_capstyle="round")

    ax.set_xticks(xs)
    ax.set_xticklabels([choice.label for choice in choices], rotation=25, ha="right")
    ax.set_xlabel(r"sector-wise choice of $\Lambda_v$")
    ax.set_ylabel(r"$\langle V_v\rangle_\psi$")
    style_axes(ax)
    make_vertex_legend(ax, n_vertices=len(exact_values))

    path = out_dir / (filename or f"k5_cutoff{cutoff}_vertex_volume_vs_lambda.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_kernel_preservation(
    rows: list[dict[str, Any]],
    *,
    cutoff: int,
    out_dir: Path,
    plot_floor: float,
    dpi: int,
    filename: str | None = None,
) -> Path:

    configure_matplotlib(compact_kernel_labels=True)
    data = [row for row in rows if int(row["cutoff"]) == cutoff]
    if not data:
        raise ValueError(f"No kernel rows available for cutoff {cutoff}.")

    orders = np.asarray([int(row["M"]) for row in data], dtype=float)
    kernel_lift = np.asarray([float(row["max_kernel_lift"]) for row in data], dtype=float)
    nonkernel_error = np.asarray([float(row["max_nonkernel_abs_error"]) for row in data], dtype=float)

    fig, ax = plt.subplots(figsize=(7.8, 5.25))
    fig.subplots_adjust(right=0.97, bottom=0.15, top=0.82)
    ax.semilogy(
        orders,
        np.maximum(kernel_lift, plot_floor),
        color=PALETTE[4],
        linestyle="None",
        marker="o",
        markersize=6.4,
        markerfacecolor=PALETTE[4],
        markeredgecolor=EXACT_COLOR,
        markeredgewidth=0.45,
        label="kernel lift",
    )
    ax.semilogy(
        orders,
        np.maximum(nonkernel_error, plot_floor),
        color=PALETTE[2],
        linestyle="-",
        linewidth=2.35,
        marker="s",
        markersize=5.3,
        markerfacecolor="white",
        markeredgewidth=1.1,
        label=r"max non-kernel spectral error",
    )
    ax.axhline(
        plot_floor,
        color=PALETTE[0],
        linestyle=":",
        linewidth=2.0,
        label=fr"zero plotted at $10^{{{int(round(math.log10(plot_floor)))}}}$",
    )

    labeled_orders = np.asarray([5, 50, 100, 150, 200], dtype=float)
    labeled_orders = labeled_orders[(labeled_orders >= np.min(orders)) & (labeled_orders <= np.max(orders))]
    ax.set_xticks(labeled_orders)
    ax.set_xticklabels([str(int(order)) for order in labeled_orders])
    ax.set_xlim(float(np.min(orders)) - 5.0, float(np.max(orders)) + 8.0)
    ax.set_xlabel(r"number of shifted systems $M$")
    ax.set_ylabel("absolute error")
    ax.set_ylim(plot_floor * 0.35, max(float(np.max(nonkernel_error)) * 2.2, plot_floor * 100.0))
    style_axes(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.035),
        borderaxespad=0.0,
        frameon=False,
        handlelength=2.2,
        ncol=2,
        columnspacing=1.5,
        labelspacing=0.45,
    )

    path = out_dir / (filename or f"k5_cutoff{cutoff}_srq_kernel_preservation_exact_zero_modes_vs_M.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
