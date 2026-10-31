from __future__ import annotations

import csv
import math
import time

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from srqalv.plotting import configure_matplotlib
from srqalv.plotting import style_axes
from srqalv.runs.mc_samples import build_sector
from srqalv.runs.mc_samples import bytes_to_gib
from srqalv.runs.mc_samples import srq_action_on_ones

Array = np.ndarray


def common_draws(*, channel_dims: tuple[int, ...], chains: int, sample_count: int, seed: int) -> tuple[Array, ...]:
    rng = np.random.default_rng(int(seed))
    return tuple(
        rng.integers(0, int(dim), size=(int(chains), int(sample_count)), endpoint=False)
        for dim in channel_dims
    )


def summarize_vertex_samples(values: Array, draws: Array) -> tuple[float, float]:
    sampled = np.asarray(values, dtype=np.float64)[draws]
    chain_means = np.mean(sampled, axis=1)
    mean = float(np.mean(chain_means))
    stderr = (
        float(np.std(chain_means, ddof=1) / math.sqrt(chain_means.size))
        if chain_means.size > 1
        else float("nan")
    )
    return mean, stderr


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cutoff",
                "M",
                "sample_count",
                "chains",
                "total_samples",
                "vertex",
                "local_dim",
                "lambda_v",
                "mean",
                "stderr",
                "max_relres",
                "srq_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_vs_m(rows: list[dict[str, float | int]], *, cutoff: int, sample_count: int, chains: int, out_dir: Path, dpi: int) -> Path:
    colors = ["#f9dbbd", "#ffa5ab", "#da627d", "#a53860", "#450920"]
    fig, ax = plt.subplots(figsize=(8.4, 5.35))
    fig.subplots_adjust(left=0.13, right=0.82, bottom=0.16, top=0.97)
    for vertex in range(5):
        vals = sorted((row for row in rows if int(row["vertex"]) == vertex), key=lambda row: int(row["M"]))
        xs = np.asarray([int(row["M"]) for row in vals], dtype=np.float64)
        means = np.asarray([float(row["mean"]) for row in vals], dtype=np.float64)
        errors = np.asarray([float(row["stderr"]) for row in vals], dtype=np.float64)
        color = colors[vertex]
        ax.plot(xs, means, marker="o", linewidth=2.35, markersize=5.5, color=color, label=fr"$v_{vertex}$")
        ax.fill_between(xs, means - errors, means + errors, color=color, alpha=0.20, linewidth=0.0)
    ticks = sorted({int(row["M"]) for row in rows})
    display_ticks = [tick for tick in (5, 10, 20, 50, 100, 200, 500) if tick in ticks]
    ax.set_xscale("log")
    ax.set_xlim(float(ticks[0]) * 0.88, float(ticks[-1]) * 1.12)
    ax.set_xticks(display_ticks)
    ax.set_xticklabels([str(tick) for tick in display_ticks])
    ax.set_xlabel(r"number of shifted systems $M$")
    ax.set_ylabel(r"MC estimate of $\langle V_v\rangle$")
    style_axes(ax)
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.50),
        borderaxespad=0.0,
        handlelength=1.5,
        handletextpad=0.45,
        labelspacing=0.58,
    )
    path = out_dir / f"k5_cutoff{cutoff}_banded_srq_mc_vs_M_N{int(sample_count) * int(chains)}.png"
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    return path


def run(args) -> None:
    if args.cutoff <= 0:
        raise ValueError("experiment.cutoff must be positive")
    if args.sample_count <= 0:
        raise ValueError("experiment.sample_count must be positive")
    if args.chains <= 0:
        raise ValueError("experiment.chains must be positive")
    configure_matplotlib()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    m_values = tuple(sorted(set(int(m) for m in args.orders)))

    script_start = time.perf_counter()
    sector = build_sector(
        cutoff=int(args.cutoff),
        edge_labels=tuple(int(x) for x in args.fixed_edge_labels),
        embedding_seed=int(args.embedding_seed),
    )
    draws = common_draws(
        channel_dims=sector.channel_dims,
        chains=int(args.chains),
        sample_count=int(args.sample_count),
        seed=int(args.seed),
    )

    rows: list[dict[str, float | int]] = []
    # reuse the same draws so MC noise does not obscure the M dependence
    np.savez_compressed(args.out_dir / "channel_draws.npz", **{f"vertex_{v}": d for v, d in enumerate(draws)})
    chain_data = {}
    for shifts in m_values:
        for vertex, channels in enumerate(sector.channels):
            two_js = tuple(int(sector.edge_labels[edge]) for edge in sector.incident_edges[vertex])
            t0 = time.perf_counter()
            values, lambda_v, relres = srq_action_on_ones(
                two_js=two_js,
                sigma=sector.sigmas[vertex],
                channels=channels,
                shifts=int(shifts),
            )
            srq_seconds = time.perf_counter() - t0
            mean, stderr = summarize_vertex_samples(values, draws[vertex])
            chain_data[f"M{shifts}_vertex_{vertex}"] = values[draws[vertex]].mean(axis=1)
            rows.append(
                {
                    "cutoff": int(args.cutoff),
                    "M": int(shifts),
                    "sample_count": int(args.sample_count),
                    "chains": int(args.chains),
                    "total_samples": int(args.sample_count) * int(args.chains),
                    "vertex": int(vertex),
                    "local_dim": int(len(channels)),
                    "lambda_v": float(lambda_v),
                    "mean": mean,
                    "stderr": stderr,
                    "max_relres": float(relres),
                    "srq_seconds": float(srq_seconds),
                }
            )
        print(f"completed M={shifts}")

    csv_path = args.out_dir / f"k5_cutoff{args.cutoff}_banded_srq_mc_vs_M.csv"
    np.savez_compressed(args.out_dir / "chain_means.npz", **chain_data)
    write_csv(csv_path, rows)
    plot_path = plot_vs_m(
        rows,
        cutoff=int(args.cutoff),
        sample_count=int(args.sample_count),
        chains=int(args.chains),
        out_dir=args.out_dir,
        dpi=int(args.dpi),
    )

    largest_local_dim = int(max(sector.channel_dims))
    dense_q_gib = bytes_to_gib(largest_local_dim * largest_local_dim * np.dtype(np.complex128).itemsize)
    total_seconds = time.perf_counter() - script_start
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")
    print(f"cutoff={args.cutoff}, M_values={m_values}, total_seconds={total_seconds:.3f}")
