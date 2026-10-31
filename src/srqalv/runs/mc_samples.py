from __future__ import annotations

import csv
import math
import time

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_banded

from srqalv.basis import K5Graph
from srqalv.basis import SU2Hilbert
from srqalv.basis import left_comb_valid_py
from srqalv.basis import orientation_sign
from srqalv.plotting import configure_matplotlib
from srqalv.plotting import style_axes
from srqalv.srq import srq_dense_matrix
from srqalv.srq import srq_parameters
from srqalv.bounds import reduced_sigma

Array = np.ndarray


@dataclass(frozen=True)
class K5Sector:

    cutoff: int
    edge_labels: tuple[int, ...]
    incident_edges: tuple[tuple[int, ...], ...]
    sigmas: tuple[int, ...]
    signs: tuple[dict[tuple[int, int, int], int], ...]
    channels: tuple[tuple[int, ...], ...]

    @property
    def channel_dims(self) -> tuple[int, ...]:
        return tuple(len(item) for item in self.channels)

    @property
    def graph_dimension(self) -> int:
        out = 1
        for dim in self.channel_dims:
            out *= int(dim)
        return int(out)


def bytes_to_gib(value: int | float) -> float:
    return float(value) / float(2**30)


def valid_channels(two_js: tuple[int, int, int, int], cutoff: int) -> tuple[int, ...]:
    j1, j2, j3, j4 = (int(x) for x in two_js)
    lo = max(abs(j1 - j2), abs(j3 - j4))
    hi = min(j1 + j2, j3 + j4, int(cutoff))
    if hi < lo:
        return ()
    parity = (j1 + j2) & 1
    return tuple(
        channel
        for channel in range(lo, hi + 1)
        if (channel & 1) == parity and left_comb_valid_py(two_js, (channel,))
    )


def build_sector(*, cutoff: int, edge_labels: tuple[int, ...], embedding_seed: int) -> K5Sector:
    graph = K5Graph(random_embedding=True, random_embedding_seed=int(embedding_seed))
    space = SU2Hilbert(graph, two_j_max=int(cutoff), exact=False, random_init="loop_warmup")
    if len(edge_labels) != space.n_edges:
        raise ValueError(f"Expected {space.n_edges} fixed edge labels, got {len(edge_labels)}.")
    if any(label < 0 or label > int(cutoff) for label in edge_labels):
        raise ValueError("Every fixed edge label must lie between zero and cutoff.")

    incident_edges = tuple(tuple(int(x) for x in row) for row in space.metadata.incident_edges_by_vertex)
    signs = []
    sigmas = []
    channels = []
    for vertex in range(space.n_vertices):
        vertex_id = space.layout.vertex_ids[vertex]
        tokens = tuple(space.metadata.edge_tokens[edge] for edge in incident_edges[vertex])
        local_signs = {
            triple: int(orientation_sign(graph, vertex_id, tuple(tokens[i] for i in triple)))
            for triple in combinations(range(4), 3)
        }
        two_js = tuple(int(edge_labels[edge]) for edge in incident_edges[vertex])
        local_channels = valid_channels(two_js, int(cutoff))
        if not local_channels:
            raise ValueError(f"Vertex {vertex} has no admissible channels for {two_js}.")
        signs.append(local_signs)
        sigmas.append(reduced_sigma(local_signs))
        channels.append(local_channels)

    return K5Sector(
        cutoff=int(cutoff),
        edge_labels=tuple(int(x) for x in edge_labels),
        incident_edges=incident_edges,
        sigmas=tuple(sigmas),
        signs=tuple(signs),
        channels=tuple(channels),
    )


def srq_grid(shifts: int) -> tuple[Array, Array, float]:
    params = srq_parameters(int(shifts))
    ks = np.arange(-params.kminus, params.kplus + 1, dtype=np.float64)
    kh = ks * params.h
    return np.exp(kh), np.exp(kh / 4.0), float(params.h)


def analytic_lambda(two_js: tuple[int, int, int, int], sigma: int) -> float:
    if int(sigma) == 0:
        return 0.0
    js = np.asarray(two_js[:3], dtype=np.float64) / 2.0
    return float(6.0 * abs(int(sigma)) * np.prod(np.sqrt(js * (js + 1.0))))


def four_valent_offdiag_vector(two_js: tuple[int, int, int, int], channels: tuple[int, ...]) -> Array:
    if len(channels) <= 1:
        return np.zeros((0,), dtype=np.float64)
    upper = np.asarray(channels[1:], dtype=np.float64)
    j1, j2, j3, j4 = (float(x) for x in two_js)
    x = upper
    left = (x + j1 + j2 + 2.0) * (-j1 + j2 + x) * (j1 - j2 + x) * (j1 + j2 - x + 2.0)
    right = (x + j3 + j4 + 2.0) * (-j3 + j4 + x) * (j3 - j4 + x) * (j3 + j4 - x + 2.0)
    product = left * right
    denom = 16.0 * np.sqrt(np.maximum((x - 1.0) * (x + 1.0), 1.0))
    out = np.sqrt(np.maximum(product, 0.0)) / denom
    out[product <= 1.0e-28] = 0.0
    return out.astype(np.float64)


def abar_bands(
    *,
    two_js: tuple[int, int, int, int],
    sigma: int,
    channels: tuple[int, ...],
    lambda_bound: float,
) -> tuple[Array, Array]:

    n = len(channels)
    if lambda_bound <= 0.0 or n == 0:
        return np.zeros((n,), dtype=np.float64), np.zeros((max(n - 2, 0),), dtype=np.float64)

    mags = four_valent_offdiag_vector(two_js, channels)
    scale = float(int(sigma) * int(sigma)) / float(lambda_bound * lambda_bound)
    diag = np.zeros((n,), dtype=np.float64)
    if mags.size:
        sq = mags * mags
        diag[:-1] += sq
        diag[1:] += sq
    off2 = -mags[:-1] * mags[1:] if mags.size >= 2 else np.zeros((0,), dtype=np.float64)
    return scale * diag, scale * off2


def apply_abar(diag: Array, off2: Array, x: Array) -> Array:
    out = np.asarray(diag) * np.asarray(x)
    if off2.size:
        out = out.copy()
        out[:-2] += off2 * x[2:]
        out[2:] += off2 * x[:-2]
    return out


def solve_tridiagonal_banded(main: Array, off: Array, rhs: Array) -> Array:
    n = int(main.size)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    ab = np.zeros((3, n), dtype=np.float64)
    ab[1, :] = main
    if n > 1:
        ab[0, 1:] = off
        ab[2, :-1] = off
    return solve_banded((1, 1), ab, np.asarray(rhs, dtype=np.float64), check_finite=False)


def solve_shifted_abar(diag: Array, off2: Array, rhs: Array, tau: float) -> Array:
    out = np.empty_like(rhs, dtype=np.float64)
    # q changes channel parity, so q squared gives two tridiagonal chains
    for parity in (0, 1):
        idx = np.arange(parity, rhs.size, 2)
        if idx.size == 0:
            continue
        main = diag[idx] + float(tau)
        off = off2[parity::2][: max(idx.size - 1, 0)]
        out[idx] = solve_tridiagonal_banded(main, off, rhs[idx])
    return out


def srq_action_on_ones(
    *,
    two_js: tuple[int, int, int, int],
    sigma: int,
    channels: tuple[int, ...],
    shifts: int,
) -> tuple[Array, float, float]:

    lambda_bound = analytic_lambda(two_js, sigma)
    if lambda_bound <= 0.0:
        return np.zeros((len(channels),), dtype=np.float64), 0.0, 0.0

    diag, off2 = abar_bands(two_js=two_js, sigma=sigma, channels=channels, lambda_bound=lambda_bound)
    ones = np.ones((len(channels),), dtype=np.float64)
    rhs = apply_abar(diag, off2, ones)
    rhs_norm = max(float(np.linalg.norm(rhs)), 1.0e-300)
    taus, weights, h = srq_grid(int(shifts))
    acc = np.zeros_like(rhs)
    max_relres = 0.0
    for tau, weight in zip(taus, weights, strict=True):
        y = solve_shifted_abar(diag, off2, rhs, float(tau))
        residual = apply_abar(diag, off2, y) + float(tau) * y - rhs
        max_relres = max(max_relres, float(np.linalg.norm(residual) / rhs_norm))
        acc += float(weight) * y
    prefactor = math.sqrt(lambda_bound) * float(h) / (math.sqrt(2.0) * math.pi)
    return prefactor * acc, float(lambda_bound), max_relres


def dense_validation_error(sector: K5Sector, local_values: tuple[Array, ...], shifts: int, max_dim: int) -> float:
    if max(sector.channel_dims) > int(max_dim):
        return float("nan")
    worst = 0.0
    for vertex, channels in enumerate(sector.channels):
        two_js = tuple(int(sector.edge_labels[edge]) for edge in sector.incident_edges[vertex])
        mags = four_valent_offdiag_vector(two_js, channels)
        q = np.zeros((len(channels), len(channels)), dtype=np.complex128)
        for idx, magnitude in enumerate(mags, start=1):
            coeff = 1.0j * float(sector.sigmas[vertex]) * float(magnitude)
            q[idx - 1, idx] = -coeff
            q[idx, idx - 1] = coeff
        lambda_bound = analytic_lambda(two_js, sector.sigmas[vertex])
        dense = srq_dense_matrix(q, lambda_bound=lambda_bound, shifts=int(shifts))
        ref = np.real(dense @ np.ones((len(channels),), dtype=np.complex128))
        worst = max(worst, float(np.max(np.abs(ref - local_values[vertex]))))
    return worst


def summarize_samples(
    *,
    sector: K5Sector,
    local_values: tuple[Array, ...],
    sample_counts: tuple[int, ...],
    chains: int,
    seed: int,
    out_dir: Path | None = None,
) -> list[dict[str, float | int]]:

    rng = np.random.default_rng(int(seed))
    max_samples = int(max(sample_counts))
    rows: list[dict[str, float | int]] = []
    raw_draws = {}
    raw_means = {}
    for vertex, values in enumerate(local_values):
        draws = rng.integers(0, len(values), size=(int(chains), max_samples), endpoint=False)
        raw_draws[f"vertex_{vertex}"] = draws
        sampled = values[draws]
        for sample_count in sample_counts:
            prefix = sampled[:, : int(sample_count)]
            chain_means = np.mean(prefix, axis=1)
            raw_means[f"N{sample_count}_vertex_{vertex}"] = chain_means
            mean = float(np.mean(chain_means))
            stderr = (
                float(np.std(chain_means, ddof=1) / math.sqrt(len(chain_means)))
                if len(chain_means) > 1
                else float("nan")
            )
            rows.append(
                {
                    "sample_count": int(sample_count),
                    "total_samples": int(sample_count) * int(chains),
                    "vertex": int(vertex),
                    "mean": mean,
                    "stderr": stderr,
                }
            )
    if out_dir is not None:
        np.savez_compressed(out_dir / "channel_draws.npz", **raw_draws)
        np.savez_compressed(out_dir / "chain_means.npz", **raw_means)
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int]], *, cutoff: int, shifts: int) -> None:
    with path.open("w", newline="") as handle:
        fieldnames = ["cutoff", "M", "sample_count", "total_samples", "vertex", "mean", "stderr"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"cutoff": int(cutoff), "M": int(shifts), **row})


def plot_all_vertices_vs_samples(
    *,
    rows: list[dict[str, float | int]],
    sample_counts: tuple[int, ...],
    chains: int,
    cutoff: int,
    shifts: int,
    out_dir: Path,
    dpi: int,
) -> Path:

    colors = ["#f9dbbd", "#ffa5ab", "#da627d", "#a53860", "#450920"]
    fig, ax = plt.subplots(figsize=(8.4, 5.35))
    fig.subplots_adjust(left=0.13, right=0.82, bottom=0.16, top=0.97)
    xs = np.asarray([int(chains) * int(count) for count in sample_counts], dtype=np.float64)
    for vertex in range(5):
        vals = [row for row in rows if int(row["vertex"]) == vertex]
        vals = sorted(vals, key=lambda row: int(row["sample_count"]))
        means = np.asarray([row["mean"] for row in vals], dtype=np.float64)
        errors = np.asarray([row["stderr"] for row in vals], dtype=np.float64)
        color = colors[vertex]
        ax.plot(xs, means, color=color, linewidth=2.35, marker="o", label=fr"$v_{vertex}$")
        ax.fill_between(xs, means - errors, means + errors, color=color, alpha=0.20, linewidth=0.0)
    ax.set_xscale("log")
    ax.set_xlim(xs[0] / 1.10, xs[-1] * 1.08)
    ax.set_xticks(xs)
    ax.set_xticklabels([rf"${value/1000:.1f}\times10^3$" for value in xs])
    ax.set_xlabel(r"total Monte Carlo samples $N$")
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
    path = out_dir / f"k5_cutoff{cutoff}_M{shifts}_banded_srq_mc_all_vertices_vs_samples.png"
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    return path


def run(args) -> None:
    script_start = time.perf_counter()
    if args.cutoff <= 0:
        raise ValueError("experiment.cutoff must be positive")
    if args.order <= 0:
        raise ValueError("experiment.order must be positive")
    if args.chains <= 0:
        raise ValueError("experiment.chains must be positive")
    if not 0 <= args.probe_vertex < 5:
        raise ValueError("experiment.probe_vertex must be between 0 and 4")

    configure_matplotlib()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample_counts = tuple(sorted(set(int(x) for x in args.sample_counts)))

    t0 = time.perf_counter()
    sector = build_sector(
        cutoff=int(args.cutoff),
        edge_labels=tuple(int(x) for x in args.fixed_edge_labels),
        embedding_seed=int(args.embedding_seed),
    )
    build_seconds = time.perf_counter() - t0

    local_values: list[Array] = []
    lambda_bounds: list[float] = []
    relres_by_vertex: list[float] = []
    seconds_by_vertex: list[float] = []
    for vertex, channels in enumerate(sector.channels):
        two_js = tuple(int(sector.edge_labels[edge]) for edge in sector.incident_edges[vertex])
        t0 = time.perf_counter()
        values, lambda_bound, relres = srq_action_on_ones(
            two_js=two_js,
            sigma=sector.sigmas[vertex],
            channels=channels,
            shifts=int(args.order),
        )
        seconds_by_vertex.append(time.perf_counter() - t0)
        local_values.append(values)
        lambda_bounds.append(lambda_bound)
        relres_by_vertex.append(relres)

    local_values_tuple = tuple(local_values)
    dense_error = dense_validation_error(
        sector,
        local_values_tuple,
        shifts=int(args.order),
        max_dim=int(args.dense_check_max_dim),
    )

    t0 = time.perf_counter()
    rows = summarize_samples(
        sector=sector,
        local_values=local_values_tuple,
        sample_counts=sample_counts,
        chains=int(args.chains),
        seed=int(args.seed),
        out_dir=args.out_dir,
    )
    sampling_seconds = time.perf_counter() - t0

    csv_path = args.out_dir / f"k5_cutoff{args.cutoff}_M{args.order}_banded_srq_mc_summary.csv"
    write_csv(csv_path, rows, cutoff=int(args.cutoff), shifts=int(args.order))
    plot_path = plot_all_vertices_vs_samples(
        rows=rows,
        sample_counts=sample_counts,
        chains=int(args.chains),
        cutoff=int(args.cutoff),
        shifts=int(args.order),
        out_dir=args.out_dir,
        dpi=int(args.dpi),
    )
    total_seconds = time.perf_counter() - script_start

    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")
    print(f"cutoff={args.cutoff}, M={args.order}, graph_dim={sector.graph_dimension}")
    print(f"local_dims={sector.channel_dims}")
    print(f"max_relres={max(relres_by_vertex):.3e}")
    for vertex, seconds in enumerate(seconds_by_vertex):
        print(f"vertex {vertex}: dim={sector.channel_dims[vertex]}, seconds={seconds:.3f}")
