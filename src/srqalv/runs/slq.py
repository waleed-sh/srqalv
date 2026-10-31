from __future__ import annotations

import csv
import time

from functools import partial
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from srqalv.general_applyq import enumerate_left_comb_channels
from srqalv.plotting import configure_matplotlib
from srqalv.plotting import style_axes
from srqalv.spectral_helpers import build_or_load_transitions
from srqalv.spectral_helpers import dense_oracle_check
from srqalv.spectral_helpers import gaussian_density
from srqalv.spectral_helpers import make_csr_applyq
from srqalv.spectral_helpers import random_orientation_signs
from srqalv.spectral_helpers import sparse_row_sum_from_csr
from srqalv.spectral_helpers import validate_integer_bt
from srqalv.spectral_helpers import weighted_moments
from srqalv.srq import srq_parameters
from srqalv.srq import srq_sqrt_abs_values

Array = np.ndarray


def random_complex_probes(*, probes: int, dim: int, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(probes, dim)) + 1.0j * rng.normal(size=(probes, dim))
    values /= np.linalg.norm(values, axis=-1, keepdims=True)
    return values.astype(np.complex128)


def srq_grid(shifts: int) -> tuple[Array, Array, float]:
    params = srq_parameters(int(shifts))
    ks = np.arange(-params.kminus, params.kplus + 1, dtype=np.float64)
    kh = ks * params.h
    return np.exp(kh), np.exp(kh / 4.0), float(params.h)


@jax.jit
def sparse_apply_q_jit(rows: jax.Array, cols: jax.Array, values: jax.Array, x: jax.Array) -> jax.Array:
    out = jnp.zeros_like(x)
    return out.at[:, rows].add(values[None, :] * x[:, cols])


@jax.jit
def sparse_apply_abar_jit(
    rows: jax.Array,
    cols: jax.Array,
    values: jax.Array,
    lambda_bound: jax.Array,
    x: jax.Array,
) -> jax.Array:

    qx = sparse_apply_q_jit(rows, cols, values, x)
    qqx = sparse_apply_q_jit(rows, cols, values, qx)
    return qqx / (lambda_bound * lambda_bound)


@partial(jax.jit, static_argnames=("cg_maxiter",))
def shifted_cg_all_shifts_jit(
    rows: jax.Array,
    cols: jax.Array,
    values: jax.Array,
    rhs: jax.Array,
    taus: jax.Array,
    lambda_bound: jax.Array,
    cg_tol: jax.Array,
    *,
    cg_maxiter: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:

    shifts = taus.shape[0]
    batch = rhs.shape[0]
    dim = rhs.shape[1]
    b = jnp.broadcast_to(rhs[None, :, :], (shifts, batch, dim))
    x = jnp.zeros_like(b)
    r = b
    p = r
    rs = jnp.real(jnp.sum(jnp.conj(r) * r, axis=-1))
    bnorm2 = jnp.maximum(jnp.real(jnp.sum(jnp.conj(b) * b, axis=-1)), 1.0e-300)
    threshold = (cg_tol * cg_tol) * bnorm2

    def matvec(y: jax.Array) -> jax.Array:
        flat = y.reshape((shifts * batch, dim))
        abar = sparse_apply_abar_jit(rows, cols, values, lambda_bound, flat)
        return abar.reshape((shifts, batch, dim)) + taus[:, None, None] * y

    def cond(carry):
        iteration, _x, _r, _p, rs_i = carry
        return (iteration < int(cg_maxiter)) & jnp.any(rs_i > threshold)

    def body(carry):
        iteration, x_i, r_i, p_i, rs_i = carry
        ap = matvec(p_i)
        denom = jnp.real(jnp.sum(jnp.conj(p_i) * ap, axis=-1))
        active = (rs_i > threshold) & (jnp.abs(denom) > 1.0e-300)
        # freeze converged systems while the remaining recurrances run
        alpha = jnp.where(active, rs_i / denom, 0.0)
        x_next = x_i + alpha[:, :, None] * p_i
        r_candidate = r_i - alpha[:, :, None] * ap
        rs_candidate = jnp.real(jnp.sum(jnp.conj(r_candidate) * r_candidate, axis=-1))
        beta = jnp.where(active & (rs_i > 1.0e-300), rs_candidate / rs_i, 0.0)
        p_candidate = r_candidate + beta[:, :, None] * p_i
        r_next = jnp.where(active[:, :, None], r_candidate, r_i)
        p_next = jnp.where(active[:, :, None], p_candidate, jnp.zeros_like(p_i))
        rs_next = jnp.where(active, rs_candidate, rs_i)
        return iteration + 1, x_next, r_next, p_next, rs_next

    iterations, x, _r, _p, rs = jax.lax.while_loop(cond, body, (0, x, r, p, rs))
    relres = jnp.sqrt(jnp.maximum(rs, 0.0) / bnorm2)
    return x, relres, iterations


@partial(jax.jit, static_argnames=("cg_maxiter",))
def srq_apply_batch_jit(
    rows: jax.Array,
    cols: jax.Array,
    values: jax.Array,
    vectors: jax.Array,
    taus: jax.Array,
    weights: jax.Array,
    h: jax.Array,
    lambda_bound: jax.Array,
    cg_tol: jax.Array,
    *,
    cg_maxiter: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:

    rhs = sparse_apply_abar_jit(rows, cols, values, lambda_bound, vectors)
    solutions, relres, iterations = shifted_cg_all_shifts_jit(
        rows,
        cols,
        values,
        rhs,
        taus,
        lambda_bound,
        cg_tol,
        cg_maxiter=cg_maxiter,
    )
    weighted = jnp.sum(weights[:, None, None] * solutions, axis=0)
    prefactor = jnp.sqrt(lambda_bound) * h / (jnp.sqrt(2.0) * jnp.pi)
    return prefactor * weighted, jnp.max(relres), jnp.mean(relres), iterations


@partial(jax.jit, static_argnames=("steps", "cg_maxiter"))
def lanczos_direct_srq_batch_jit(
    rows: jax.Array,
    cols: jax.Array,
    values: jax.Array,
    init_vectors: jax.Array,
    taus: jax.Array,
    weights: jax.Array,
    h: jax.Array,
    lambda_bound: jax.Array,
    cg_tol: jax.Array,
    *,
    steps: int,
    cg_maxiter: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:

    batch, dim = init_vectors.shape
    q = init_vectors / jnp.linalg.norm(init_vectors, axis=-1, keepdims=True)
    q_prev = jnp.zeros_like(q)
    beta_prev = jnp.zeros((batch,), dtype=jnp.float64)
    basis = jnp.zeros((steps, batch, dim), dtype=init_vectors.dtype)
    alphas = jnp.zeros((steps, batch), dtype=jnp.float64)
    betas = jnp.zeros((steps, batch), dtype=jnp.float64)
    max_relres = jnp.asarray(0.0, dtype=jnp.float64)
    mean_relres_sum = jnp.asarray(0.0, dtype=jnp.float64)
    cg_sweeps = jnp.asarray(0, dtype=jnp.int32)

    def body(carry, idx):
        basis_i, q_prev_i, q_i, beta_prev_i, alphas_i, betas_i, max_r, mean_r_sum, sweeps = carry
        z, relres_max, relres_mean, iters = srq_apply_batch_jit(
            rows,
            cols,
            values,
            q_i,
            taus,
            weights,
            h,
            lambda_bound,
            cg_tol,
            cg_maxiter=cg_maxiter,
        )
        alpha = jnp.real(jnp.sum(jnp.conj(q_i) * z, axis=-1))
        z = z - alpha[:, None] * q_i - beta_prev_i[:, None] * q_prev_i

        coeffs = jnp.sum(jnp.conj(basis_i) * z[None, :, :], axis=-1)
        mask = (jnp.arange(steps) < idx)[:, None, None]
        z = z - jnp.sum(jnp.where(mask, coeffs[:, :, None] * basis_i, 0.0), axis=0)

        beta = jnp.linalg.norm(z, axis=-1)
        safe_beta = jnp.where(beta > 1.0e-14, beta, 1.0)
        q_next = z / safe_beta[:, None]
        q_next = jnp.where((beta > 1.0e-14)[:, None], q_next, q_i)

        basis_i = basis_i.at[idx].set(q_i)
        alphas_i = alphas_i.at[idx].set(alpha)
        betas_i = betas_i.at[idx].set(beta)
        return (
            basis_i,
            q_i,
            q_next,
            beta,
            alphas_i,
            betas_i,
            jnp.maximum(max_r, relres_max),
            mean_r_sum + relres_mean,
            sweeps + iters,
        ), None

    (basis, q_prev, q, beta_prev, alphas, betas, max_relres, mean_relres_sum, cg_sweeps), _ = jax.lax.scan(
        body,
        (basis, q_prev, q, beta_prev, alphas, betas, max_relres, mean_relres_sum, cg_sweeps),
        jnp.arange(steps),
    )
    del basis, q_prev, q, beta_prev
    return (
        jnp.swapaxes(alphas, 0, 1),
        jnp.swapaxes(betas[:-1], 0, 1),
        max_relres,
        mean_relres_sum / float(steps),
        cg_sweeps,
    )


@jax.jit
def tridiagonal_eigh_jit(alphas: jax.Array, betas: jax.Array) -> tuple[jax.Array, jax.Array]:
    batch, steps = alphas.shape
    diag_idx = jnp.arange(steps)
    tri = jnp.zeros((batch, steps, steps), dtype=jnp.float64)
    tri = tri.at[:, diag_idx, diag_idx].set(alphas)
    off_idx = jnp.arange(steps - 1)
    tri = tri.at[:, off_idx, off_idx + 1].set(betas)
    tri = tri.at[:, off_idx + 1, off_idx].set(betas)
    return jnp.linalg.eigh(tri)


def run_lanczos_for_m(
    *,
    rows_j: jax.Array,
    cols_j: jax.Array,
    values_j: jax.Array,
    probes: Array,
    shifts: int,
    steps: int,
    probe_batch_size: int,
    lambda_bound: float,
    cg_tol: float,
    cg_maxiter: int,
) -> tuple[Array, Array, float, float, float, float, int]:

    taus, weights, h = srq_grid(shifts)
    taus_j = jnp.asarray(taus, dtype=jnp.float64)
    weights_j = jnp.asarray(weights, dtype=jnp.float64)
    h_j = jnp.asarray(h, dtype=jnp.float64)
    lambda_j = jnp.asarray(lambda_bound, dtype=jnp.float64)
    tol_j = jnp.asarray(cg_tol, dtype=jnp.float64)

    all_alphas: list[Array] = []
    all_betas: list[Array] = []
    max_relres = 0.0
    mean_relres_weighted = 0.0
    total_probe_count = 0
    total_cg_sweeps = 0

    t0 = time.perf_counter()
    for start in range(0, probes.shape[0], probe_batch_size):
        chunk = probes[start : start + probe_batch_size]
        chunk_j = jnp.asarray(chunk, dtype=jnp.complex128)
        alphas_j, betas_j, relmax_j, relmean_j, sweeps_j = lanczos_direct_srq_batch_jit(
            rows_j,
            cols_j,
            values_j,
            chunk_j,
            taus_j,
            weights_j,
            h_j,
            lambda_j,
            tol_j,
            steps=int(steps),
            cg_maxiter=int(cg_maxiter),
        )
        jax.block_until_ready(alphas_j)
        all_alphas.append(np.asarray(alphas_j, dtype=np.float64))
        all_betas.append(np.asarray(betas_j, dtype=np.float64))
        chunk_size = chunk.shape[0]
        max_relres = max(max_relres, float(relmax_j))
        mean_relres_weighted += float(relmean_j) * chunk_size
        total_probe_count += chunk_size
        total_cg_sweeps += int(sweeps_j)
    lanczos_seconds = time.perf_counter() - t0

    alphas = np.concatenate(all_alphas, axis=0)
    betas = np.concatenate(all_betas, axis=0)
    t0 = time.perf_counter()
    theta_j, evecs_j = tridiagonal_eigh_jit(jnp.asarray(alphas), jnp.asarray(betas))
    jax.block_until_ready(theta_j)
    eig_seconds = time.perf_counter() - t0
    weights = np.asarray(jnp.abs(evecs_j[:, 0, :]) ** 2, dtype=np.float64)
    nodes = np.asarray(theta_j, dtype=np.float64)
    mean_relres = mean_relres_weighted / max(total_probe_count, 1)
    return nodes, weights, lanczos_seconds, eig_seconds, max_relres, mean_relres, total_cg_sweeps


def curve_label(shifts: int, probes: int, *, multiple_m: bool, multiple_probe_counts: bool) -> str:
    if multiple_m and multiple_probe_counts:
        return fr"SRQ-SLQ, $M={shifts}$, $R={probes}$"
    if multiple_m:
        return fr"SRQ-SLQ, $M={shifts}$"
    if multiple_probe_counts:
        return fr"SRQ-SLQ, $R={probes}$"
    return "SRQ-SLQ"


def run(args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    probe_values = tuple(args.probe_values)
    max_probes = max(probe_values)
    probe_batch_size = max(1, min(int(args.probe_batch_size), max_probes))

    validation_worst = None
    if args.validate:
        validation_worst = validate_integer_bt(args.tol)

    signs, tangents = random_orientation_signs(5, seed=args.seed)
    del tangents
    channels = enumerate_left_comb_channels(args.two_js)
    transitions, build_seconds, loaded_cache, transition_cache = build_or_load_transitions(
        out_dir=args.out_dir,
        two_js=args.two_js,
        channels=channels,
        signs=signs,
        seed=args.seed,
        tol=args.tol,
        rebuild=args.rebuild_cache,
    )
    _apply_q_numpy, q_csr = make_csr_applyq(transitions)
    diff = q_csr - q_csr.getH()
    hermitian_error = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
    lambda_bound = sparse_row_sum_from_csr(q_csr) * (1.0 + 1.0e-12)

    exact_peak_values_by_m: dict[int, Array] = {}
    exact_peak_note = "disabled"
    if args.exact_peaks or args.exact_histogram:
        if transitions.dim <= int(args.exact_peaks_max_dim):
            q_dense = q_csr.toarray()
            q_dense = 0.5 * (q_dense + q_dense.conj().T)
            q_eigs = np.linalg.eigvalsh(q_dense)
            exact_peak_values_by_m = {
                int(shifts): srq_sqrt_abs_values(q_eigs, lambda_bound=lambda_bound, shifts=int(shifts))
                for shifts in args.orders
            }
            exact_peak_note = "finite-SRQ dense spectrum"
        else:
            exact_peak_note = f"skipped: dim {transitions.dim} > {args.exact_peaks_max_dim}"

    exact_v_values: Array | None = None
    dense_check_error = float("nan")
    dense_check_seconds = float("nan")
    dense_check_magnetic_dim = int(np.prod(np.asarray(args.two_js, dtype=np.int64) + 1))
    if args.dense_check:
        exact_v_values, dense_check_error, dense_check_seconds, dense_check_magnetic_dim = dense_oracle_check(
            two_js=args.two_js,
            channels=channels,
            signs=signs,
            q_csr=q_csr,
            max_dim=args.dense_check_max_dim,
            max_magnetic_dim=args.dense_check_max_magnetic_dim,
            tol=args.dense_check_tol,
        )

    probes = random_complex_probes(
        probes=max_probes,
        dim=transitions.dim,
        seed=args.seed + int(args.probe_seed_offset),
    )
    np.savez_compressed(args.out_dir / "probes.npz", probes=probes)
    rows_j = jnp.asarray(transitions.rows, dtype=jnp.int32)
    cols_j = jnp.asarray(transitions.cols, dtype=jnp.int32)
    values_j = jnp.asarray(transitions.values, dtype=jnp.complex128)

    curve_keys = tuple((int(shifts), int(probe_count)) for shifts in args.orders for probe_count in probe_values)
    all_nodes: dict[tuple[int, int], Array] = {}
    all_weights: dict[tuple[int, int], Array] = {}
    all_density: dict[tuple[int, int], Array] = {}
    seconds_by_m: dict[int, tuple[float, float]] = {}
    relres_by_m: dict[int, tuple[float, float]] = {}
    cg_sweeps_by_m: dict[int, int] = {}

    for shifts in args.orders:
        nodes_m, weights_m, lanczos_seconds, eig_seconds, max_relres, mean_relres, cg_sweeps = run_lanczos_for_m(
            rows_j=rows_j,
            cols_j=cols_j,
            values_j=values_j,
            probes=probes,
            shifts=int(shifts),
            steps=int(args.lanczos_steps),
            probe_batch_size=probe_batch_size,
            lambda_bound=lambda_bound,
            cg_tol=float(args.cg_tol),
            cg_maxiter=int(args.cg_maxiter),
        )
        seconds_by_m[int(shifts)] = (lanczos_seconds, eig_seconds)
        # retain the unsmoothed weighted Ritz measures for seperate checks
        np.savez_compressed(
            args.out_dir / f"slq_measure_M{shifts}.npz",
            nodes=nodes_m, weights=weights_m,
            max_relres=max_relres, mean_relres=mean_relres, cg_sweeps=cg_sweeps,
        )
        relres_by_m[int(shifts)] = (max_relres, mean_relres)
        cg_sweeps_by_m[int(shifts)] = cg_sweeps
        for probe_count in probe_values:
            key = (int(shifts), int(probe_count))
            all_nodes[key] = nodes_m[:probe_count].reshape(-1)
            all_weights[key] = (weights_m[:probe_count] / float(probe_count)).reshape(-1)

    maxima = [float(np.max(nodes)) for nodes in all_nodes.values()]
    if exact_v_values is not None and exact_v_values.size:
        maxima.append(float(np.max(exact_v_values)))
    for values in exact_peak_values_by_m.values():
        if values.size:
            maxima.append(float(np.max(values)))
    x_max = max(maxima) * 1.04 if maxima else 1.0
    grid = np.linspace(0.0, x_max, int(args.grid_points))
    bandwidth = max((x_max / 95.0) * float(args.density_bandwidth_scale), 1.0e-8)
    # these are the finite-M SRQ eigenvalues, not sqrt(abs(Q))
    for shifts, values in exact_peak_values_by_m.items():
        np.savetxt(
            args.out_dir / f"exact_finite_srq_spectrum_M{shifts}.csv",
            np.column_stack((np.arange(values.size), values)),
            delimiter=",", header="index,lambda_V", comments="", fmt=["%d", "%.16e"],
        )
        hist, edges = np.histogram(values, bins=args.exact_histogram_bins, range=(0.0, x_max), density=True)
        np.savetxt(
            args.out_dir / f"exact_finite_srq_histogram_M{shifts}_bins{args.exact_histogram_bins}.csv",
            np.column_stack((edges[:-1], edges[1:], hist)), delimiter=",",
            header="bin_left,bin_right,density", comments="", fmt="%.16e",
        )
    np.savez_compressed(args.out_dir / "density_grid.npz", grid=grid, bandwidth=bandwidth)
    exact_density = None
    if exact_v_values is not None:
        exact_density = gaussian_density(exact_v_values, np.ones_like(exact_v_values), grid, bandwidth)
    for key, nodes in all_nodes.items():
        all_density[key] = gaussian_density(nodes, all_weights[key], grid, bandwidth)

    spin_label = "_".join(str(value) for value in args.two_js)
    csv_path = args.out_dir / f"five_valent_twojs_{spin_label}_bt_direct_srq_slq_density.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["lambda_V"]
        if exact_density is not None:
            header.append("exact_dense_density")
        header.extend(f"M{shifts}_R{probe_count}_density" for shifts, probe_count in curve_keys)
        writer.writerow(header)
        for idx, value in enumerate(grid):
            row = [f"{value:.16e}"]
            if exact_density is not None:
                row.append(f"{exact_density[idx]:.16e}")
            row.extend(f"{all_density[key][idx]:.16e}" for key in curve_keys)
            writer.writerow(row)

    colors = ["#a53860", "#450920", "#da627d", "#ffa5ab", "#f9dbbd"]
    linestyles = ["-", "-", "--", "-.", ":"]
    fig, ax = plt.subplots(figsize=(8.4, 5.35))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.16, top=0.97)
    multiple_m = len(args.orders) > 1
    multiple_probe_counts = len(probe_values) > 1
    if exact_density is not None:
        ax.fill_between(
            grid,
            exact_density,
            color="#f9dbbd",
            alpha=0.62,
            linewidth=0.0,
            label="Exact",
        )
    if args.exact_histogram and exact_peak_values_by_m:
        hist_m = int(args.orders[0])
        hist_values = exact_peak_values_by_m[hist_m]
        hist_density, hist_edges = np.histogram(
            hist_values,
            bins=max(1, int(args.exact_histogram_bins)),
            range=(0.0, x_max),
            density=True,
        )
        ax.stairs(
            hist_density,
            hist_edges,
            color="#8ecae6",
            linewidth=1.9,
            alpha=0.92,
            label=(fr"Exact spectrum, $M={hist_m}$" if multiple_m else "Exact spectrum"),
        )
    for idx, (shifts, probe_count) in enumerate(curve_keys):
        ax.plot(
            grid,
            all_density[(shifts, probe_count)],
            color=colors[idx % len(colors)],
            linestyle=linestyles[(idx // len(colors)) % len(linestyles)],
            linewidth=2.35,
            label=curve_label(
                shifts,
                probe_count,
                multiple_m=multiple_m,
                multiple_probe_counts=multiple_probe_counts,
            ),
        )
    if args.exact_peaks and exact_peak_values_by_m:
        peak_m = int(args.orders[0])
        peak_values = exact_peak_values_by_m[peak_m]
        ymax = max(
            [float(np.max(values)) for values in all_density.values()]
            + ([float(np.max(exact_density))] if exact_density is not None else [0.0])
        )
        peak_height = float(args.exact_peak_height) * ymax if ymax > 0.0 else 1.0
        ax.vlines(
            peak_values,
            ymin=0.0,
            ymax=peak_height,
            color="#219ebc",
            alpha=float(args.exact_peak_alpha),
            linewidth=float(args.exact_peak_linewidth),
            label=(fr"Exact eigs, $M={peak_m}$" if multiple_m else "Exact eigs"),
        )
        ax.set_ylim(bottom=0.0, top=max(ymax * 1.08, peak_height * 1.25))
    ax.set_xlabel(r"$\lambda_V$")
    ax.set_ylabel("normalized spectral density")
    style_axes(ax)
    ax.legend(frameon=False)
    figure_path = args.out_dir / f"five_valent_twojs_{spin_label}_bt_direct_srq_slq_density.png"
    fig.savefig(figure_path, dpi=args.dpi)
    plt.close(fig)

    print(f"Wrote {figure_path}")
    print(f"Wrote {csv_path}")
    print(f"dim={transitions.dim}, nnz={q_csr.nnz}, Lambda={lambda_bound:.6g}")
    for shifts in args.orders:
        lanczos_seconds, _eig_seconds = seconds_by_m[int(shifts)]
        max_relres, mean_relres = relres_by_m[int(shifts)]
        print(
            f"M={int(shifts)}: direct SRQ-SLQ seconds={lanczos_seconds:.3f}, "
            f"max_relres={max_relres:.3e}, mean_relres={mean_relres:.3e}"
        )
    for shifts, probe_count in curve_keys:
        key = (shifts, probe_count)
        mean, std = weighted_moments(all_nodes[key], all_weights[key])
        print(f"M={shifts}, R={probe_count}: mean={mean:.6g}, std={std:.6g}")
