from __future__ import annotations
import argparse
import time
from pathlib import Path
from itertools import combinations
import numpy as np
import scipy.sparse as sp
from srqalv.bt_applyq import SparseQTransitions
from srqalv.bt_eq47 import bt_q_commutator_matrix, bt_total_q_transitions
from srqalv.general_applyq import GeneralApplyQ, enumerate_left_comb_channels
Array = np.ndarray


def parse_two_js(value: str) -> tuple[int, ...]:
    vals = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(vals) != 5 or any(val < 0 for val in vals):
        raise argparse.ArgumentTypeError("Provide exactly five non-negative doubled-spin labels.")
    if any(val % 2 for val in vals):
        raise argparse.ArgumentTypeError("This BT Eq. 47 landmark script is restricted to validated integer spins.")
    return vals


def parse_int_list(value: str) -> tuple[int, ...]:
    vals = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not vals or any(val <= 0 for val in vals):
        raise argparse.ArgumentTypeError("Provide one or more positive integers.")
    return vals


def random_orientation_signs(valence: int, *, seed: int) -> tuple[dict[tuple[int, int, int], int], Array]:
    rng = np.random.default_rng(seed)
    tangents = rng.normal(size=(valence, 3))
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    signs: dict[tuple[int, int, int], int] = {}
    for triple in combinations(range(valence), 3):
        det = float(np.linalg.det(tangents[list(triple), :]))
        signs[triple] = 1 if det > 0.0 else -1
    return signs, tangents


def validate_integer_bt(tol: float) -> float:
    worst = 0.0
    sectors = (
        (2, 2, 2, 2),
        (2, 2, 2, 2, 2),
        (2, 2, 2, 2, 2, 2),
    )
    for two_js in sectors:
        channels = enumerate_left_comb_channels(two_js)
        for triple in combinations(range(len(two_js)), 3):
            oracle = GeneralApplyQ.create(two_js, channel_tuples=channels, triple_signs={triple: 1})
            dense_q = oracle.dense_q()
            bt_q = 1.0j * bt_q_commutator_matrix(two_js, channels, triple, tol=tol)
            err = float(np.max(np.abs(bt_q - dense_q))) if dense_q.size else 0.0
            worst = max(worst, err)
            if err > tol:
                label = tuple(index + 1 for index in triple)
                raise RuntimeError(f"BT dense validation failed for two_js={two_js}, q{label}: {err:.3e}")
    return worst


def cache_path(out_dir: Path, two_js: tuple[int, ...], seed: int, tol: float) -> Path:
    spin_label = "_".join(str(value) for value in two_js)
    return out_dir / f"bt_total_q_twojs_{spin_label}_seed_{seed}_tol_{tol:.0e}.npz"


def save_transitions(path: Path, transitions: SparseQTransitions) -> None:
    np.savez_compressed(
        path,
        rows=transitions.rows,
        cols=transitions.cols,
        values=transitions.values,
        dim=np.asarray([transitions.dim], dtype=np.int64),
    )


def load_transitions(path: Path) -> SparseQTransitions:
    data = np.load(path)
    return SparseQTransitions(
        rows=np.asarray(data["rows"], dtype=np.int32),
        cols=np.asarray(data["cols"], dtype=np.int32),
        values=np.asarray(data["values"], dtype=np.complex128),
        dim=int(np.asarray(data["dim"])[0]),
        label="cached BT Eq47 total Q",
    )


def build_or_load_transitions(
    *,
    out_dir: Path,
    two_js: tuple[int, ...],
    channels: tuple[tuple[int, ...], ...],
    signs: dict[tuple[int, int, int], int],
    seed: int,
    tol: float,
    rebuild: bool,
) -> tuple[SparseQTransitions, float, bool, Path]:

    path = cache_path(out_dir, two_js, seed, tol)
    if path.exists() and not rebuild:
        t0 = time.perf_counter()
        transitions = load_transitions(path)
        return transitions, time.perf_counter() - t0, True, path

    t0 = time.perf_counter()
    transitions = bt_total_q_transitions(two_js, channels, triple_signs=signs, tol=tol, coalesce=True)
    seconds = time.perf_counter() - t0
    save_transitions(path, transitions)
    return transitions, seconds, False, path


def make_csr_applyq(transitions: SparseQTransitions):
    q_csr = sp.csr_matrix(
        (transitions.values, (transitions.rows, transitions.cols)),
        shape=(transitions.dim, transitions.dim),
        dtype=np.complex128,
    )
    q_csr.sum_duplicates()
    q_csr.eliminate_zeros()

    def apply_q(vectors: Array) -> Array:
        arr = np.asarray(vectors, dtype=np.complex128)
        if arr.ndim == 1:
            arr = arr[None, :]
        return np.asarray(q_csr.dot(arr.T).T, dtype=np.complex128)

    return apply_q, q_csr


def sparse_row_sum_from_csr(q_csr: sp.csr_matrix) -> float:
    return float(np.max(np.asarray(abs(q_csr).sum(axis=1)).ravel()))


def gaussian_density(values: Array, weights: Array, grid: Array, bandwidth: float) -> Array:
    vals = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if vals.size == 0 or np.sum(w) <= 0.0:
        return np.zeros_like(grid)
    w = w / np.sum(w)
    scaled = (grid[:, None] - vals[None, :]) / float(bandwidth)
    density = np.exp(-0.5 * scaled * scaled) @ w
    density /= float(bandwidth) * np.sqrt(2.0 * np.pi)
    return density


def weighted_moments(values: Array, weights: Array) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.sum(w)
    mean = float(np.sum(w * vals))
    var = float(np.sum(w * (vals - mean) ** 2))
    return mean, float(np.sqrt(max(var, 0.0)))


def dense_oracle_check(
    *,
    two_js: tuple[int, ...],
    channels: tuple[tuple[int, ...], ...],
    signs: dict[tuple[int, int, int], int],
    q_csr: sp.csr_matrix,
    max_dim: int,
    max_magnetic_dim: int,
    tol: float,
) -> tuple[Array, float, float, int]:

    dim = int(q_csr.shape[0])
    magnetic_dim = int(np.prod(np.asarray(two_js, dtype=np.int64) + 1))
    if dim > int(max_dim) or magnetic_dim > int(max_magnetic_dim):
        raise SystemExit(
            "--dense-check is intended for small sectors. "
            f"Requested dim={dim}, magnetic_dim={magnetic_dim}; limits are "
            f"dim<={max_dim}, magnetic_dim<={max_magnetic_dim}. "
            "Increase the limits explicitly if this is intentional."
        )

    t0 = time.perf_counter()
    q_dense = np.asarray(q_csr.toarray(), dtype=np.complex128)
    oracle = GeneralApplyQ.create(two_js, channel_tuples=channels, triple_signs=signs)
    q_oracle = oracle.dense_q()
    max_abs_error = float(np.max(np.abs(q_dense - q_oracle))) if q_dense.size else 0.0
    if max_abs_error > float(tol):
        raise RuntimeError(f"Dense oracle check failed: max_abs_error={max_abs_error:.3e} > {tol:.3e}.")

    evals = np.linalg.eigvalsh(0.5 * (q_dense + q_dense.conj().T)).real
    exact_v = np.sqrt(np.abs(evals))
    return exact_v, max_abs_error, time.perf_counter() - t0, magnetic_dim

