from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class SparseQTransitions:

    rows: Array
    cols: Array
    values: Array
    dim: int
    label: str = "Q"

    def dense(self) -> Array:
        out = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for row, col, value in zip(self.rows, self.cols, self.values, strict=True):
            out[int(row), int(col)] += complex(value)
        return out

    def apply_numpy(self, x: Array) -> Array:
        arr = np.asarray(x, dtype=np.complex128)
        out = np.zeros_like(arr)
        flat_in = arr.reshape((-1, self.dim))
        flat_out = out.reshape((-1, self.dim))
        for row_in, row_out in zip(flat_in, flat_out, strict=True):
            np.add.at(row_out, self.rows, self.values * row_in[self.cols])
        return out


def four_valent_channels(two_js: tuple[int, int, int, int]) -> tuple[int, ...]:
    j1, j2, j3, j4 = (int(x) for x in two_js)
    lo = max(abs(j1 - j2), abs(j3 - j4))
    hi = min(j1 + j2, j3 + j4)
    if hi < lo:
        return ()
    return tuple(
        channel
        for channel in range(lo, hi + 1)
        if (j1 + j2 + channel) % 2 == 0 and (j3 + j4 + channel) % 2 == 0
    )


def four_valent_offdiag(
    two_js: tuple[int, int, int, int],
    upper_channel: int,
) -> float:

    j1, j2, j3, j4 = (float(x) for x in two_js)
    x = float(upper_channel)
    if x <= 1.0:
        return 0.0
    left = (x + j1 + j2 + 2.0) * (-j1 + j2 + x) * (j1 - j2 + x) * (j1 + j2 - x + 2.0)
    right = (x + j3 + j4 + 2.0) * (-j3 + j4 + x) * (j3 - j4 + x) * (j3 + j4 - x + 2.0)
    product = left * right
    if product <= 1.0e-28:
        return 0.0
    return float(np.sqrt(product) / (16.0 * np.sqrt((x - 1.0) * (x + 1.0))))


def four_valent_offdiagonal_for_channels(
    two_js: tuple[int, int, int, int],
    channels: tuple[int, ...] | None = None,
) -> Array:

    local_channels = tuple(four_valent_channels(two_js) if channels is None else channels)
    if len(local_channels) <= 1:
        return np.zeros((0,), dtype=np.float64)
    expected = four_valent_channels(two_js)
    if tuple(local_channels) != expected:
        raise ValueError(f"channels={local_channels!r} are not the admissible j12 channels {expected!r}.")
    return np.asarray(
        [four_valent_offdiag(two_js, int(local_channels[idx])) for idx in range(1, len(local_channels))],
        dtype=np.float64,
    )


def four_valent_q_matrix(
    two_js: tuple[int, int, int, int],
    *,
    sigma: int | float = 1,
    channels: tuple[int, ...] | None = None,
) -> Array:

    local_channels = tuple(four_valent_channels(two_js) if channels is None else channels)
    offdiag = four_valent_offdiagonal_for_channels(two_js, local_channels)
    out = np.zeros((len(local_channels), len(local_channels)), dtype=np.complex128)
    for idx, magnitude in enumerate(offdiag, start=1):
        out[idx - 1, idx] = -1.0j * float(sigma) * magnitude
        out[idx, idx - 1] = 1.0j * float(sigma) * magnitude
    return out


def four_valent_transitions(
    two_js: tuple[int, int, int, int],
    *,
    sigma: int | float = 1,
    channels: tuple[int, ...] | None = None,
) -> SparseQTransitions:

    local_channels = tuple(four_valent_channels(two_js) if channels is None else channels)
    offdiag = four_valent_offdiagonal_for_channels(two_js, local_channels)
    rows: list[int] = []
    cols: list[int] = []
    values: list[complex] = []
    for idx, magnitude in enumerate(offdiag, start=1):
        coeff = 1.0j * float(sigma) * float(magnitude)
        rows.extend((idx - 1, idx))
        cols.extend((idx, idx - 1))
        values.extend((-coeff, coeff))
    return SparseQTransitions(
        rows=np.asarray(rows, dtype=np.int32),
        cols=np.asarray(cols, dtype=np.int32),
        values=np.asarray(values, dtype=np.complex128),
        dim=len(local_channels),
        label="BT four-valent sparse Q",
    )


def q123_left_comb_transitions(
    two_js: tuple[int, ...],
    channel_tuples: tuple[tuple[int, ...], ...],
    *,
    sigma: int | float = 1,
) -> SparseQTransitions:

    if len(two_js) < 4:
        return SparseQTransitions(
            rows=np.zeros((0,), dtype=np.int32),
            cols=np.zeros((0,), dtype=np.int32),
            values=np.zeros((0,), dtype=np.complex128),
            dim=len(channel_tuples),
            label="BT q123 sparse Q",
        )
    expected_channels = max(len(two_js) - 3, 0)
    for channels in channel_tuples:
        if len(channels) != expected_channels:
            raise ValueError(f"Expected {expected_channels} left-comb channels, got {channels!r}.")

    index = {tuple(channels): pos for pos, channels in enumerate(channel_tuples)}
    rows: list[int] = []
    cols: list[int] = []
    values: list[complex] = []
    for lower_pos, lower_channels in enumerate(channel_tuples):
        if not lower_channels:
            continue
        upper_channels = (int(lower_channels[0]) + 2, *tuple(int(x) for x in lower_channels[1:]))
        upper_pos = index.get(upper_channels)
        if upper_pos is None:
            continue
        upper_a2 = upper_channels[0]
        a3 = int(two_js[3]) if len(two_js) == 4 else upper_channels[1]
        magnitude = four_valent_offdiag((int(two_js[0]), int(two_js[1]), int(two_js[2]), a3), upper_a2)
        coeff = 1.0j * float(sigma) * float(magnitude)
        rows.extend((lower_pos, upper_pos))
        cols.extend((upper_pos, lower_pos))
        values.extend((-coeff, coeff))

    return SparseQTransitions(
        rows=np.asarray(rows, dtype=np.int32),
        cols=np.asarray(cols, dtype=np.int32),
        values=np.asarray(values, dtype=np.complex128),
        dim=len(channel_tuples),
        label="BT q123 sparse Q",
    )


def q123_left_comb_q_matrix(
    two_js: tuple[int, ...],
    channel_tuples: tuple[tuple[int, ...], ...],
    *,
    sigma: int | float = 1,
) -> Array:

    return q123_left_comb_transitions(two_js, channel_tuples, sigma=sigma).dense()


def transitions_from_matrix(matrix: Array, *, tol: float = 1.0e-13, label: str = "sparse Q") -> SparseQTransitions:
    q = np.asarray(matrix, dtype=np.complex128)
    rows, cols = np.nonzero(np.abs(q) > tol)
    return SparseQTransitions(
        rows=rows.astype(np.int32),
        cols=cols.astype(np.int32),
        values=q[rows, cols].astype(np.complex128),
        dim=int(q.shape[0]),
        label=label,
    )


def apply_four_valent_q_numpy(offdiag: Array, sigma: int | float, x: Array) -> Array:
    arr = np.asarray(x, dtype=np.complex128)
    coeff = 1.0j * float(sigma) * np.asarray(offdiag, dtype=np.float64)
    out = np.zeros_like(arr)
    out[..., :-1] += -coeff * arr[..., 1:]
    out[..., 1:] += coeff * arr[..., :-1]
    return out


@jax.jit
def apply_four_valent_q_jax(offdiag: jax.Array, sigma: jax.Array, x: jax.Array) -> jax.Array:
    coeff = (1.0j * sigma) * offdiag
    out = jnp.zeros_like(x)
    out = out.at[..., :-1].add(-coeff * x[..., 1:])
    out = out.at[..., 1:].add(coeff * x[..., :-1])
    return out


@jax.jit
def apply_sparse_q_jax(rows: jax.Array, cols: jax.Array, values: jax.Array, x: jax.Array) -> jax.Array:
    out = jnp.zeros_like(x)
    return out.at[..., rows].add(values * x[..., cols])


def make_numpy_applyq_from_transitions(transitions: SparseQTransitions) -> Callable[[Array], Array]:
    return transitions.apply_numpy
