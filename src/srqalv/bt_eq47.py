from __future__ import annotations

import math

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np

from srqalv.bt_applyq import SparseQTransitions
from srqalv.bt_applyq import four_valent_offdiag

Array = np.ndarray


def _triangle_ok(a: int, b: int, c: int) -> bool:
    return abs(a - b) <= c <= a + b and ((a + b + c) % 2) == 0


@lru_cache(maxsize=None)
def _delta_factor(a: int, b: int, c: int) -> float:
    if not _triangle_ok(a, b, c):
        return 0.0
    x = (a + b - c) // 2
    y = (a - b + c) // 2
    z = (-a + b + c) // 2
    w = (a + b + c) // 2 + 1
    return math.exp(0.5 * (math.lgamma(x + 1) + math.lgamma(y + 1) + math.lgamma(z + 1) - math.lgamma(w + 1)))


@lru_cache(maxsize=None)
def wigner_6j(
    j1: int,
    j2: int,
    j3: int,
    l1: int,
    l2: int,
    l3: int,
) -> float:

    labels = tuple(int(x) for x in (j1, j2, j3, l1, l2, l3))
    j1, j2, j3, l1, l2, l3 = labels
    prefactor = (
        _delta_factor(j1, j2, j3)
        * _delta_factor(j1, l2, l3)
        * _delta_factor(l1, j2, l3)
        * _delta_factor(l1, l2, j3)
    )
    if prefactor == 0.0:
        return 0.0

    sums = (
        j1 + j2 + j3,
        j1 + l2 + l3,
        l1 + j2 + l3,
        l1 + l2 + j3,
        j1 + j2 + l1 + l2,
        j1 + j3 + l1 + l3,
        j2 + j3 + l2 + l3,
    )
    if any(total % 2 for total in sums):
        return 0.0
    a1, a2, a3, a4 = (total // 2 for total in sums[:4])
    b1, b2, b3 = (total // 2 for total in sums[4:])
    z_min = max(a1, a2, a3, a4)
    z_max = min(b1, b2, b3)
    if z_min > z_max:
        return 0.0

    total = 0.0
    for z in range(z_min, z_max + 1):
        den_args = (z - a1, z - a2, z - a3, z - a4, b1 - z, b2 - z, b3 - z)
        if any(arg < 0 for arg in den_args):
            continue
        log_term = math.lgamma(z + 2) - sum(math.lgamma(arg + 1) for arg in den_args)
        total += (-1.0 if z % 2 else 1.0) * math.exp(log_term)
    return prefactor * total


def _phase_from_twice(total_twice: int) -> float:
    total_twice = int(total_twice)
    if total_twice % 2:
        return 0.0
    return -1.0 if ((total_twice // 2) % 2) else 1.0


def _sqrt_dim(two_a: int, two_b: int) -> float:
    return math.sqrt((int(two_a) + 1.0) * (int(two_b) + 1.0))


def _x_factor(two_a: int, two_b: int) -> float:
    a = float(two_a)
    b = float(two_b)
    return a * (a + 1.0) * (a + 2.0) * b * (b + 1.0) * (b + 2.0)


def _casimir(two_a: int) -> float:
    a = float(two_a)
    return a * (a + 2.0) / 4.0


def _full_a(two_js: tuple[int, ...], channels: tuple[int, ...], n: int) -> int:
    valence = len(two_js)
    if n == 1:
        return int(two_js[0])
    if 2 <= n <= valence - 2:
        return int(channels[n - 2])
    if n == valence - 1:
        return int(two_js[-1])
    if n == valence:
        return 0
    raise IndexError(f"a_{n} is out of range for valence {valence}.")


def _same_outside(
    two_js: tuple[int, ...],
    channels: tuple[int, ...],
    channels_p: tuple[int, ...],
    ranges: tuple[range, ...],
) -> bool:

    return all(
        _full_a(two_js, channels, n) == _full_a(two_js, channels_p, n)
        for one_range in ranges
        for n in one_range
    )


def q123_matrix_element(
    two_js: tuple[int, ...],
    channels: tuple[int, ...],
    channels_p: tuple[int, ...],
) -> float:

    if len(channels) != len(channels_p):
        return 0.0
    if channels[1:] != channels_p[1:]:
        return 0.0
    a2 = int(channels[0])
    a2p = int(channels_p[0])
    if a2 == a2p + 2:
        a3 = int(two_js[3]) if len(two_js) == 4 else int(channels[1])
        return four_valent_offdiag((two_js[0], two_js[1], two_js[2], a3), a2)
    if a2p == a2 + 2:
        a3 = int(two_js[3]) if len(two_js) == 4 else int(channels[1])
        return -four_valent_offdiag((two_js[0], two_js[1], two_js[2], a3), a2p)
    return 0.0


def bt_matrix_element(
    two_js: tuple[int, ...],
    channels: tuple[int, ...],
    channels_p: tuple[int, ...],
    triple: tuple[int, int, int],
) -> float:

    spins = tuple(int(x) for x in two_js)
    valence = len(spins)
    I, J, K = (int(x) + 1 for x in triple)
    # the BT expressions use one-based leg indices
    if not (1 <= I < J < K <= valence):
        raise ValueError(f"Invalid ordered triple {triple!r}.")
    if len(channels) != max(valence - 3, 0) or len(channels_p) != max(valence - 3, 0):
        raise ValueError("Channel tuple length does not match gauge-invariant valence.")

    if (I, J, K) == (1, 2, 3):
        return q123_matrix_element(spins, channels, channels_p)
    if I == 1 and J == 2:
        return _case_i1_j2(spins, channels, channels_p, K)
    if I == 1:
        return _case_i1(spins, channels, channels_p, J, K)
    return _case_generic(spins, channels, channels_p, I, J, K)


def _case_i1_j2(
    two_js: tuple[int, ...],
    channels: tuple[int, ...],
    channels_p: tuple[int, ...],
    K: int,
) -> float:
    # eq 57 from BT, with q123 handled before this branch

    N = len(two_js)
    if not _same_outside(two_js, channels, channels_p, (range(K, N + 1),)):
        return 0.0

    a2 = _full_a(two_js, channels, 2)
    a2p = _full_a(two_js, channels_p, 2)
    aK = _full_a(two_js, channels, K)
    aKm1 = _full_a(two_js, channels, K - 1)
    aKm1p = _full_a(two_js, channels_p, K - 1)

    phase_twice = two_js[K - 1] - two_js[0] - two_js[1] + aK + 2
    phase_twice += sum(two_js[n - 1] for n in range(2, 2))
    phase_twice -= sum(two_js[p - 1] for p in range(3, K))
    value = 0.25 * _phase_from_twice(phase_twice)
    value *= math.sqrt(_x_factor(two_js[1], two_js[K - 1]))
    value *= _sqrt_dim(a2, a2p)
    value *= wigner_6j(two_js[0], two_js[1], a2, 2, a2p, two_js[1])
    for n in range(3, K):
        an = _full_a(two_js, channels, n)
        anp = _full_a(two_js, channels_p, n)
        anm1 = _full_a(two_js, channels, n - 1)
        anm1p = _full_a(two_js, channels_p, n - 1)
        value *= _sqrt_dim(an, anp)
        value *= _phase_from_twice(anm1p + anm1 + 2)
        value *= wigner_6j(two_js[n - 1], anm1p, anp, 2, an, anm1)
    value *= wigner_6j(aK, two_js[K - 1], aKm1, 2, aKm1p, two_js[K - 1])
    value *= _casimir(a2) - _casimir(a2p)


    return float(-2.0 * value)


def _bracket_j(
    two_js: tuple[int, ...],
    channels: tuple[int, ...],
    channels_p: tuple[int, ...],
    J: int,
) -> float:

    aJ = _full_a(two_js, channels, J)
    aJp = _full_a(two_js, channels_p, J)
    aJm1 = _full_a(two_js, channels, J - 1)
    aJm1p = _full_a(two_js, channels_p, J - 1)
    jJ = two_js[J - 1]
    term1 = _phase_from_twice(aJp + aJm1p)
    term1 *= wigner_6j(aJ, jJ, aJm1p, 2, aJm1, jJ)
    term1 *= wigner_6j(aJm1p, jJ, aJp, 2, aJ, jJ)
    term2 = _phase_from_twice(aJ + aJm1)
    term2 *= wigner_6j(aJp, jJ, aJm1, 2, aJm1p, jJ)
    term2 *= wigner_6j(aJm1, jJ, aJ, 2, aJp, jJ)
    return term1 - term2


def _case_i1(
    two_js: tuple[int, ...],
    channels: tuple[int, ...],
    channels_p: tuple[int, ...],
    J: int,
    K: int,
) -> float:
    # eq 56 from BT for I = 1 and J > 2

    N = len(two_js)
    if not _same_outside(two_js, channels, channels_p, (range(K, N + 1),)):
        return 0.0
    aK = _full_a(two_js, channels, K)
    aKm1 = _full_a(two_js, channels, K - 1)
    aKm1p = _full_a(two_js, channels_p, K - 1)
    aJ = _full_a(two_js, channels, J)
    aJp = _full_a(two_js, channels_p, J)

    phase_twice = two_js[K - 1] - two_js[0] + aK + 2
    phase_twice += sum(two_js[n - 1] for n in range(2, J))
    phase_twice -= sum(two_js[p - 1] for p in range(J + 1, K))
    value = 0.25 * _phase_from_twice(phase_twice)
    value *= math.sqrt(_x_factor(two_js[0], two_js[J - 1]))
    value *= math.sqrt(_x_factor(two_js[J - 1], two_js[K - 1]))
    value *= _sqrt_dim(aJ, aJp)
    for n in range(2, J):
        an = _full_a(two_js, channels, n)
        anp = _full_a(two_js, channels_p, n)
        anm1 = _full_a(two_js, channels, n - 1)
        anm1p = _full_a(two_js, channels_p, n - 1)
        value *= _sqrt_dim(an, anp)
        value *= _phase_from_twice(anm1p + anm1 + 2)
        value *= wigner_6j(two_js[n - 1], anm1p, anp, 2, an, anm1)
    for n in range(J + 1, K):
        an = _full_a(two_js, channels, n)
        anp = _full_a(two_js, channels_p, n)
        anm1 = _full_a(two_js, channels, n - 1)
        anm1p = _full_a(two_js, channels_p, n - 1)
        value *= _sqrt_dim(an, anp)
        value *= _phase_from_twice(anm1p + anm1 + 2)
        value *= wigner_6j(two_js[n - 1], anm1p, anp, 2, an, anm1)
    value *= wigner_6j(aK, two_js[K - 1], aKm1, 2, aKm1p, two_js[K - 1])
    value *= _bracket_j(two_js, channels, channels_p, J)
    return float(value)


def _case_generic(
    two_js: tuple[int, ...],
    channels: tuple[int, ...],
    channels_p: tuple[int, ...],
    I: int,
    J: int,
    K: int,
) -> float:
    # eq 47 from BT in the left-comb basis

    N = len(two_js)
    if not _same_outside(two_js, channels, channels_p, (range(2, I), range(K, N + 1))):
        return 0.0
    aIm1 = _full_a(two_js, channels, I - 1)
    aI = _full_a(two_js, channels, I)
    aIp = _full_a(two_js, channels_p, I)
    aJ = _full_a(two_js, channels, J)
    aJp = _full_a(two_js, channels_p, J)
    aK = _full_a(two_js, channels, K)
    aKm1 = _full_a(two_js, channels, K - 1)
    aKm1p = _full_a(two_js, channels_p, K - 1)

    phase_twice = two_js[K - 1] + two_js[I - 1] + aIm1 + aK
    phase_twice += aI - aIp
    phase_twice += sum(two_js[n - 1] for n in range(I + 1, J))
    phase_twice -= sum(two_js[p - 1] for p in range(J + 1, K))
    value = 0.25 * _phase_from_twice(phase_twice)
    value *= math.sqrt(_x_factor(two_js[I - 1], two_js[J - 1]))
    value *= math.sqrt(_x_factor(two_js[J - 1], two_js[K - 1]))
    value *= _sqrt_dim(aI, aIp)
    value *= _sqrt_dim(aJ, aJp)
    value *= wigner_6j(aIm1, two_js[I - 1], aI, 2, aIp, two_js[I - 1])
    for n in range(I + 1, J):
        an = _full_a(two_js, channels, n)
        anp = _full_a(two_js, channels_p, n)
        anm1 = _full_a(two_js, channels, n - 1)
        anm1p = _full_a(two_js, channels_p, n - 1)
        value *= _sqrt_dim(an, anp)
        value *= _phase_from_twice(anm1p + anm1 + 2)
        value *= wigner_6j(two_js[n - 1], anm1p, anp, 2, an, anm1)
    for n in range(J + 1, K):
        an = _full_a(two_js, channels, n)
        anp = _full_a(two_js, channels_p, n)
        anm1 = _full_a(two_js, channels, n - 1)
        anm1p = _full_a(two_js, channels_p, n - 1)
        value *= _sqrt_dim(an, anp)
        value *= _phase_from_twice(anm1p + anm1 + 2)
        value *= wigner_6j(two_js[n - 1], anm1p, anp, 2, an, anm1)
    value *= wigner_6j(aK, two_js[K - 1], aKm1, 2, aKm1p, two_js[K - 1])
    value *= _bracket_j(two_js, channels, channels_p, J)
    return float(value)


def bt_q_commutator_matrix(
    two_js: tuple[int, ...],
    channel_tuples: tuple[tuple[int, ...], ...],
    triple: tuple[int, int, int],
    *,
    tol: float = 1.0e-13,
) -> Array:

    dim = len(channel_tuples)
    out = np.zeros((dim, dim), dtype=np.float64)
    for row, channels in enumerate(channel_tuples):
        for col, channels_p in enumerate(channel_tuples):
            out[row, col] = bt_matrix_element(two_js, channels, channels_p, triple)
    out[np.abs(out) < tol] = 0.0
    return out


def bt_q_transitions(
    two_js: tuple[int, ...],
    channel_tuples: tuple[tuple[int, ...], ...],
    triple: tuple[int, int, int],
    *,
    sigma: int | float = 1,
    tol: float = 1.0e-13,
) -> SparseQTransitions:

    q_comm = bt_q_commutator_matrix(two_js, channel_tuples, triple, tol=tol)
    q_h = 1.0j * float(sigma) * q_comm
    rows, cols = np.nonzero(np.abs(q_h) > tol)
    return SparseQTransitions(
        rows=rows.astype(np.int32),
        cols=cols.astype(np.int32),
        values=q_h[rows, cols].astype(np.complex128),
        dim=len(channel_tuples),
        label=f"BT Eq47 q{tuple(int(x)+1 for x in triple)}",
    )


def coalesce_sparse_transitions(
    transitions: SparseQTransitions,
    *,
    tol: float = 1.0e-13,
    label: str | None = None,
) -> SparseQTransitions:

    if transitions.rows.size == 0:
        return SparseQTransitions(
            rows=np.zeros((0,), dtype=np.int32),
            cols=np.zeros((0,), dtype=np.int32),
            values=np.zeros((0,), dtype=np.complex128),
            dim=transitions.dim,
            label=label or transitions.label,
        )

    keys = transitions.rows.astype(np.int64) * int(transitions.dim) + transitions.cols.astype(np.int64)
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    values_sorted = transitions.values[order]
    starts = np.r_[0, np.nonzero(keys_sorted[1:] != keys_sorted[:-1])[0] + 1]
    summed = np.add.reduceat(values_sorted, starts)
    kept_keys = keys_sorted[starts]
    mask = np.abs(summed) > float(tol)
    kept_keys = kept_keys[mask]
    summed = summed[mask]
    return SparseQTransitions(
        rows=(kept_keys // int(transitions.dim)).astype(np.int32),
        cols=(kept_keys % int(transitions.dim)).astype(np.int32),
        values=summed.astype(np.complex128),
        dim=transitions.dim,
        label=label or transitions.label,
    )


def bt_total_q_transitions(
    two_js: tuple[int, ...],
    channel_tuples: tuple[tuple[int, ...], ...],
    *,
    triple_signs: dict[tuple[int, int, int], int] | None = None,
    tol: float = 1.0e-13,
    coalesce: bool = True,
) -> SparseQTransitions:

    valence = len(two_js)
    triples = tuple(combinations(range(valence), 3))
    if triple_signs is None:
        signs = {triple: 1 for triple in triples}
    else:
        signs = {tuple(sorted(tuple(int(x) for x in triple))): int(sign) for triple, sign in triple_signs.items()}

    rows: list[Array] = []
    cols: list[Array] = []
    values: list[Array] = []
    dim = len(channel_tuples)
    for triple in triples:
        sign = signs.get(triple, 0)
        if sign == 0:
            continue
        part = bt_q_transitions(two_js, channel_tuples, triple, sigma=sign, tol=tol)
        rows.append(part.rows)
        cols.append(part.cols)
        values.append(part.values)

    if not rows:
        combined = SparseQTransitions(
            rows=np.zeros((0,), dtype=np.int32),
            cols=np.zeros((0,), dtype=np.int32),
            values=np.zeros((0,), dtype=np.complex128),
            dim=dim,
            label="BT Eq47 total Q",
        )
    else:
        combined = SparseQTransitions(
            rows=np.concatenate(rows).astype(np.int32),
            cols=np.concatenate(cols).astype(np.int32),
            values=np.concatenate(values).astype(np.complex128),
            dim=dim,
            label="BT Eq47 total Q",
        )
    if coalesce:
        return coalesce_sparse_transitions(combined, tol=tol, label="BT Eq47 total Q")
    return combined


def sparse_row_sum_bound(transitions: SparseQTransitions) -> float:
    row_sums = np.zeros((int(transitions.dim),), dtype=np.float64)
    if transitions.rows.size:
        np.add.at(row_sums, transitions.rows, np.abs(transitions.values))
    return float(np.max(row_sums)) if row_sums.size else 0.0


@dataclass(frozen=True)
class BTValidationResult:

    two_js: tuple[int, ...]
    triple: tuple[int, int, int]
    dim: int
    max_abs_error: float
    antisymmetry_error: float
