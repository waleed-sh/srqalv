from __future__ import annotations
from functools import lru_cache
from typing import Iterable


def coupling_ok_py(a: int, b: int, c: int) -> bool:
    a_i = int(a)
    b_i = int(b)
    c_i = int(c)
    return abs(a_i - b_i) <= c_i <= a_i + b_i and ((a_i + b_i + c_i) & 1) == 0


def expected_channel_count(valence: int) -> int:
    valence_i = int(valence)
    if valence_i < 0:
        raise ValueError("valence must be non-negative.")
    return max(valence_i - 3, 0)


def left_comb_valid_py(edge_two_j: Iterable[int], channels: Iterable[int]) -> bool:
    edges = tuple(int(x) for x in edge_two_j)
    ch = tuple(int(x) for x in channels)
    valence = len(edges)
    if len(ch) != expected_channel_count(valence):
        return False
    if valence == 0:
        return True
    if valence == 1:
        return edges[0] == 0
    if valence == 2:
        return edges[0] == edges[1]
    if valence == 3:
        return coupling_ok_py(edges[0], edges[1], edges[2])

    if not coupling_ok_py(edges[0], edges[1], ch[0]):
        return False
    for idx in range(1, len(ch)):
        if not coupling_ok_py(ch[idx - 1], edges[idx + 1], ch[idx]):
            return False
    return coupling_ok_py(ch[-1], edges[-2], edges[-1])


def allowed_couplings(a: int, b: int, max_label: int) -> tuple[int, ...]:
    lo = abs(int(a) - int(b))
    hi = min(int(a) + int(b), int(max_label))
    if hi < lo:
        return ()
    parity = (int(a) + int(b)) & 1
    return tuple(c for c in range(lo, hi + 1) if (c & 1) == parity)


def all_left_comb_channels(
    edge_two_j: Iterable[int],
    channel_max_labels: Iterable[int],
    *,
    limit: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    if limit is not None and limit < 1:
        return ()
    edges = tuple(int(x) for x in edge_two_j)
    maxima = tuple(int(x) for x in channel_max_labels)
    valence = len(edges)
    if len(maxima) != expected_channel_count(valence):
        return ()
    if valence <= 3:
        return ((),) if left_comb_valid_py(edges, ()) else ()

    n_channels = len(maxima)

    @lru_cache(maxsize=None)
    def feasible(idx: int, previous: int) -> bool:
        if idx == n_channels:
            return coupling_ok_py(previous, edges[-2], edges[-1])
        source = edges[0] if idx == 0 else previous
        partner = edges[1] if idx == 0 else edges[idx + 1]
        return any(
            feasible(idx + 1, candidate)
            for candidate in allowed_couplings(source, partner, maxima[idx])
        )

    out: list[tuple[int, ...]] = []

    def visit(idx: int, previous: int, prefix: tuple[int, ...]) -> None:
        if limit is not None and len(out) >= limit:
            return
        if idx == n_channels:
            if coupling_ok_py(previous, edges[-2], edges[-1]):
                out.append(prefix)
            return
        source = edges[0] if idx == 0 else previous
        partner = edges[1] if idx == 0 else edges[idx + 1]
        for candidate in allowed_couplings(source, partner, maxima[idx]):
            if feasible(idx + 1, candidate):
                visit(idx + 1, candidate, (*prefix, candidate))

    visit(0, 0, ())
    return tuple(out)

