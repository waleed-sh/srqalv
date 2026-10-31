from __future__ import annotations

import time

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Callable

import numpy as np

from srqalv.basis import K5Graph
from srqalv.basis import SU2Hilbert
from srqalv.basis import orientation_sign

from srqalv.bounds import analytic_lambda_for_sector
from srqalv.bounds import reduced_sigma
from srqalv.baseline import build_dense_volume_operators
from srqalv.baseline import local_q_commutator
from srqalv.baseline import trial_wavefunction
from srqalv.srq import exact_volume_from_q

Array = np.ndarray


@dataclass(frozen=True)
class LocalSector:

    cutoff: int
    vertex: int
    two_js: tuple[int, int, int, int]
    channels: tuple[int, ...]
    cols: Array
    source_locs: Array
    rows_by_target: Array
    q: Array
    v_exact: Array
    rho: float
    analytic_lambda: float
    sigma: int


@dataclass(frozen=True)
class CutoffSystem:

    cutoff: int
    graph: K5Graph
    space: SU2Hilbert
    states: Array
    sectors: list[LocalSector]
    psi: Array
    probabilities: Array
    exact_action: Array
    exact_value: complex
    exact_local: Array
    build_seconds: float
    dense_validation_action_rel: float | None
    dense_validation_expectation_abs: float | None


def sector_q_and_volume(
    two_js: tuple[int, int, int, int],
    channel_maxima: tuple[int, ...],
    signs: dict[tuple[int, int, int], int],
) -> tuple[tuple[int, ...], Array, Array, float]:

    channels: tuple[int, ...] | None = None
    q_comm_sum: Array | None = None
    for triple in combinations(range(4), 3):
        triple_channels, q_comm = local_q_commutator(two_js, channel_maxima, triple)
        if channels is None:
            channels = triple_channels
            q_comm_sum = np.zeros_like(q_comm)
        elif channels != triple_channels:
            raise RuntimeError(f"Inconsistent local channels for {two_js=}.")
        q_comm_sum = q_comm_sum + int(signs[triple]) * q_comm

    if channels is None or q_comm_sum is None:
        raise RuntimeError("A 4-valent sector must generate grasping triples.")

    q_h = 1.0j * q_comm_sum
    q_h = 0.5 * (q_h + q_h.conj().T)
    q_h[np.abs(q_h) < 1.0e-13] = 0.0
    rho = float(np.max(np.abs(np.linalg.eigvalsh(q_h)))) if q_h.size else 0.0
    return channels, q_h, exact_volume_from_q(q_h), rho


def build_cutoff_system(cutoff: int, *, embedding_seed: int = 1) -> CutoffSystem:
    start = time.perf_counter()
    graph = K5Graph(random_embedding=True, random_embedding_seed=embedding_seed)
    space = SU2Hilbert(graph, two_j_max=cutoff, exact=True, max_exact_states=10_000_000)
    states = np.asarray(space.all_states(max_states=10_000_000), dtype=np.int32)
    basis_index = {tuple(row.tolist()): idx for idx, row in enumerate(states)}
    sectors: list[LocalSector] = []

    for vertex in range(space.n_vertices):
        vertex_id = space.layout.vertex_ids[vertex]
        incident_edges = tuple(int(x) for x in space.metadata.incident_edges_by_vertex[vertex])
        incident_tokens = tuple(space.metadata.edge_tokens[edge] for edge in incident_edges)
        channel_sites = tuple(int(x) for x in space.layout.vertex_channel_sites[vertex])
        channel_maxima = tuple(int(x) for x in space.layout.vertex_channel_max_labels[vertex])
        if len(incident_edges) != 4 or len(channel_sites) != 1:
            raise ValueError("The publication K5 study expects five 4-valent vertices.")
        channel_site = channel_sites[0]

        signs = {
            triple: int(orientation_sign(graph, vertex_id, tuple(incident_tokens[i] for i in triple)))
            for triple in combinations(range(4), 3)
        }
        sigma = reduced_sigma(signs)

        grouped_cols: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
        for idx, state in enumerate(states):
            grouped_cols[tuple(int(state[edge]) for edge in incident_edges)].append(idx)

        for two_js, cols_list in sorted(grouped_cols.items()):
            channels, q_h, v_exact, rho = sector_q_and_volume(two_js, channel_maxima, signs)
            channel_to_loc = {channel: loc for loc, channel in enumerate(channels)}
            cols = np.asarray(cols_list, dtype=np.int64)
            source_locs = np.asarray(
                [channel_to_loc[int(states[col, channel_site])] for col in cols],
                dtype=np.int64,
            )
            rows_by_target = np.empty((cols.size, len(channels)), dtype=np.int64)
            for row_pos, col in enumerate(cols):
                source_state = states[col]
                for target_loc, target_channel in enumerate(channels):
                    target = source_state.copy()
                    target[channel_site] = target_channel
                    rows_by_target[row_pos, target_loc] = basis_index[tuple(target.tolist())]

            sectors.append(
                LocalSector(
                    cutoff=cutoff,
                    vertex=vertex,
                    two_js=two_js,
                    channels=channels,
                    cols=cols,
                    source_locs=source_locs,
                    rows_by_target=rows_by_target,
                    q=q_h,
                    v_exact=v_exact,
                    rho=rho,
                    analytic_lambda=analytic_lambda_for_sector(two_js, sigma),
                    sigma=sigma,
                )
            )

    psi = trial_wavefunction(states, n_edges=space.n_edges)
    probabilities = np.abs(psi) ** 2
    exact_action = apply_total(sectors, psi, matrix_getter=lambda sector: sector.v_exact)
    exact_value = expectation_from_action(psi, exact_action)
    exact_local = exact_action / psi

    dense_action_rel: float | None = None
    dense_expectation_abs: float | None = None
    if cutoff == 1:
        dense = build_dense_volume_operators(two_j_max=1, embedding_seed=embedding_seed)
        dense_action = dense.total_volume @ psi
        dense_action_rel = float(np.linalg.norm(exact_action - dense_action) / np.linalg.norm(dense_action))
        dense_value = expectation_from_action(psi, dense_action)
        dense_expectation_abs = abs(exact_value - dense_value)

    return CutoffSystem(
        cutoff=cutoff,
        graph=graph,
        space=space,
        states=states,
        sectors=sectors,
        psi=psi,
        probabilities=probabilities,
        exact_action=exact_action,
        exact_value=exact_value,
        exact_local=exact_local,
        build_seconds=time.perf_counter() - start,
        dense_validation_action_rel=dense_action_rel,
        dense_validation_expectation_abs=dense_expectation_abs,
    )


def apply_sector_matrix(sector: LocalSector, matrix: Array, vector: Array, out: Array) -> None:
    values = vector[sector.cols]
    for target_loc in range(matrix.shape[0]):
        amplitudes = matrix[target_loc, sector.source_locs] * values
        np.add.at(out, sector.rows_by_target[:, target_loc], amplitudes)


def apply_total(
    sectors: list[LocalSector],
    vector: Array,
    *,
    matrix_getter: Callable[[LocalSector], Array],
) -> Array:

    out = np.zeros_like(vector, dtype=np.complex128)
    for sector in sectors:
        apply_sector_matrix(sector, matrix_getter(sector), vector, out)
    out[np.abs(out) < 1.0e-13] = 0.0
    return out


def expectation_from_action(psi: Array, action: Array) -> complex:
    return complex(np.vdot(psi, action) / np.vdot(psi, psi))


def local_estimator(action: Array, psi: Array) -> Array:
    return action / psi
