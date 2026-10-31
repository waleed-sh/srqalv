from __future__ import annotations
import math
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
import numpy as np
from srqalv.basis import K5Graph, SU2Hilbert, all_left_comb_channels, orientation_sign
Array = np.ndarray


@dataclass(frozen=True)
class DenseVolumeOperators:

    states: Array
    q_commutators: dict[tuple[int, tuple[int, int, int]], Array]
    q_hermitian: dict[tuple[int, tuple[int, int, int]], Array]
    q_vertex_commutators: dict[int, Array]
    q_vertex_hermitian: dict[int, Array]
    vertex_volumes: dict[int, Array]
    total_volume: Array
    total_oriented_density: Array
    triple_signs: dict[int, dict[tuple[int, int, int], int]]


def m_values(two_j: int) -> tuple[int, ...]:
    two_j = int(two_j)
    return tuple(range(-two_j, two_j + 1, 2))


def factorial(n: int) -> float:
    if n < 0:
        return 0.0
    return math.gamma(n + 1)


def triangle_ok(two_j1: int, two_j2: int, two_j3: int) -> bool:
    return (
        abs(two_j1 - two_j2) <= two_j3 <= two_j1 + two_j2
        and ((two_j1 + two_j2 + two_j3) % 2) == 0
    )


def wigner_3j(
    two_j1: int,
    two_j2: int,
    two_j3: int,
    two_m1: int,
    two_m2: int,
    two_m3: int,
) -> float:

    if two_m1 + two_m2 + two_m3 != 0:
        return 0.0
    if not triangle_ok(two_j1, two_j2, two_j3):
        return 0.0
    if any(abs(m) > j for m, j in ((two_m1, two_j1), (two_m2, two_j2), (two_m3, two_j3))):
        return 0.0
    if any(((j - m) % 2) for m, j in ((two_m1, two_j1), (two_m2, two_j2), (two_m3, two_j3))):
        return 0.0

    a = (two_j1 + two_j2 - two_j3) // 2
    b = (two_j1 - two_j2 + two_j3) // 2
    c = (-two_j1 + two_j2 + two_j3) // 2
    d = (two_j1 + two_j2 + two_j3) // 2 + 1

    log_delta = math.log(factorial(a)) + math.log(factorial(b)) + math.log(factorial(c))
    log_delta -= math.log(factorial(d))

    log_m = 0.0
    for two_j, two_m in (
        (two_j1, two_m1),
        (two_j1, -two_m1),
        (two_j2, two_m2),
        (two_j2, -two_m2),
        (two_j3, two_m3),
        (two_j3, -two_m3),
    ):
        log_m += math.log(factorial((two_j + two_m) // 2))

    z_min = max(
        0,
        (two_j2 - two_j3 - two_m1) // 2,
        (two_j1 - two_j3 + two_m2) // 2,
    )
    z_max = min(
        (two_j1 + two_j2 - two_j3) // 2,
        (two_j1 - two_m1) // 2,
        (two_j2 + two_m2) // 2,
    )
    if z_min > z_max:
        return 0.0

    total = 0.0
    for z in range(z_min, z_max + 1):
        den_args = (
            z,
            (two_j1 + two_j2 - two_j3) // 2 - z,
            (two_j1 - two_m1) // 2 - z,
            (two_j2 + two_m2) // 2 - z,
            (two_j3 - two_j2 + two_m1) // 2 + z,
            (two_j3 - two_j1 - two_m2) // 2 + z,
        )
        if any(arg < 0 for arg in den_args):
            continue
        denom = math.prod(factorial(arg) for arg in den_args)
        total += ((-1.0) ** z) / denom

    phase_exp = (two_j1 - two_j2 - two_m3) // 2
    phase = -1.0 if phase_exp % 2 else 1.0
    return phase * math.exp(0.5 * (log_delta + log_m)) * total


def clebsch_gordan(
    two_j1: int,
    two_m1: int,
    two_j2: int,
    two_m2: int,
    two_j: int,
    two_m: int,
) -> float:

    if two_m != two_m1 + two_m2:
        return 0.0
    if not triangle_ok(two_j1, two_j2, two_j):
        return 0.0
    phase_exp = (two_j1 - two_j2 + two_m) // 2
    phase = -1.0 if phase_exp % 2 else 1.0
    return phase * math.sqrt(two_j + 1.0) * wigner_3j(
        two_j1,
        two_j2,
        two_j,
        two_m1,
        two_m2,
        -two_m,
    )


@lru_cache(maxsize=None)
def spin_generators(two_j: int) -> tuple[Array, Array, Array]:
    labels = m_values(two_j)
    index = {two_m: pos for pos, two_m in enumerate(labels)}
    dim = len(labels)
    j = two_j / 2.0
    j_plus = np.zeros((dim, dim), dtype=np.complex128)
    for col, two_m in enumerate(labels):
        target = two_m + 2
        if target not in index:
            continue
        m = two_m / 2.0
        row = index[target]
        j_plus[row, col] = math.sqrt(j * (j + 1.0) - m * (m + 1.0))
    j_minus = j_plus.conj().T
    jx = 0.5 * (j_plus + j_minus)
    jy = (j_plus - j_minus) / (2.0j)
    jz = np.diag(np.asarray(labels, dtype=np.float64) / 2.0).astype(np.complex128)
    return jx, jy, jz


def kron_all(matrices: Iterable[Array]) -> Array:
    iterator = iter(matrices)
    out = np.asarray(next(iterator), dtype=np.complex128)
    for matrix in iterator:
        out = np.kron(out, np.asarray(matrix, dtype=np.complex128))
    return out


@lru_cache(maxsize=None)
def many_body_generator(two_js: tuple[int, ...], leg: int, component: int) -> Array:
    pieces = []
    for idx, two_j in enumerate(two_js):
        dim = two_j + 1
        if idx == leg:
            pieces.append(spin_generators(two_j)[component])
        else:
            pieces.append(np.eye(dim, dtype=np.complex128))
    return kron_all(pieces)


@lru_cache(maxsize=None)
def pair_j_squared(two_js: tuple[int, ...], first: int, second: int) -> Array:
    dim = math.prod(two_j + 1 for two_j in two_js)
    out = np.zeros((dim, dim), dtype=np.complex128)
    for component in range(3):
        total = many_body_generator(two_js, first, component)
        total = total + many_body_generator(two_js, second, component)
        out = out + total @ total
    return out


@lru_cache(maxsize=None)
def left_comb_basis(
    two_js: tuple[int, int, int, int],
    channel_maxima: tuple[int, ...],
) -> tuple[tuple[int, ...], Array]:

    if len(channel_maxima) != 1:
        raise ValueError("This dense baseline is restricted to 4-valent vertices.")

    channel_tuples = all_left_comb_channels(two_js, channel_maxima)
    channels = tuple(int(channel[0]) for channel in channel_tuples)
    magnetic_labels = tuple(product(*(m_values(two_j) for two_j in two_js)))
    magnetic_index = {label: pos for pos, label in enumerate(magnetic_labels)}
    vectors = []

    for two_a in channels:
        vec = np.zeros((len(magnetic_labels),), dtype=np.complex128)
        for two_m1, two_m2, two_m3, two_m4 in magnetic_labels:
            two_ma = two_m1 + two_m2
            two_mb = two_ma + two_m3
            coeff = clebsch_gordan(two_js[0], two_m1, two_js[1], two_m2, two_a, two_ma)
            coeff *= clebsch_gordan(two_a, two_ma, two_js[2], two_m3, two_js[3], two_mb)
            coeff *= clebsch_gordan(two_js[3], two_mb, two_js[3], two_m4, 0, 0)
            if coeff:
                vec[magnetic_index[(two_m1, two_m2, two_m3, two_m4)]] += coeff
        norm = np.linalg.norm(vec)
        if norm <= 1.0e-13:
            raise RuntimeError(f"Built zero intertwiner vector for {two_js=} and {two_a=}.")
        vectors.append(vec / norm)

    if not vectors:
        return channels, np.zeros((len(magnetic_labels), 0), dtype=np.complex128)

    basis = np.column_stack(vectors)
    gram_error = np.linalg.norm(basis.conj().T @ basis - np.eye(len(channels)))
    if gram_error > 1.0e-10:
        raise RuntimeError(f"Left-comb basis is not orthonormal; error={gram_error:.3e}.")
    return channels, basis


@lru_cache(maxsize=None)
def local_q_commutator(
    two_js: tuple[int, int, int, int],
    channel_maxima: tuple[int, ...],
    triple: tuple[int, int, int],
) -> tuple[tuple[int, ...], Array]:

    channels, basis = left_comb_basis(two_js, channel_maxima)
    if basis.shape[1] == 0:
        return channels, np.zeros((0, 0), dtype=np.complex128)

    first, second, third = triple
    left = pair_j_squared(two_js, first, second)
    right = pair_j_squared(two_js, second, third)
    # form the commutator before projecting into the intertwiner basis
    comm = left @ right - right @ left
    projected = basis.conj().T @ comm @ basis
    projected[np.abs(projected) < 1.0e-13] = 0.0
    return channels, projected


def sqrt_abs_hermitian(matrix: Array) -> Array:
    hermitian = 0.5 * (matrix + matrix.conj().T)
    evals, evecs = np.linalg.eigh(hermitian)
    return (evecs * np.sqrt(np.abs(evals))) @ evecs.conj().T


def build_dense_volume_operators(
    *,
    two_j_max: int = 1,
    embedding_seed: int = 1,
    z_abs: float = 1.0,
) -> DenseVolumeOperators:

    if two_j_max != 1:
        raise ValueError("This experiment is intentionally restricted to two_j_max=1.")
    if z_abs < 0.0:
        raise ValueError("--z-abs must be non-negative.")

    graph = K5Graph(random_embedding=True, random_embedding_seed=embedding_seed)
    space = SU2Hilbert(graph, two_j_max=two_j_max, exact=True, max_exact_states=1_000_000)
    states = np.asarray(space.all_states(max_states=1_000_000), dtype=np.int32)
    basis_index = {tuple(row.tolist()): idx for idx, row in enumerate(states)}
    dimension = states.shape[0]

    q_commutators: dict[tuple[int, tuple[int, int, int]], Array] = {}
    q_hermitian: dict[tuple[int, tuple[int, int, int]], Array] = {}
    q_vertex_commutators: dict[int, Array] = {}
    q_vertex_hermitian: dict[int, Array] = {}
    vertex_volumes: dict[int, Array] = {}
    triple_signs: dict[int, dict[tuple[int, int, int], int]] = {}

    for vertex in range(space.n_vertices):
        vertex_id = space.layout.vertex_ids[vertex]
        incident_edges = tuple(space.metadata.incident_edges_by_vertex[vertex])
        incident_tokens = tuple(space.metadata.edge_tokens[edge] for edge in incident_edges)
        channel_sites = space.layout.vertex_channel_sites[vertex]
        channel_maxima = tuple(space.layout.vertex_channel_max_labels[vertex])
        if len(incident_edges) != 4 or len(channel_sites) != 1:
            raise ValueError("This experiment expects every K5 vertex to be 4-valent.")
        channel_site = channel_sites[0]
        q_sum = np.zeros((dimension, dimension), dtype=np.complex128)
        signs_for_vertex: dict[tuple[int, int, int], int] = {}

        for triple in combinations(range(4), 3):
            q_global = np.zeros((dimension, dimension), dtype=np.complex128)
            sign = orientation_sign(
                graph,
                vertex_id,
                tuple(incident_tokens[leg] for leg in triple),
            )
            signs_for_vertex[triple] = int(sign)

            for col, state in enumerate(states):
                two_js = tuple(int(state[edge]) for edge in incident_edges)
                channels, q_local = local_q_commutator(two_js, channel_maxima, triple)
                if q_local.size == 0:
                    continue
                source_channel = int(state[channel_site])
                try:
                    source_local = channels.index(source_channel)
                except ValueError as exc:
                    raise RuntimeError(
                        f"State channel {source_channel} not found in local channels {channels}."
                    ) from exc

                for target_local, target_channel in enumerate(channels):
                    amplitude = q_local[target_local, source_local]
                    if abs(amplitude) <= 1.0e-13:
                        continue
                    target = state.copy()
                    target[channel_site] = target_channel
                    row = basis_index[tuple(target.tolist())]
                    q_global[row, col] += amplitude

            q_global[np.abs(q_global) < 1.0e-13] = 0.0
            key = (vertex, triple)
            q_commutators[key] = q_global
            q_hermitian[key] = 1.0j * q_global
            q_sum = q_sum + sign * q_global

        q_sum[np.abs(q_sum) < 1.0e-13] = 0.0
        q_vertex_commutators[vertex] = q_sum
        q_h = 1.0j * q_sum
        q_h[np.abs(q_h) < 1.0e-13] = 0.0
        q_vertex_hermitian[vertex] = q_h
        vertex_volumes[vertex] = sqrt_abs_hermitian(z_abs * q_h)
        triple_signs[vertex] = signs_for_vertex

    total_volume = sum(vertex_volumes.values(), np.zeros((dimension, dimension), dtype=np.complex128))
    total_oriented_density = sum(
        q_vertex_hermitian.values(),
        np.zeros((dimension, dimension), dtype=np.complex128),
    )
    total_volume[np.abs(total_volume) < 1.0e-13] = 0.0
    total_oriented_density[np.abs(total_oriented_density) < 1.0e-13] = 0.0
    return DenseVolumeOperators(
        states=states,
        q_commutators=q_commutators,
        q_hermitian=q_hermitian,
        q_vertex_commutators=q_vertex_commutators,
        q_vertex_hermitian=q_vertex_hermitian,
        vertex_volumes=vertex_volumes,
        total_volume=total_volume,
        total_oriented_density=total_oriented_density,
        triple_signs=triple_signs,
    )


def trial_wavefunction(states: Array, n_edges: int) -> Array:
    weights = np.arange(1, states.shape[1] + 1, dtype=np.float64)
    reverse = weights[::-1]
    edge_sum = states[:, :n_edges].sum(axis=1).astype(np.float64)
    channel_sum = states[:, n_edges:].sum(axis=1).astype(np.float64)
    log_amp = 0.12 * (edge_sum - edge_sum.mean()) + 0.035 * (channel_sum - channel_sum.mean())
    phase = 0.11 * (states @ weights) + 0.07 * np.sin(states @ reverse)
    return np.exp(log_amp + 1.0j * phase)

