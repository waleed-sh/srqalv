from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from itertools import product

import numpy as np

from srqalv.basis import all_left_comb_channels

from srqalv.baseline import clebsch_gordan
from srqalv.baseline import m_values
from srqalv.baseline import spin_generators

Array = np.ndarray


def default_channel_maxima(two_js: tuple[int, ...]) -> tuple[int, ...]:
    spins = tuple(int(x) for x in two_js)
    return tuple(sum(spins[: idx + 2]) for idx in range(max(len(spins) - 3, 0)))


def enumerate_left_comb_channels(two_js: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return all_left_comb_channels(tuple(int(x) for x in two_js), default_channel_maxima(tuple(int(x) for x in two_js)))


def _magnetic_labels(two_js: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(product(*(m_values(two_j) for two_j in two_js)))


def left_comb_basis_general(
    two_js: tuple[int, ...],
    channel_tuples: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[tuple[tuple[int, ...], ...], Array]:

    spins = tuple(int(x) for x in two_js)
    valence = len(spins)
    if valence < 2:
        raise ValueError("A gauge-invariant vertex must have at least two legs.")
    channels_all = tuple(enumerate_left_comb_channels(spins) if channel_tuples is None else channel_tuples)
    magnetic_labels = _magnetic_labels(spins)
    magnetic_index = {label: pos for pos, label in enumerate(magnetic_labels)}
    vectors: list[Array] = []

    for channels in channels_all:
        channels = tuple(int(x) for x in channels)
        if len(channels) != max(valence - 3, 0):
            raise ValueError(f"Invalid channel count for valence {valence}: {channels!r}.")
        vec = np.zeros((len(magnetic_labels),), dtype=np.complex128)
        for magnetic in magnetic_labels:
            if sum(magnetic) != 0:
                continue

            if valence == 2:
                coeff = clebsch_gordan(spins[0], magnetic[0], spins[1], magnetic[1], 0, 0)
            elif valence == 3:
                two_a = spins[2]
                two_ma = magnetic[0] + magnetic[1]
                coeff = clebsch_gordan(spins[0], magnetic[0], spins[1], magnetic[1], two_a, two_ma)
                coeff *= clebsch_gordan(two_a, two_ma, spins[2], magnetic[2], 0, 0)
            else:
                two_a = channels[0]
                two_ma = magnetic[0] + magnetic[1]
                coeff = clebsch_gordan(spins[0], magnetic[0], spins[1], magnetic[1], two_a, two_ma)
                previous_j = two_a
                previous_m = two_ma
                for channel_index in range(1, len(channels)):
                    edge_index = channel_index + 1
                    target_j = channels[channel_index]
                    target_m = previous_m + magnetic[edge_index]
                    coeff *= clebsch_gordan(
                        previous_j,
                        previous_m,
                        spins[edge_index],
                        magnetic[edge_index],
                        target_j,
                        target_m,
                    )
                    previous_j = target_j
                    previous_m = target_m

                penultimate_edge = valence - 2
                final_j = spins[-1]
                final_m = previous_m + magnetic[penultimate_edge]
                coeff *= clebsch_gordan(
                    previous_j,
                    previous_m,
                    spins[penultimate_edge],
                    magnetic[penultimate_edge],
                    final_j,
                    final_m,
                )
                coeff *= clebsch_gordan(final_j, final_m, spins[-1], magnetic[-1], 0, 0)

            if coeff:
                vec[magnetic_index[magnetic]] += coeff

        norm = np.linalg.norm(vec)
        if norm <= 1.0e-13:
            raise RuntimeError(f"Built zero intertwiner vector for {spins=} and {channels=}.")
        vectors.append(vec / norm)

    if not vectors:
        return channels_all, np.zeros((len(magnetic_labels), 0), dtype=np.complex128)

    basis = np.column_stack(vectors)
    gram_error = np.linalg.norm(basis.conj().T @ basis - np.eye(len(channels_all)))
    if gram_error > 1.0e-10:
        raise RuntimeError(f"Left-comb basis is not orthonormal; error={gram_error:.3e}.")
    return channels_all, basis


def _apply_axis(arr: Array, op: Array, leg: int) -> Array:
    axis = int(leg) + 1
    moved = np.tensordot(np.asarray(op, dtype=np.complex128), arr, axes=([1], [axis]))
    return np.moveaxis(moved, 0, axis)


def apply_pair_casimir_magnetic(two_js: tuple[int, ...], first: int, second: int, vectors: Array) -> Array:
    spins = tuple(int(x) for x in two_js)
    dims = tuple(two_j + 1 for two_j in spins)
    batch = np.asarray(vectors, dtype=np.complex128).reshape((-1, *dims))
    j_first = spins[int(first)] / 2.0
    j_second = spins[int(second)] / 2.0
    out = (j_first * (j_first + 1.0) + j_second * (j_second + 1.0)) * batch
    generators_first = spin_generators(spins[int(first)])
    generators_second = spin_generators(spins[int(second)])
    for component in range(3):
        tmp = _apply_axis(batch, generators_second[component], int(second))
        tmp = _apply_axis(tmp, generators_first[component], int(first))
        out = out + 2.0 * tmp
    return out.reshape((batch.shape[0], -1))


def apply_q_commutator_magnetic(
    two_js: tuple[int, ...],
    triple: tuple[int, int, int],
    vectors: Array,
) -> Array:

    first, second, third = (int(x) for x in triple)
    left_then_right = apply_pair_casimir_magnetic(
        two_js,
        first,
        second,
        apply_pair_casimir_magnetic(two_js, second, third, vectors),
    )
    right_then_left = apply_pair_casimir_magnetic(
        two_js,
        second,
        third,
        apply_pair_casimir_magnetic(two_js, first, second, vectors),
    )
    return left_then_right - right_then_left


@dataclass(frozen=True)
class GeneralApplyQ:

    two_js: tuple[int, ...]
    channel_tuples: tuple[tuple[int, ...], ...]
    triple_signs: dict[tuple[int, int, int], int]
    basis: Array

    @classmethod
    def create(
        cls,
        two_js: tuple[int, ...],
        *,
        channel_tuples: tuple[tuple[int, ...], ...] | None = None,
        triple_signs: dict[tuple[int, int, int], int] | None = None,
    ) -> "GeneralApplyQ":

        spins = tuple(int(x) for x in two_js)
        channels, basis = left_comb_basis_general(spins, channel_tuples)
        if triple_signs is None:
            signs = {triple: 1 for triple in combinations(range(len(spins)), 3)}
        else:
            signs = {tuple(sorted(tuple(int(x) for x in triple))): int(sign) for triple, sign in triple_signs.items()}
        return cls(two_js=spins, channel_tuples=channels, triple_signs=signs, basis=basis)

    @property
    def dim(self) -> int:
        return len(self.channel_tuples)

    @property
    def magnetic_dim(self) -> int:
        return int(self.basis.shape[0])

    def apply_q(self, vectors: Array) -> Array:
        batch = np.asarray(vectors, dtype=np.complex128)
        if batch.ndim == 1:
            batch = batch[None, :]
        if batch.shape[-1] != self.dim:
            raise ValueError(f"Expected vectors with final dimension {self.dim}, got {batch.shape[-1]}.")
        magnetic = batch @ self.basis.T
        out_magnetic = np.zeros_like(magnetic)
        for triple, sign in sorted(self.triple_signs.items()):
            if sign == 0:
                continue
            out_magnetic += (1.0j * int(sign)) * apply_q_commutator_magnetic(self.two_js, triple, magnetic)
        projected = out_magnetic @ self.basis.conj()
        projected[np.abs(projected) < 1.0e-13] = 0.0
        return projected.reshape(batch.shape)

    def dense_q(self) -> Array:
        return self.apply_q(np.eye(self.dim, dtype=np.complex128)).T


def complete_triple_signs(
    valence: int,
    *,
    default: int = 1,
) -> dict[tuple[int, int, int], int]:

    return {triple: int(default) for triple in combinations(range(int(valence)), 3)}
