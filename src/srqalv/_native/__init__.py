from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from ._core import _ApplyQ
from ._core import _VolumeSRQ
from ._core import enumerate_left_comb_channels
from ._core import mpi_enabled
from ._core import mpi_finalize
from ._core import mpi_finalized
from ._core import mpi_init_thread
from ._core import mpi_initialized
from ._core import mpi_world_rank
from ._core import mpi_world_size
from ._core import openmp_enabled
from ._core import openmp_max_threads
from ._core import openmp_set_num_threads
from ._core import parse_spin_list
from ._core import parse_triple_signs

__all__ = [
    "ApplyQ",
    "VolumeSRQ",
    "apply_q",
    "estimate_lambda",
    "volume_srq",
    "enumerate_left_comb_channels",
    "mpi_enabled",
    "mpi_finalize",
    "mpi_finalized",
    "mpi_init_thread",
    "mpi_initialized",
    "mpi_world_rank",
    "mpi_world_size",
    "openmp_enabled",
    "openmp_max_threads",
    "openmp_set_num_threads",
    "parse_spin_list",
    "parse_triple_signs",
]


TripleKey = tuple[int, int, int]
TripleEntry = tuple[int, int, int, int] | tuple[TripleKey, int]


def _as_int_list(values: Sequence[int], *, name: str) -> list[int]:
    try:
        return [int(value) for value in values]
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of integers") from exc


def _normalise_channels(channels: Sequence[Sequence[int]] | None) -> list[list[int]]:
    if channels is None:
        return []
    return [_as_int_list(channel, name="channel") for channel in channels]


def _normalise_triple_signs(triple_signs: str | Mapping[Any, int] | Iterable[TripleEntry] | None) -> str:
    if triple_signs is None:
        return "all"
    if isinstance(triple_signs, str):
        return triple_signs

    if isinstance(triple_signs, Mapping):
        entries = triple_signs.items()
    else:
        entries = triple_signs

    parts: list[str] = []
    for entry in entries:
        try:
            length = len(entry)
        except TypeError as exc:
            raise TypeError(
                "triple_signs entries must be (i, j, k, sign) or ((i, j, k), sign)"
            ) from exc

        if length == 4:
            i, j, k, sign = entry
        elif length == 2:
            triple, sign = entry
            if len(triple) != 3:
                raise ValueError("triple key must have length three")
            i, j, k = triple
        else:
            raise ValueError(
                "triple_signs entries must be (i, j, k, sign) or ((i, j, k), sign)"
            )
        parts.append(f"{int(i)},{int(j)},{int(k)}:{int(sign)}")

    return ";".join(parts) if parts else "none"


class ApplyQ:

    def __init__(
        self,
        two_js: Sequence[int],
        triple_signs: str | Mapping[Any, int] | Iterable[TripleEntry] | None = "all",
        channels: Sequence[Sequence[int]] | None = None,
    ) -> None:
        self._op = _ApplyQ(
            _as_int_list(two_js, name="two_js"),
            _normalise_triple_signs(triple_signs),
            _normalise_channels(channels),
        )

    @property
    def dim(self) -> int:
        return int(self._op.dim)

    @property
    def valence(self) -> int:
        return int(self._op.valence)

    @property
    def two_js(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self._op.two_js)

    @property
    def channels(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(int(value) for value in channel) for channel in self._op.channels)

    @property
    def sixj_cache_size(self) -> int:
        return int(self._op.sixj_cache_size)

    def __call__(self, vector: Any) -> np.ndarray:
        array = np.ascontiguousarray(vector, dtype=np.complex128)
        if array.ndim != 1:
            raise ValueError("ApplyQ expects a one-dimensional vector")
        return self._op(array)

    def apply(self, vector: Any) -> np.ndarray:
        return self(vector)

    def apply_into(self, vector: Any, output: np.ndarray) -> np.ndarray:
        input_array = np.ascontiguousarray(vector, dtype=np.complex128)
        output_array = np.asarray(output)
        if input_array.ndim != 1 or output_array.ndim != 1:
            raise ValueError("ApplyQ.apply_into expects one-dimensional arrays")
        if output_array.dtype != np.complex128:
            raise TypeError("output must have dtype numpy.complex128")
        if not output_array.flags.c_contiguous:
            raise ValueError("output must be C-contiguous")
        if not output_array.flags.writeable:
            raise ValueError("output must be writable")
        self._op.apply_into(input_array, output_array)
        return output

    def apply_mpi(self, vector: Any) -> np.ndarray:
        array = np.ascontiguousarray(vector, dtype=np.complex128)
        if array.ndim != 1:
            raise ValueError("ApplyQ.apply_mpi expects a one-dimensional vector")
        return self._op.apply_mpi(array)

    def apply_mpi_into(self, vector: Any, output: np.ndarray) -> np.ndarray:
        input_array = np.ascontiguousarray(vector, dtype=np.complex128)
        output_array = np.asarray(output)
        if input_array.ndim != 1 or output_array.ndim != 1:
            raise ValueError("ApplyQ.apply_mpi_into expects one-dimensional arrays")
        if output_array.dtype != np.complex128:
            raise TypeError("output must have dtype numpy.complex128")
        if not output_array.flags.c_contiguous:
            raise ValueError("output must be C-contiguous")
        if not output_array.flags.writeable:
            raise ValueError("output must be writable")
        self._op.apply_mpi_into(input_array, output_array)
        return output

    def matrix_element(self, bra: Sequence[int], ket: Sequence[int], i: int, j: int, k: int) -> float:
        return float(
            self._op.matrix_element(
                _as_int_list(bra, name="bra"),
                _as_int_list(ket, name="ket"),
                int(i),
                int(j),
                int(k),
            )
        )

    def row_sum_bound(self) -> float:
        return float(self._op.row_sum_bound())

    def __repr__(self) -> str:
        return f"ApplyQ(two_js={self.two_js!r}, dim={self.dim}, valence={self.valence})"


def apply_q(
    two_js: Sequence[int],
    vector: Any,
    triple_signs: str | Mapping[Any, int] | Iterable[TripleEntry] | None = "all",
    channels: Sequence[Sequence[int]] | None = None,
) -> np.ndarray:

    return ApplyQ(two_js, triple_signs=triple_signs, channels=channels)(vector)


def estimate_lambda(
    two_js: Sequence[int],
    triple_signs: str | Mapping[Any, int] | Iterable[TripleEntry] | None = "all",
    channels: Sequence[Sequence[int]] | None = None,
    *,
    method: str = "row_sum",
    safety: float | None = None,
) -> float:

    if method != "row_sum":
        raise ValueError("only method='row_sum' is currently implemented")
    bound = ApplyQ(two_js, triple_signs=triple_signs, channels=channels).row_sum_bound()
    if bound <= 0.0:
        return 0.0
    factor = 1.0 + 64.0 * np.finfo(np.float64).eps if safety is None else float(safety)
    if factor < 1.0:
        raise ValueError("safety must be at least 1")
    return float(bound * factor)


class VolumeSRQ:

    def __init__(
        self,
        two_js: Sequence[int],
        lambda_v: float | None = None,
        shifts: int | None = None,
        triple_signs: str | Mapping[Any, int] | Iterable[TripleEntry] | None = "all",
        channels: Sequence[Sequence[int]] | None = None,
        cg_tol: float = 1.0e-12,
        cg_maxiter: int = 0,
    ) -> None:
        if shifts is None:
            raise TypeError("VolumeSRQ requires shifts=...; lambda_v may be omitted for automatic row-sum selection")
        self._lambda_source = "auto_row_sum" if lambda_v is None else "explicit"
        self._op = _VolumeSRQ(
            _as_int_list(two_js, name="two_js"),
            _normalise_triple_signs(triple_signs),
            _normalise_channels(channels),
            -1.0 if lambda_v is None else float(lambda_v),
            int(shifts),
            float(cg_tol),
            int(cg_maxiter),
        )

    @property
    def dim(self) -> int:
        return int(self._op.dim)

    @property
    def valence(self) -> int:
        return int(self._op.valence)

    @property
    def two_js(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self._op.two_js)

    @property
    def channels(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(int(value) for value in channel) for channel in self._op.channels)

    @property
    def lambda_v(self) -> float:
        return float(self._op.lambda_bound)

    @property
    def lambda_bound(self) -> float:
        return self.lambda_v

    @property
    def lambda_source(self) -> str:
        return self._lambda_source

    @property
    def shifts(self) -> int:
        return int(self._op.shifts)

    @property
    def cg_tol(self) -> float:
        return float(self._op.cg_tol)

    @property
    def cg_maxiter(self) -> int:
        return int(self._op.cg_maxiter)

    def diagnostics(self) -> dict[str, float | int | bool]:
        return dict(self._op.diagnostics())

    def __call__(self, vector: Any) -> np.ndarray:
        array = np.ascontiguousarray(vector, dtype=np.complex128)
        if array.ndim != 1:
            raise ValueError("VolumeSRQ expects a one-dimensional vector")
        return self._op(array)

    def apply(self, vector: Any) -> np.ndarray:
        return self(vector)

    def apply_into(self, vector: Any, output: np.ndarray) -> np.ndarray:
        input_array = np.ascontiguousarray(vector, dtype=np.complex128)
        output_array = np.asarray(output)
        if input_array.ndim != 1 or output_array.ndim != 1:
            raise ValueError("VolumeSRQ.apply_into expects one-dimensional arrays")
        if output_array.dtype != np.complex128:
            raise TypeError("output must have dtype numpy.complex128")
        if not output_array.flags.c_contiguous:
            raise ValueError("output must be C-contiguous")
        if not output_array.flags.writeable:
            raise ValueError("output must be writable")
        self._op.apply_into(input_array, output_array)
        return output

    def apply_mpi(self, vector: Any) -> np.ndarray:
        array = np.ascontiguousarray(vector, dtype=np.complex128)
        if array.ndim != 1:
            raise ValueError("VolumeSRQ.apply_mpi expects a one-dimensional vector")
        return self._op.apply_mpi(array)

    def apply_mpi_into(self, vector: Any, output: np.ndarray) -> np.ndarray:
        input_array = np.ascontiguousarray(vector, dtype=np.complex128)
        output_array = np.asarray(output)
        if input_array.ndim != 1 or output_array.ndim != 1:
            raise ValueError("VolumeSRQ.apply_mpi_into expects one-dimensional arrays")
        if output_array.dtype != np.complex128:
            raise TypeError("output must have dtype numpy.complex128")
        if not output_array.flags.c_contiguous:
            raise ValueError("output must be C-contiguous")
        if not output_array.flags.writeable:
            raise ValueError("output must be writable")
        self._op.apply_mpi_into(input_array, output_array)
        return output

    def __repr__(self) -> str:
        return (
            "VolumeSRQ("
            f"two_js={self.two_js!r}, dim={self.dim}, valence={self.valence}, "
            f"lambda_v={self.lambda_v:.6g}, lambda_source={self.lambda_source!r}, "
            f"shifts={self.shifts})"
        )


def volume_srq(
    two_js: Sequence[int],
    vector: Any,
    lambda_v: float | None = None,
    shifts: int | None = None,
    triple_signs: str | Mapping[Any, int] | Iterable[TripleEntry] | None = "all",
    channels: Sequence[Sequence[int]] | None = None,
    cg_tol: float = 1.0e-12,
    cg_maxiter: int = 0,
) -> np.ndarray:

    return VolumeSRQ(
        two_js,
        lambda_v=lambda_v,
        shifts=shifts,
        triple_signs=triple_signs,
        channels=channels,
        cg_tol=cg_tol,
        cg_maxiter=cg_maxiter,
    )(vector)
