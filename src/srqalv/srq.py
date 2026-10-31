from __future__ import annotations

import math

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class SRQParameters:

    shifts: int
    kminus: int
    kplus: int
    h: float


def srq_parameters(shifts: int) -> SRQParameters:
    if shifts <= 0:
        raise ValueError("shifts must be positive.")
    # balance the two tails for alpha = 1/4
    kminus = math.ceil(3.0 * (int(shifts) - 1) / 4.0)
    kplus = int(shifts) - 1 - kminus
    h = 4.0 * math.pi / math.sqrt(3.0 * int(shifts))
    return SRQParameters(shifts=int(shifts), kminus=kminus, kplus=kplus, h=h)


def srq_sqrt_abs_values(q_eigs: Array, *, lambda_bound: float, shifts: int) -> Array:
    eigs = np.asarray(q_eigs, dtype=np.float64)
    if lambda_bound <= 0.0:
        return np.zeros_like(eigs)

    mu = np.square(eigs / float(lambda_bound))
    if mu.size and float(np.max(mu)) > 1.0 + 1.0e-10:
        raise ValueError(
            "Unsafe SRQ scale: lambda_bound is smaller than the spectral radius."
        )

    params = srq_parameters(shifts)
    acc = np.zeros_like(mu, dtype=np.float64)
    for k in range(-params.kminus, params.kplus + 1):
        kh = k * params.h
        tau = math.exp(kh)
        acc += math.exp(kh / 4.0) * mu / (mu + tau)

    return math.sqrt(lambda_bound) * params.h / (math.sqrt(2.0) * math.pi) * acc


def srq_dense_matrix(q: Array, *, lambda_bound: float, shifts: int) -> Array:
    qh = np.asarray(q, dtype=np.complex128)
    if qh.ndim != 2 or qh.shape[0] != qh.shape[1]:
        raise ValueError("q must be a square matrix.")
    if lambda_bound <= 0.0:
        return np.zeros_like(qh)

    qh = 0.5 * (qh + qh.conj().T)
    abar = (qh @ qh) / float(lambda_bound * lambda_bound)
    identity = np.eye(qh.shape[0], dtype=np.complex128)
    params = srq_parameters(shifts)

    acc = np.zeros_like(qh)
    for k in range(-params.kminus, params.kplus + 1):
        kh = k * params.h
        tau = math.exp(kh)
        acc += math.exp(kh / 4.0) * np.linalg.solve(abar + tau * identity, abar)

    out = math.sqrt(lambda_bound) * params.h / (math.sqrt(2.0) * math.pi) * acc
    out = 0.5 * (out + out.conj().T)
    out[np.abs(out) < 1.0e-13] = 0.0
    return out


def exact_volume_from_q(q: Array) -> Array:
    qh = 0.5 * (np.asarray(q, dtype=np.complex128) + np.asarray(q, dtype=np.complex128).conj().T)
    evals, evecs = np.linalg.eigh(qh)
    values = np.sqrt(np.abs(evals.real))
    out = (evecs * values) @ evecs.conj().T
    out = 0.5 * (out + out.conj().T)
    out[np.abs(out) < 1.0e-13] = 0.0
    return out
