from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Any

import numpy as np

from srqalv.k5_system import Array
from srqalv.k5_system import CutoffSystem
from srqalv.k5_system import LocalSector
from srqalv.k5_system import apply_total
from srqalv.k5_system import expectation_from_action
from srqalv.srq import srq_sqrt_abs_values


@dataclass(frozen=True)
class LambdaChoice:

    key: str
    label: str
    family: str
    factor: float


def lambda_scan_choices() -> tuple[LambdaChoice, ...]:
    return (
        LambdaChoice("rho_1p00", r"$1.00\,\rho$", "rho", 1.00),
        LambdaChoice("rho_1p05", r"$1.05\,\rho$", "rho", 1.05),
        LambdaChoice("rho_1p25", r"$1.25\,\rho$", "rho", 1.25),
        LambdaChoice("rho_1p50", r"$1.50\,\rho$", "rho", 1.50),
        LambdaChoice("rho_2p00", r"$2.00\,\rho$", "rho", 2.00),
        LambdaChoice("analytic_1p00", r"$\Lambda_{\rm an}$", "analytic", 1.00),
        LambdaChoice("analytic_1p25", r"$1.25\,\Lambda_{\rm an}$", "analytic", 1.25),
        LambdaChoice("analytic_2p00", r"$2.00\,\Lambda_{\rm an}$", "analytic", 2.00),
    )


def lambda_for_choice(sector: LocalSector, choice: LambdaChoice) -> float:
    if sector.rho <= 1.0e-13:
        return 0.0
    if choice.family == "rho":
        return sector.rho * choice.factor * (1.0 + 1.0e-12)
    if choice.family == "analytic":
        return sector.analytic_lambda * choice.factor
    raise ValueError(f"Unknown Lambda family {choice.family!r}.")


def lambda_choice_from_mode(mode: str) -> LambdaChoice:
    if mode == "analytic":
        return LambdaChoice("analytic", r"$\Lambda_{\rm an}$", "analytic", 1.0)
    if mode == "rho":
        return LambdaChoice("rho", r"$\rho$", "rho", 1.0)
    raise ValueError("mode must be 'analytic' or 'rho'.")


def srq_matrix_for_sector_choice(sector: LocalSector, *, shifts: int, choice: LambdaChoice) -> Array:
    if np.linalg.norm(sector.q) <= 1.0e-13:
        return np.zeros_like(sector.q)

    lambda_bound = lambda_for_choice(sector, choice)
    if lambda_bound <= 1.0e-14:
        raise RuntimeError(
            f"Zero Lambda for nonzero sector at cutoff={sector.cutoff}, "
            f"vertex={sector.vertex}, spins={sector.two_js}, choice={choice.key}."
        )
    if lambda_bound < sector.rho * (1.0 - 1.0e-10):
        raise RuntimeError(
            f"Unsafe Lambda underbounds rho: Lambda={lambda_bound:.16e}, "
            f"rho={sector.rho:.16e}, choice={choice.key}."
        )

    qh = 0.5 * (sector.q + sector.q.conj().T)
    evals, evecs = np.linalg.eigh(qh)
    values = srq_sqrt_abs_values(evals.real, lambda_bound=lambda_bound, shifts=shifts)
    matrix = (evecs * values) @ evecs.conj().T
    matrix = 0.5 * (matrix + matrix.conj().T)
    matrix[np.abs(matrix) < 1.0e-13] = 0.0
    return matrix


def exact_vertex_values(system: CutoffSystem) -> dict[int, float]:
    values: dict[int, float] = {}
    for vertex in range(system.space.n_vertices):
        sectors = [sector for sector in system.sectors if sector.vertex == vertex]
        action = apply_total(sectors, system.psi, matrix_getter=lambda sector: sector.v_exact)
        values[vertex] = float(expectation_from_action(system.psi, action).real)
    return values


def srq_vertex_values(system: CutoffSystem, *, shifts: int, choice: LambdaChoice) -> dict[int, float]:
    matrices = {
        id(sector): srq_matrix_for_sector_choice(sector, shifts=shifts, choice=choice)
        for sector in system.sectors
    }
    values: dict[int, float] = {}
    for vertex in range(system.space.n_vertices):
        sectors = [sector for sector in system.sectors if sector.vertex == vertex]
        action = apply_total(sectors, system.psi, matrix_getter=lambda sector: matrices[id(sector)])
        values[vertex] = float(expectation_from_action(system.psi, action).real)
    return values


def weighted_lambda_ratio(system: CutoffSystem, *, vertex: int, choice: LambdaChoice) -> float:
    weights = []
    ratios = []
    probabilities = system.probabilities / np.sum(system.probabilities)
    for sector in system.sectors:
        if sector.vertex != vertex or sector.rho <= 1.0e-13:
            continue
        weight = float(np.sum(probabilities[sector.cols]))
        if weight <= 0.0:
            continue
        weights.append(weight)
        ratios.append(lambda_for_choice(sector, choice) / sector.rho)
    if not weights:
        return math.nan
    return float(np.average(np.asarray(ratios), weights=np.asarray(weights)))


def sector_repetition(sector: LocalSector) -> int:
    if not sector.channels:
        return 0
    if sector.cols.size % len(sector.channels) != 0:
        return 1
    return int(sector.cols.size // len(sector.channels))


def sector_kernel_metrics(
    sector: LocalSector,
    *,
    shifts: int,
    choice: LambdaChoice,
    zero_rtol: float,
    zero_atol: float,
) -> dict[str, float | int]:

    qh = 0.5 * (sector.q + sector.q.conj().T)
    eigs = np.linalg.eigvalsh(qh).real if qh.size else np.zeros((0,), dtype=np.float64)
    rho = float(np.max(np.abs(eigs))) if eigs.size else 0.0
    zero_tol = max(float(zero_atol), float(zero_rtol) * max(rho, sector.rho))

    # test the exact kernel after removing eigensolver roundoff at zero
    snapped = eigs.copy()
    snapped[np.abs(snapped) <= zero_tol] = 0.0
    kernel = np.abs(snapped) <= 0.0

    lambda_bound = lambda_for_choice(sector, choice)
    approx = (
        srq_sqrt_abs_values(snapped, lambda_bound=lambda_bound, shifts=shifts)
        if lambda_bound > 0.0
        else np.zeros_like(snapped)
    )
    exact = np.sqrt(np.abs(snapped))
    nonkernel = ~kernel
    repeat = sector_repetition(sector)

    kernel_lift = float(np.max(np.abs(approx[kernel]))) if np.any(kernel) else 0.0
    nonkernel_abs_error = (
        float(np.max(np.abs(approx[nonkernel] - exact[nonkernel]))) if np.any(nonkernel) else 0.0
    )
    nonkernel_rel_error = 0.0
    if np.any(nonkernel):
        denom = np.maximum(exact[nonkernel], 1.0e-300)
        nonkernel_rel_error = float(np.max(np.abs(approx[nonkernel] - exact[nonkernel]) / denom))

    return {
        "kernel_lift": kernel_lift,
        "nonkernel_abs_error": nonkernel_abs_error,
        "nonkernel_rel_error": nonkernel_rel_error,
        "local_kernel_modes": int(np.count_nonzero(kernel)),
        "local_nonkernel_modes": int(np.count_nonzero(nonkernel)),
        "full_kernel_modes": int(repeat * np.count_nonzero(kernel)),
        "full_nonkernel_modes": int(repeat * np.count_nonzero(nonkernel)),
    }


def aggregate_kernel_metrics(
    sectors: list[LocalSector],
    *,
    cutoff: int,
    shifts: int,
    choice: LambdaChoice,
    zero_rtol: float = 1.0e-12,
    zero_atol: float = 1.0e-14,
) -> dict[str, Any]:

    totals = {
        "kernel_lift": 0.0,
        "nonkernel_abs_error": 0.0,
        "nonkernel_rel_error": 0.0,
        "local_kernel_modes": 0,
        "local_nonkernel_modes": 0,
        "full_kernel_modes": 0,
        "full_nonkernel_modes": 0,
    }
    for sector in sectors:
        metrics = sector_kernel_metrics(
            sector,
            shifts=shifts,
            choice=choice,
            zero_rtol=zero_rtol,
            zero_atol=zero_atol,
        )
        totals["kernel_lift"] = max(totals["kernel_lift"], float(metrics["kernel_lift"]))
        totals["nonkernel_abs_error"] = max(
            totals["nonkernel_abs_error"],
            float(metrics["nonkernel_abs_error"]),
        )
        totals["nonkernel_rel_error"] = max(
            totals["nonkernel_rel_error"],
            float(metrics["nonkernel_rel_error"]),
        )
        for key in ("local_kernel_modes", "local_nonkernel_modes", "full_kernel_modes", "full_nonkernel_modes"):
            totals[key] += int(metrics[key])

    return {
        "cutoff": cutoff,
        "M": shifts,
        "lambda_key": choice.key,
        "max_kernel_lift": totals["kernel_lift"],
        "max_nonkernel_abs_error": totals["nonkernel_abs_error"],
        "max_nonkernel_rel_error": totals["nonkernel_rel_error"],
        "local_kernel_modes": totals["local_kernel_modes"],
        "local_nonkernel_modes": totals["local_nonkernel_modes"],
        "full_kernel_modes": totals["full_kernel_modes"],
        "full_nonkernel_modes": totals["full_nonkernel_modes"],
    }
