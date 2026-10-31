from __future__ import annotations

from pathlib import Path
from typing import Any

from srqalv.io_utils import write_csv
from srqalv.k5_system import CutoffSystem
from srqalv.k5_system import build_cutoff_system
from srqalv.observables import aggregate_kernel_metrics
from srqalv.observables import exact_vertex_values
from srqalv.observables import lambda_choice_from_mode
from srqalv.observables import lambda_scan_choices
from srqalv.observables import srq_vertex_values
from srqalv.observables import weighted_lambda_ratio
from srqalv.plotting import plot_iteration_scan
from srqalv.plotting import plot_kernel_preservation
from srqalv.plotting import plot_lambda_scan


def run_vertex_publication_study(
    *,
    cutoffs: tuple[int, ...],
    orders: tuple[int, ...],
    iteration_lambda: str,
    lambda_scan_order: int,
    embedding_seed: int,
    out_dir: Path,
    dpi: int,
    m_xlabel: str = r"number of shifted systems $M$",
) -> dict[str, Path | list[Path]]:

    out_dir.mkdir(parents=True, exist_ok=True)
    iteration_choice = lambda_choice_from_mode(iteration_lambda)
    choices = lambda_scan_choices()
    systems: dict[int, CutoffSystem] = {}
    exact_by_cutoff: dict[int, dict[int, float]] = {}
    iteration_rows: list[dict[str, Any]] = []
    lambda_rows: list[dict[str, Any]] = []
    plot_paths: list[Path] = []

    for cutoff in cutoffs:
        system = build_cutoff_system(cutoff, embedding_seed=embedding_seed)
        systems[cutoff] = system
        exact_values = exact_vertex_values(system)
        exact_by_cutoff[cutoff] = exact_values

        for order in orders:
            values = srq_vertex_values(system, shifts=order, choice=iteration_choice)
            for vertex in range(system.space.n_vertices):
                iteration_rows.append(
                    {
                        "cutoff": cutoff,
                        "dimension": system.states.shape[0],
                        "vertex": vertex,
                        "M": order,
                        "lambda_key": iteration_choice.key,
                        "lambda_label": iteration_choice.label,
                        "exact_value": exact_values[vertex],
                        "srq_value": values[vertex],
                        "bias": values[vertex] - exact_values[vertex],
                        "abs_bias": abs(values[vertex] - exact_values[vertex]),
                    }
                )

        for choice in choices:
            values = srq_vertex_values(system, shifts=lambda_scan_order, choice=choice)
            for vertex in range(system.space.n_vertices):
                lambda_rows.append(
                    {
                        "cutoff": cutoff,
                        "dimension": system.states.shape[0],
                        "vertex": vertex,
                        "M": lambda_scan_order,
                        "lambda_key": choice.key,
                        "lambda_label": choice.label,
                        "lambda_family": choice.family,
                        "lambda_factor": choice.factor,
                        "weighted_lambda_over_rho": weighted_lambda_ratio(system, vertex=vertex, choice=choice),
                        "exact_value": exact_values[vertex],
                        "srq_value": values[vertex],
                        "bias": values[vertex] - exact_values[vertex],
                        "abs_bias": abs(values[vertex] - exact_values[vertex]),
                    }
                )

        plot_paths.append(
            plot_iteration_scan(
                cutoff=cutoff,
                exact_values=exact_values,
                rows=iteration_rows,
                orders=orders,
                out_dir=out_dir,
                dpi=dpi,
                xlabel=m_xlabel,
            )
        )
        plot_paths.append(
            plot_lambda_scan(
                cutoff=cutoff,
                exact_values=exact_values,
                rows=lambda_rows,
                choices=choices,
                out_dir=out_dir,
                dpi=dpi,
            )
        )

    iteration_csv = out_dir / "k5_vertex_volume_vs_M.csv"
    lambda_csv = out_dir / "k5_vertex_volume_vs_lambda.csv"
    write_csv(iteration_csv, iteration_rows)
    write_csv(lambda_csv, lambda_rows)
    return {
        "iteration_csv": iteration_csv,
        "lambda_csv": lambda_csv,
        "plots": plot_paths,
    }


def run_kernel_publication_study(
    *,
    cutoffs: tuple[int, ...],
    orders: tuple[int, ...],
    lambda_mode: str,
    embedding_seed: int,
    zero_rtol: float,
    zero_atol: float,
    plot_floor: float,
    out_dir: Path,
    dpi: int,
) -> dict[str, Path | list[Path]]:

    out_dir.mkdir(parents=True, exist_ok=True)
    choice = lambda_choice_from_mode(lambda_mode)
    rows: list[dict[str, Any]] = []
    plot_paths: list[Path] = []

    for cutoff in cutoffs:
        system = build_cutoff_system(cutoff, embedding_seed=embedding_seed)
        for order in orders:
            rows.append(
                aggregate_kernel_metrics(
                    system.sectors,
                    cutoff=cutoff,
                    shifts=order,
                    choice=choice,
                    zero_rtol=zero_rtol,
                    zero_atol=zero_atol,
                )
            )
        plot_paths.append(
            plot_kernel_preservation(
                rows,
                cutoff=cutoff,
                out_dir=out_dir,
                plot_floor=plot_floor,
                dpi=dpi,
            )
        )

    csv_path = out_dir / "k5_srq_kernel_preservation_metrics.csv"
    write_csv(csv_path, rows)
    return {"csv": csv_path, "plots": plot_paths}


