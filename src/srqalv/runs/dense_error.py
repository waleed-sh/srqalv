from __future__ import annotations

import csv
import time

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from srqalv._native import VolumeSRQ
from srqalv.plotting import configure_matplotlib
from srqalv.plotting import style_axes
from srqalv.srq import exact_volume_from_q
from srqalv.four_valent import four_valent_q_commutator


def assemble_srq_matrix(*, two_j: int, shifts: int, lambda_v: float | None, cg_tol: float, cg_maxiter: int) -> tuple[np.ndarray, float]:
    vol = VolumeSRQ(
        [int(two_j)] * 4,
        lambda_v=lambda_v,
        shifts=int(shifts),
        triple_signs={(0, 1, 2): 1},
        cg_tol=float(cg_tol),
        cg_maxiter=int(cg_maxiter),
    )
    out = np.empty((vol.dim, vol.dim), dtype=np.complex128)
    eye = np.eye(vol.dim, dtype=np.complex128)
    for col in range(vol.dim):
        # keep the shared Krylov path used by the paper's dense error check
        out[:, col] = vol(eye[:, col])
    return out, float(vol.lambda_v)


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["two_j", "dim", "M", "lambda_mode", "lambda_v", "max_abs_error", "frobenius_rel_error"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_errors(rows: list[dict[str, float | int | str]], *, two_j: int, out_dir: Path, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 5.35))
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.16, top=0.96)
    palette = {"auto_row_sum": "#a53860", "exact_rho": "#450920"}
    labels = {"auto_row_sum": r"SRQ, row-sum $\Lambda_v$", "exact_rho": r"SRQ, exact $\rho$"}
    for mode in ("auto_row_sum", "exact_rho"):
        vals = [row for row in rows if row["lambda_mode"] == mode]
        vals = sorted(vals, key=lambda row: int(row["M"]))
        xs = np.asarray([int(row["M"]) for row in vals], dtype=np.float64)
        ys = np.asarray([float(row["max_abs_error"]) for row in vals], dtype=np.float64)
        ax.plot(xs, ys, marker="o", linewidth=2.35, markersize=5.5, color=palette[mode], label=labels[mode])
    ax.set_yscale("log")
    ticks = sorted({int(row["M"]) for row in rows})
    display_ticks = [tick for tick in (5, 50, 100, 200, 300, 400, 500) if tick in ticks]
    ax.set_xlim(float(ticks[0]) - 12.0, float(ticks[-1]) + 18.0)
    ax.set_xticks(display_ticks)
    ax.set_xticklabels([str(tick) for tick in display_ticks])
    ax.set_xlabel(r"number of shifted systems $M$")
    ax.set_ylabel(r"$\max_{ab}|(V^{\rm SRQ}_{M}-V)_{ab}|$")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right", handlelength=1.8, handletextpad=0.55)
    path = out_dir / f"four_valent_twoj{two_j}_srq_dense_max_abs_error.png"
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    return path


def run(args) -> None:
    if args.two_j <= 2:
        raise ValueError("experiment.two_j must be larger than 2 for this error study")
    configure_matplotlib()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    m_values = tuple(sorted(set(int(m) for m in args.orders)))

    t0 = time.perf_counter()
    _channels, q_comm = four_valent_q_commutator(int(args.two_j))
    q = 1.0j * q_comm
    exact = exact_volume_from_q(q)
    eigs = np.linalg.eigvalsh(0.5 * (q + q.conj().T)).real
    rho = float(np.max(np.abs(eigs))) if eigs.size else 0.0
    exact_norm = max(float(np.linalg.norm(exact)), 1.0e-300)

    rows: list[dict[str, float | int | str]] = []
    matrices = {"Q": q, "V_exact": exact}
    for shifts in m_values:
        for mode, lambda_v in (("auto_row_sum", None), ("exact_rho", rho * (1.0 + 1.0e-12))):
            srq, resolved_lambda = assemble_srq_matrix(
                two_j=int(args.two_j),
                shifts=int(shifts),
                lambda_v=lambda_v,
                cg_tol=float(args.cg_tol),
                cg_maxiter=int(args.cg_maxiter),
            )
            diff = srq - exact
            matrices[f"V_srq_M{shifts}_{mode}"] = srq
            rows.append(
                {
                    "two_j": int(args.two_j),
                    "dim": int(exact.shape[0]),
                    "M": int(shifts),
                    "lambda_mode": mode,
                    "lambda_v": resolved_lambda,
                    "max_abs_error": float(np.max(np.abs(diff))) if diff.size else 0.0,
                    "frobenius_rel_error": float(np.linalg.norm(diff) / exact_norm),
                }
            )

    csv_path = args.out_dir / f"four_valent_twoj{args.two_j}_srq_dense_error.csv"
    np.savez_compressed(args.out_dir / "volume_matrices.npz", **matrices)
    write_csv(csv_path, rows)
    plot_path = plot_errors(rows, two_j=int(args.two_j), out_dir=args.out_dir, dpi=int(args.dpi))
    seconds = time.perf_counter() - t0
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")
    print(f"two_j={args.two_j}, dim={exact.shape[0]}, rho={rho:.6g}")
    for mode in ("auto_row_sum", "exact_rho"):
        mode_rows = [row for row in rows if row["lambda_mode"] == mode]
        best = min(float(row["max_abs_error"]) for row in mode_rows)
        last = [row for row in mode_rows if int(row["M"]) == max(m_values)][0]
        print(f"{mode}: lambda={float(last['lambda_v']):.6g}, best_max_abs_error={best:.3e}")
