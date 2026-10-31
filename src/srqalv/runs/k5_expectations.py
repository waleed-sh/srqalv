from __future__ import annotations


from pathlib import Path

from srqalv.experiments import run_vertex_publication_study


def run(args) -> None:
    print("Generating K5 SRQ vertex-volume publication figures")
    outputs = run_vertex_publication_study(
        cutoffs=args.cutoffs,
        orders=args.orders,
        iteration_lambda=args.iteration_lambda,
        lambda_scan_order=args.lambda_scan_order,
        embedding_seed=args.embedding_seed,
        out_dir=args.out_dir,
        dpi=args.dpi,
        m_xlabel=args.m_label,
    )
    print(f"Wrote {outputs['iteration_csv']}")
    print(f"Wrote {outputs['lambda_csv']}")
    for plot_path in outputs["plots"]:
        print(f"Wrote {plot_path}")
