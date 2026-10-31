from __future__ import annotations


from pathlib import Path

from srqalv.experiments import run_kernel_publication_study


def run(args) -> None:
    print("Generating K5 SRQ kernel-preservation figures")
    outputs = run_kernel_publication_study(
        cutoffs=args.cutoffs,
        orders=args.orders,
        lambda_mode=args.lambda_mode,
        embedding_seed=args.embedding_seed,
        zero_rtol=args.zero_rtol,
        zero_atol=args.zero_atol,
        plot_floor=args.plot_floor,
        out_dir=args.out_dir,
        dpi=args.dpi,
    )
    print(f"Wrote {outputs['csv']}")
    for plot_path in outputs["plots"]:
        print(f"Wrote {plot_path}")
