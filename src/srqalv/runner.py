from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

from hydra import compose
from omegaconf import DictConfig, OmegaConf


RUNNERS = {"k5_expectations", "kernel", "dense_error", "mc_samples", "mc_orders", "slq"}


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def validate(experiment, dpi):
    positive_int(dpi, "dpi")
    for key in ("orders", "cutoffs", "sample_counts", "probe_values"):
        if key in experiment:
            values = experiment[key]
            if not isinstance(values, list) or not values:
                raise ValueError(f"experiment.{key} must be a nonempty list")
            for value in values:
                positive_int(value, f"experiment.{key}")
    for key in ("cutoff", "two_j", "order", "sample_count", "chains", "lanczos_steps",
                "probe_batch_size", "grid_points", "exact_histogram_bins", "lambda_scan_order"):
        if key in experiment:
            positive_int(experiment[key], f"experiment.{key}")
    for key, size in (("two_js", 5), ("fixed_edge_labels", 10)):
        if key in experiment:
            values = experiment[key]
            if not isinstance(values, list) or len(values) != size:
                raise ValueError(f"experiment.{key} must contain {size} doubled-spin labels")
            if any(type(v) is not int or v < 0 for v in values):
                raise ValueError(f"experiment.{key} must contain nonnegative integers")
    for key in ("iteration_lambda", "lambda_mode"):
        if key in experiment and experiment[key] not in {"analytic", "rho"}:
            raise ValueError(f"experiment.{key} must be analytic or rho")
    if "cg_tol" in experiment and not 0 < experiment["cg_tol"] < 1:
        raise ValueError("experiment.cg_tol must lie between zero and one")


def run_one(cfg: DictConfig, output: Path) -> None:
    settings = OmegaConf.to_container(cfg.experiment, resolve=True)
    validate(settings, cfg.dpi)
    kind = settings.pop("kind")
    name = settings.pop("name")
    dataset = settings.pop("dataset")
    if kind not in RUNNERS:
        raise ValueError(f"Unknown experiment kind: {kind}")
    for key, value in (("name", name), ("dataset", dataset)):
        if not value or value in {".", ".."} or Path(value).name != value:
            raise ValueError(f"experiment.{key} must be a directory name")
    data_dir = output / "data" / dataset
    data_dir.mkdir(parents=True, exist_ok=True)
    settings.update(out_dir=data_dir, dpi=cfg.dpi)
    for key, value in settings.items():
        if isinstance(value, list):
            settings[key] = tuple(value)

    config_dir = output / "configs"
    config_dir.mkdir(exist_ok=True)
    OmegaConf.save(cfg, config_dir / f"{name}.yaml", resolve=True)
    print(f"Running {name}", flush=True)
    import_module(f"srqalv.runs.{kind}").run(SimpleNamespace(**settings))

    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    for path in data_dir.glob("*.png"):
        path.replace(figures / path.name)


def run(cfg: DictConfig, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.merge(cfg, {"output_dir": str(output)})
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    if cfg.experiment.name == "all":
        # compose each preset afresh so one study cannot change the next one's settings
        for name in cfg.experiment.experiments:
            if name == "all":
                raise ValueError("The all preset cannot include itself")
            study = compose(config_name="config", overrides=[f"experiment={name}"])
            study.dpi = cfg.dpi
            study.output_dir = str(output)
            run_one(study, output)
    else:
        run_one(cfg, output)
    print(f"Data and figures written to {output}", flush=True)
