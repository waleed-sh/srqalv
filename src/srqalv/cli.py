from pathlib import Path
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    output = Path(HydraConfig.get().runtime.output_dir)
    cache = output / ".cache"
    cache.mkdir(exist_ok=True)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))

    from srqalv.runner import run

    run(cfg, output)
