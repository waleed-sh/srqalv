import ast
from pathlib import Path
import subprocess
import sys

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
import numpy as np
import pytest

import srqalv
from srqalv.runner import validate


PRESETS = (
    "k5_expectations", "kernel", "dense_error", "mc_samples", "mc_orders",
    "slq_spin12", "slq_probes",
)


@pytest.fixture
def configs():
    with initialize_config_module(config_module="srqalv.configs", version_base="1.3"):
        yield


@pytest.mark.parametrize("name", PRESETS)
def test_preset_covers_run_arguments(configs, name):
    cfg = compose(config_name="config", overrides=[f"experiment={name}"])
    settings = OmegaConf.to_container(cfg.experiment, resolve=True)
    validate(settings, cfg.dpi)
    module = Path(srqalv.__file__).parent / "runs" / f"{settings['kind']}.py"
    tree = ast.parse(module.read_text())
    assert any(isinstance(node, ast.FunctionDef) and node.name == "run"
               and len(node.args.args) == 1 for node in tree.body)
    needed = {node.attr for node in ast.walk(tree)
              if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
              and node.value.id == "args"}
    assert needed <= settings.keys() | {"dpi", "out_dir"}


def test_publication_settings(configs):
    suite = compose(config_name="config", overrides=["experiment=all"])
    assert list(suite.experiment.experiments) == list(PRESETS)
    studies = {name: compose(config_name="config", overrides=[f"experiment={name}"]).experiment
               for name in PRESETS}
    assert list(studies["dense_error"].orders) == [5, 10, 20, 50, 100, 120, 160, 200, 250, 300, 400, 500]
    assert studies["dense_error"].orders == studies["mc_orders"].orders
    assert studies["mc_samples"].cutoff == 250000
    assert studies["mc_orders"].sample_count * studies["mc_orders"].chains == 262144
    assert list(studies["slq_spin12"].two_js) == [24] * 5
    assert list(studies["slq_probes"].probe_values) == [2, 8, 32, 128]
    assert not studies["slq_probes"].dense_check


@pytest.mark.parametrize("override", ["experiment.orders=[]", "experiment.orders=[1.5]",
                                     "experiment.orders=[0]", "experiment.cg_tol=-1"])
def test_invalid_settings(configs, override):
    cfg = compose(config_name="config", overrides=["experiment=dense_error", override])
    with pytest.raises(ValueError):
        validate(OmegaConf.to_container(cfg.experiment, resolve=True), cfg.dpi)


def test_native_extension():
    from srqalv._native import VolumeSRQ

    volume = VolumeSRQ([2] * 4, shifts=5, triple_signs={(0, 1, 2): 1})
    result = volume(np.ones(volume.dim, dtype=np.complex128))
    assert result.shape == (3,)
    assert np.isfinite(result).all()


def test_command_outside_checkout(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "srqalv", "experiment=slq_spin12", "--cfg", "job", "--resolve"],
        cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    assert "lanczos_steps: 48" in result.stdout
    assert not list(tmp_path.iterdir())


def test_multirun_separates_outputs(tmp_path):
    output = tmp_path / "sweep"
    subprocess.run(
        [sys.executable, "-m", "srqalv", "-m", "experiment=kernel", "experiment.cutoffs=[1]",
         "experiment.orders=[5]", "experiment.embedding_seed=1,2", "dpi=72", f"output_dir={output}"],
        cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    for job, seed in enumerate((1, 2)):
        run = output / str(job)
        cfg = OmegaConf.load(run / "configs/kernel.yaml")
        assert cfg.experiment.embedding_seed == seed
        assert Path(cfg.output_dir) == run
        assert len(list((run / "figures").glob("*.png"))) == 1
        assert len(list((run / "data").rglob("*.csv"))) == 1
        assert (run / ".hydra/overrides.yaml").exists()
