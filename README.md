# srqalv

Hydra-based reproducible of the SRQ volume-operator experiments conducted in the pre-print 
"[A matrix free action of the Ashtekar-Lewandowski volume operator of loop quantum gravity](https://arxiv.org/abs/2606.18397)".

The repository contains the numerical methods, publication presets and plotting routines.
Reference data are distributed separately through [Zenodo](https://zenodo.org/records/23837394).

## Installation

Use Python 3.13 and a C++17 compiler.

From the repository root, create a virtual environment, activate it and install the dependencies as below:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

This installs the pinned numerical dependencies and builds the native extension  automatically. 
The experiments run on CPU and do not need MPI or a GPU.

For development, use `python -m pip install -e '.[test]'` instead.

## Reproduce the Paper

```sh
srqalv experiment=all output_dir=results/paper
```

This runs all seven publication presets and writes the data and 11 PNG figures.
The equivalent shell entry point is `sh reproduce.sh output_dir=results/paper`.

Run a single experiment with:

```sh
srqalv experiment=k5_expectations
srqalv experiment=kernel
srqalv experiment=dense_error
srqalv experiment=mc_samples
srqalv experiment=mc_orders
srqalv experiment=slq_spin12
srqalv experiment=slq_probes
```

| Preset | Calculation | Paper figure |
| --- | --- | --- |
| `k5_expectations` | K5 vertex expectations versus $M$ and $\Lambda_v$, cutoffs 1 and 2 | 1 and 4 |
| `dense_error` | Four-valent matrix-element error, $2j=20$, $M$ through 500 | 2 |
| `mc_samples` | K5 cutoff 250000, sample-count sweep at $M=100$ | 3, left |
| `mc_orders` | Same K5 sector, $M$ sweep through 500 | 3, right |
| `kernel` | Kernel preservation at cutoffs 1 and 2 | 5 |
| `slq_spin12` | Five-valent SRQ-SLQ, $2j=24$, $M=20$, 64 probes | 6 |
| `slq_probes` | Five-valent probe sweep, $2j=4$, $M=200$ | 7 |

Without `output_dir`, Hydra creates a timestamped directory under `outputs/`.
Explicit output directories are reused, so **rerunning into one replaces files**
**with matching names**. Keep downloaded reference data in a different directory.

## Change Parameters

Settings are ordinary Hydra overrides. Quote lists in the shell:

```sh
srqalv experiment=dense_error experiment.two_j=40 'experiment.orders=[10,50,100]'
srqalv experiment=k5_expectations 'experiment.cutoffs=[1]' 'experiment.orders=[5,20,100]'
srqalv experiment=slq_probes 'experiment.probe_values=[8,32,128,256]'
srqalv experiment=mc_orders experiment.sample_count=4096 dpi=300
```

`experiment.orders` lists the SRQ orders $M$. `mc_samples` instead uses the
single value `experiment.order`. Spin inputs are doubled spins, $2j$.
`sample_count` and `sample_counts` count samples per group, so the total is
multiplied by `experiment.chains`.

For separate runs over several parameter values, use Hydra multirun:

```sh
srqalv -m experiment=dense_error experiment.two_j=10,20,40 output_dir=results/spin_sweep
```

Jobs get numbered subdirectories. View the effective configuration without
running a calculation:

```sh
srqalv experiment=slq_spin12 --cfg job --resolve
srqalv --help
```

`experiment=all` selects the unchanged paper presets. Modify individual presets
with the commands above, or edit their YAML files before running the full suite.
The top-level `dpi` override applies to all experiments in a suite.

## Outputs

```text
results/paper/
  data/          numerical tables and arrays, grouped into datasets 01-07
  figures/       high-resolution PNGs
  configs/       resolved settings for each experiment
  config.yaml    resolved settings for the whole invocation
  .hydra/        Hydra configuration and command-line overrides
```

The dataset names and numerical file formats match the Zenodo deposit.
Timing columns and floating-point details can depend on the machine and libraries.


## Notes

All calculations use $V_v=|Q_v|^{1/2}$ with its dimensional prefactor set to one.
The low-cutoff K5 and four-valent error plots use the literal dense volume as the
reference. The kernel plot evaluates the scalar SRQ response after numerically
zero eigenvalues are snapped to zero.

The high-cutoff Monte Carlo studies fix the ten edge spins and sample local
intertwiner channels in the uniform positive state. Their bands show one sampling
standard error, not the SRQ approximation error.

The spectral histograms are spectra of the **finite-$M$ SRQ approximation**, not
the literal volume. The curves are smoothed, probe-averaged weighted Ritz measures
from direct SRQ-SLQ. These runs use cached sparse BT transitions inside the JAX
shifted solves, without forming a dense volume matrix for the SLQ action.


## License

Code licensing and attribution are in `LICENSE` and `NOTICE`.
