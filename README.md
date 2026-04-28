![cnpc](./resources/cnpc.jpg)

# CosmoNPC

CosmoNPC is a Python/MPI code for mesh-based measurements of large-scale structure statistics in box-like and survey-like geometries.

Current focus:

- power spectrum multipoles: `pk`
- Sugiyama bispectrum multipoles: `bk_sugi`
- auto and cross correlations
- box and survey catalogs
- MPI-distributed FFT-based estimators

## Current Scope

- `pk`
  - box-like
  - survey-like
  - auto / cross
  - user-specified `poles`

- `bk_sugi`
  - box-like
  - survey-like
  - auto / cross
  - `tracer_type` = `aaa`, `aab`, `abb`, `abc`
  - `data_vector_mode` = `diagonal` or `full`
  - `shotnoise_mode` = `ana`, `fft`, or `both`

`bk_sco` is not part of the stable workflow yet.

## Package Layout

Main modules:

- `task_executor.py`
- `catalog_processor.py`
- `mesh_generator.py`
- `math_evaluator.py`
- `clustering_estimator.py`
- `param_helper.py`

Representative configs:

- `src/cosmonpc/config/pk_box.py`
- `src/cosmonpc/config/pk_survey.py`
- `src/cosmonpc/config/bk_sugi_box.py`
- `src/cosmonpc/config/bk_sugi_survey.py`

Example:

- `example.py`

## Install

```bash
pip install git+https://github.com/YunchenXie/CosmoNPC.git
```

Core dependencies are listed in `pyproject.toml`. In practice you need at least:

- `numpy`
- `mpi4py`
- `pmesh`
- `sympy`
- `numexpr`
- `astropy`
- `fitsio`
- `h5py`
- `scipy`

## Minimal Usage

The code is config-driven.

```python
from copy import deepcopy
from cosmonpc import run_stats
from cosmonpc.config import bk_sugi_survey

config = deepcopy(bk_sugi_survey.CONFIG)
config["output_dir"] = "./results"
run_stats(config)
```

Run with MPI, for example:

```bash
mpirun -n 4 python example.py
```

## Key Configuration Fields

- `statistic`: `pk` or `bk_sugi`
- `geometry`: `box-like` or `survey-like`
- `correlation_mode`: `auto` or `cross`
- `tracer_type`: bispectrum tracer pattern such as `aaa`, `aab`, `abb`, `abc`
- `catalogs`: input file paths
- `column_names`: catalog column mapping
- `nmesh`: mesh size `[nx, ny, nz]`
- `boxsize`: box size in `Mpc/h`
- `sampler`: `ngp`, `cic`, `tsc`, or `pcs`
- `interlaced`: whether to use interlaced painting
- `k_min`, `k_max`, `k_bins`: k-bin settings
- `poles`: power-spectrum multipoles
- `angu_config`: bispectrum angular configuration `[ell_1, ell_2, L]`
- `data_vector_mode`: `diagonal` or `full` for `bk_sugi`
- `shotnoise_mode`: `ana`, `fft`, or `both`
- `normalization_scheme`: normalization mode
- `alpha_scheme`: survey alpha convention
- `high_order_mode`: `default`, `fast`, or `compare` for survey `pk` high-order multipoles
- `output_dir`: output directory

## Input Formats

Box-like catalogs:

- `.npy`, `.fits`, `.h5`, `.hdf5`
- Cartesian positions are expected
- optional velocity columns can be used for RSD

Survey-like catalogs:

- currently `.fits`
- typical fields include `RA`, `DEC`, `Z`, `WEIGHT_FKP`, completeness weights, and `NZ`

## Output

Results are written to `output_dir` as `.npy` dictionaries.

Typical outputs include:

- final measured statistic
- raw signal term
- shot-noise terms
- normalization metadata
- effective k values and mode counts

## Technical Features

- Symmetry-aware estimator design is used aggressively to reduce redundant work. The code exploits conjugate, swap, and tracer/leg symmetry whenever possible, especially in bispectrum loops.
- The survey and box implementations include an analytical shot-noise treatment for the Sugiyama estimator, together with an FFT-based $S_{\ell_1\ell_2L}|_{i=j\neq k}$ evaluation path for cross-checking and comparison. The original Sugiyama shot-noise prescription is no longer the supported route in this code.
- In `bk_sugi` full mode, `block_size` accelerates the calculation by trading memory for reuse: larger blocks keep more intermediate shell/binned fields in memory, so the code can reuse them across many `(k1, k2)` pairs instead of rebuilding them repeatedly.

## Notes

- `full` bispectrum mode supports block-based evaluation through `block_size`.
- Survey and box pipelines do not share exactly the same normalization and shot-noise structure.
- Small smoke tests can be run with reduced `nmesh`, smaller catalogs, and fewer bins before launching production jobs.
