# CosmoNPC

CosmoNPC is a Python/MPI code for mesh-based measurements of large-scale structure statistics, **based on a suite of optimized multipole expansion algorithms**, with an emphasis on power spectrum multipoles and bispectrum multipoles in both box-like and survey-like geometries.

The present implementation is aimed at:

- power spectrum measurements (`pk`)
- Sugiyama-style bispectrum multipoles (`bk_sugi`)
- auto-correlations and cross-correlations
- simulation boxes and survey catalogs
- MPI-distributed FFT-based estimators

## Measurement Scope

### Power Spectrum

The current implementation includes:

- auto power spectrum measurements
- cross power spectrum measurements
- box-like geometry
- survey-like geometry
- multipole measurements specified by `poles`

### Bispectrum

The current implementation includes:

- auto bispectrum measurements
- cross bispectrum measurements
- box-like geometry
- survey-like geometry
- `tracer_type` choices including `aaa`, `aab`, `abb`, `abc`
- diagonal and full data-vector modes for `bk_sugi`

Cross statistics are part of the core code structure for both `pk` and `bk`, rather than a later extension.

## Repository Structure

Main package modules:

- `run.py`
- `catalog_processor.py`
- `mesh_generator.py`
- `math_funcs.py`
- `stat_algorithm.py`

Representative configuration files:

- `config/pk_box.py`
- `config/pk_survey.py`
- `config/bk_sugi_box.py`
- `config/bk_sugi_survey.py`

Example run script:

- `example.py`

## Dependencies

The Python dependencies currently required by the code are listed in `pyproject.toml`:

- `numpy`
- `mpi4py`
- `h5py`
- `astropy`
- `fitsio`
- `pmesh`
- `sympy`
- `numexpr`

Notes:

- `numexpr` is required because some symbolic kernels are evaluated through `sympy.lambdify(..., 'numexpr')`.
- `pmesh` is a core dependency of the mesh and FFT workflow.
- MPI support must be available in the Python environment for `mpi4py`.

To install `CosmoNPC`, simply run 
```bash
pip install git+https://github.com/YunchenXie/CosmoNPC.git
```

## Running the Code

The workflow is config-driven.

1. Import one default python configuration dictionary from `cosmonpc.config`
2. Deepcopy and update the `CONFIG` dictionary for your dataset and measurement target.
3. Pass the updated `CONFIG` to `cosmonpc.run_stats` function.
4. Run with MPI.

A typical invocation is:

```bash
mpirun -n 4 python example.py
```

The active configuration is selected near the top of `example.py`.

## Configuration Overview

The code is controlled by a single `CONFIG` dictionary. Important entries include:

- `statistic`
  - `pk`
  - `bk_sugi`

- `geometry`
  - `box-like`
  - `survey-like`

- `correlation_mode`
  - `auto`
  - `cross`

- `tracer_type`
  - relevant for bispectrum
  - examples include `aaa`, `aab`, `abb`, `abc`

- `nmesh`
  - mesh resolution `[nx, ny, nz]`

- `boxsize`
  - physical box size in units of Mpc/h

- `sampler`
  - assignment window such as `tsc`, `cic`, `ngp` or `pcs`

- `compensation`
  - whether to apply Fourier-space compensation for the assignment window

- `poles`
  - power spectrum multipoles

- `angu_config` for bispectrum
  - bispectrum angular configuration `[ell_1, ell_2, L]`

- `data_vector_mode` for bispectrum
  - `diagonal`
  - `full`

- `block_size` for full bispectrum data vectors
  - only relevant for `bk_sugi` with `data_vector_mode="full"`
  - controls the block partition of the full `(k1, k2)` data vector

- `normalization_scheme`
  - survey normalization mode

- `alpha_scheme`
  - survey random/data normalization convention

## Interpretation of `block_size`

`block_size` is a performance and memory-control parameter for `bk_sugi` when `data_vector_mode="full"`.

In full bispectrum mode, the output is a two-dimensional data vector `B(k_1, k_2)`. A straightforward implementation would evaluate the whole `(k_1, k_2)` rectangle at once. This minimizes repeated work, but can become memory-intensive when `nmesh` is large.

`block_size` controls how this rectangle is partitioned into smaller square blocks:

- `block_size = "full"`
  - treat the whole `(k_1, k_2)` rectangle as a single block
  - usually minimizes repeated work
  - usually maximizes memory usage

- `block_size = 1`
  - treat each `(k_1, k_2)` location as its own smallest block
  - usually minimizes memory usage
  - usually increases repeated construction of intermediate binned fields

- `block_size = N`
  - where `N` is an integer between `1` and `k_bins`
  - split the full data vector into `N x N` square blocks
  - this is the practical compromise between speed and memory

Within one block, CosmoNPC can reuse precomputed binned fields instead of reconstructing them for every individual `(k_1, k_2)` pair. This is why a larger `block_size` can be faster. The trade-off is that more intermediate mesh fields remain in memory at the same time.

In practical terms:

- larger `block_size` means more reuse and higher memory usage
- smaller `block_size` means less reuse and lower memory usage

This parameter has no practical effect for `data_vector_mode="diagonal"`, because the diagonal mode does not evaluate the full `(k_1, k_2)` rectangle.

A reasonable usage strategy is:

- use `block_size = "full"` only when memory is clearly sufficient
- use a moderate integer value when running large meshes in `full` mode
- reduce `block_size` if the job becomes memory-limited

## Input Catalogs

### Box-like Catalogs

Box-like inputs typically use Cartesian positions:

- `x`, `y`, `z`

Optional velocity columns can be supplied when redshift-space distortion is applied. In that case, columns such as `VX`, `VY`, and `VZ` are needed, depending on the catalog format.

### Survey-like Catalogs

Survey-like inputs typically use sky coordinates and survey weights, for example:

- `RA`
- `DEC`
- `Z`
- `WEIGHT_FKP`
- `WEIGHT`, completeness and systematic weights
- `NZ`

The exact interpretation depends on:

- `column_names`
- `comp_weight_plan`
- `geometry`

## Output

Results are written to the directory specified by `output_dir`.

Depending on the selected statistic, the output may include:

- raw signal estimates
- shot-noise terms
- normalized final measurements
- metadata needed for validation and post-processing

## Technical Characteristics

Some technical characteristics of the current implementation are:

- heavy numerical work is delegated to `pmesh`, `numpy`, and `numexpr`
- MPI is used as the main parallelization layer
- the bispectrum code separates signal and shot-noise logic
- the `full` bispectrum mode supports block-based evaluation to balance speed and memory
- symmetry relations are used to reduce repeated work in selected angular configurations
- both auto and cross statistics are built into the code structure rather than treated as afterthoughts
