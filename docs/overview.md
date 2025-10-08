# Project overview

This document summarises the architecture of the **LRG EEG Functional
Connectivity** toolkit after the 2024 refactor.

## Package layout

```
src/lrg_eegfc/
├── __init__.py              # Public API exports
├── cli.py                   # Command line entry points
├── constants.py             # Frequency bands and parameter keys
├── correlation.py           # Correlation matrix utilities
├── io.py                    # Dataset loading helpers
├── plotting.py              # Plot generation helpers
└── workflow.py              # High level orchestration helpers
```

### Data flow

1. **Loading** – :mod:`lrg_eegfc.io` contains `load_timeseries` and related
   helpers that normalise each `.mat` file into NumPy arrays.  Optional channel
   metadata is automatically merged when available.
2. **Processing** – :mod:`lrg_eegfc.correlation` implements the
   Marchenko–Pastur cleaning step, percolation-based threshold selection and the
   band-wise correlation matrix computation.
3. **Workflow orchestration** – :mod:`lrg_eegfc.workflow` ties loading and
   processing together and produces a ready-to-use
   :class:`~lrg_eegfc.workflow.BandComputationResult` instance.
4. **Presentation** – :mod:`lrg_eegfc.plotting` provides plotting helpers and is
   used both by the CLI and notebooks.

## Command line interface

The `lrg-eegfc-corr` entry point defined in :mod:`lrg_eegfc.cli` exposes a
single command that covers the typical workflow: load a patient/phase
recording, band-pass filter it, compute a correlation matrix, apply a threshold
and optionally produce plots.  All filesystem paths are configurable and the
command reuses cached correlation matrices unless `--overwrite` is passed.

## Dependency notes

* `lrgsglib` supplies the Laplacian-based operators (entropy and percolation
  utilities) and must be installed separately.
* `networkx`, `numpy`, `pandas`, `matplotlib`, `scipy`, `h5py` and `emd` are
  standard scientific Python dependencies already declared in ``pyproject.toml``.

## Extending the toolkit

* New features should be added as separate modules under ``lrg_eegfc`` and
  exported through ``__init__.__all__``.
* Prefer pure functions with explicit parameters; avoid hidden state and module
  level singletons.
* Keep CLI behaviour deterministic – anything random should accept a seed
  parameter with a sensible default.
