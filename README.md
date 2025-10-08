# LRG EEG Functional Connectivity

Tools for computing frequency-specific functional connectivity graphs from
stereo-EEG (SEEG) recordings collected in the [Living and Relational Graphs
(LRG)](https://github.com/giulioiannelli) research project.  The repository now
ships as a regular Python package, complete with a command line interface,
documented public APIs and optional plotting helpers.

---

- [Installation](#installation)
  - [Installing `lrgsglib`](#installing-lrgsglib)
  - [Editable development install](#editable-development-install)
- [Dataset layout](#dataset-layout)
- [Command line usage](#command-line-usage)
- [Python API overview](#python-api-overview)
- [Plotting utilities](#plotting-utilities)
- [Developer guide](#developer-guide)

---

## Installation

The project targets Python **3.11+**.  The package depends on
[`lrgsglib`](https://github.com/giulioiannelli/lrgsglib), a companion library
that provides the Laplacian-based graph operators used throughout the
workflows.  Install the core package with:

```bash
pip install .
```

The command automatically registers the ``lrg-eegfc-corr`` command line tool
and installs the runtime dependencies (`numpy`, `pandas`, `scipy`, `networkx`,
`matplotlib`, `h5py`, `emd` and `lrgsglib`).

### Installing `lrgsglib`

`lrgsglib` is included as a Git submodule for convenience.  When cloning the
repository run:

```bash
git clone https://github.com/giulioiannelli/lrg_eegfc.git
cd lrg_eegfc
git submodule update --init --recursive
pip install ./lrgsglib
```

Alternatively install it directly from its repository:

```bash
pip install git+https://github.com/giulioiannelli/lrgsglib.git
```

### Editable development install

For local development create a virtual environment and install the project in
editable mode together with the optional developer tooling:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

The `dev` extra installs pytest, black, isort, flake8 and mypy.  Run the full
quality gate with:

```bash
pytest
black --check src
isort --check src
flake8 src
mypy src
```

## Dataset layout

The command line tools assume the following directory structure by default:

```
└── data/
    └── stereoeeg_patients/
        ├── Pat_01/
        │   ├── rsPre.mat
        │   ├── taskLearn.mat
        │   ├── taskTest.mat
        │   ├── rsPost.mat
        │   ├── Implant_pat_01.csv
        │   └── channel_labels.csv
        └── …
```

Use ``--dataset-root`` to point to a different directory when running the CLI.
All generated artefacts (correlation matrices and plots) are written to
``data/correlations/<patient>/`` by default.

## Command line usage

The package exposes the ``lrg-eegfc-corr`` entry point which computes a band-
specific correlation matrix for a patient and optionally creates a set of
plots:

```bash
lrg-eegfc-corr \
    --patient Pat_01 \
    --phase rsPre \
    --band beta \
    --dataset-root /path/to/data/stereoeeg_patients \
    --plot-all
```

Key options:

* ``--filter-time`` – restrict the analysis to the first *N* samples.
* ``--jump-index`` – choose which percolation jump to use when selecting the
  correlation threshold (0 = single giant component).
* ``--channel-names`` – path to a ``ChannelNames.mat`` file with the
  ``ChannelNames`` variable, used to label the dendrogram/graph plots.
* ``--plot-*`` flags – enable individual plots; ``--plot-all`` toggles every
  available plot.

Run ``lrg-eegfc-corr --help`` for the full list of options.

## Python API overview

The public API lives in :mod:`lrg_eegfc`.  Highlights include:

| Function | Description |
| --- | --- |
| ``load_timeseries(patient, phase, root_path)`` | Load a SEEG recording into a ``(channels, samples)`` array. |
| ``load_patient_dataset(patient, root_path)`` | Return a dictionary of ``phase -> PatientRecording`` objects. |
| ``build_correlation_network(timeseries, threshold=...)`` | Produce a processed correlation matrix. |
| ``build_band_correlation_matrices(data_ts, fs)`` | Compute per-band correlation matrices with automatic threshold selection. |
| ``compute_band_connectivity(patient, phase, band, dataset_root)`` | Convenience wrapper that ties together loading, filtering and threshold selection. |

Refer to the in-code docstrings for full parameter documentation.

## Plotting utilities

The :mod:`lrg_eegfc.plotting` module contains the functions that back the CLI
plots (`plot_correlation_matrix`, `plot_entropy`, `plot_dendrogram`,
`plot_graph`).  They accept plain ``pathlib.Path`` destinations so they can be
used interactively inside notebooks.

## Developer guide

* New code must include type hints and informative docstrings.
* Keep imports explicit – wildcard imports are intentionally avoided.
* The ``docs/`` directory contains extended documentation for architecture and
  contribution guidelines.
* Use ``pip install -e .[dev]`` to bring in the linting and formatting tools.

Issues and pull requests are welcome!  Please open an issue with a reproducible
example when reporting bugs.
