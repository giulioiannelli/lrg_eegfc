# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LRG EEG Functional Connectivity** is a toolkit for computing frequency-specific functional connectivity (FC) networks from stereo-EEG (SEEG) recordings. It implements correlation-based and coherence-based methods for building brain network graphs, with percolation-based threshold selection and ultrametric analysis using Laplacian-based graph operators.

- **Language**: Python 3.11+
- **Domain**: Neuroscience, brain network analysis, signal processing
- **Key Dependency**: `lrgsglib` (Laplacian Renormalisation Group for Signed Graphs) - included as Git submodule

## Installation & Setup

### First-time setup:
```bash
# Clone with submodules
git submodule update --init --recursive

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install lrgsglib dependency
pip install ./lrgsglib

# Install this package in editable mode
pip install -e .[dev]
```

### Development dependencies:
The `[dev]` extra includes pytest, black, isort, flake8, mypy.

### Jupyter notebooks:
```bash
pip install -e .[jupyter]
```

## Common Commands

### Running analyses:
```bash
# Compute correlation-based FC network for a patient/phase/band
lrg-eegfc-corr --patient Pat_01 --phase rsPre --band beta --plot-all

# Specify custom paths
lrg-eegfc-corr \
  --patient Pat_01 \
  --phase rsPre \
  --band alpha \
  --dataset-root /path/to/data/stereoeeg_patients \
  --output-root /path/to/output \
  --jump-index 0 \
  --plot-all
```

### Code quality checks:
```bash
# Run all quality gates (do this before commits)
pytest
black --check src
isort --check src
flake8 src
mypy src

# Auto-format code
black src
isort src
```

### Running single tests:
```bash
# Run specific test file
pytest tests/test_corrmat.py

# Run specific test function
pytest tests/test_corrmat.py::test_build_corr_network -v

# Run with markers
pytest -m "not slow"
```

## Architecture

### Package structure:
```
src/lrg_eegfc/
├── __init__.py           # Public API exports
├── cli.py                # Command line interface (lrg-eegfc-corr)
├── constants.py          # EEG frequency bands, phase labels
├── workflow.py           # High-level workflows (compute_band_connectivity)
├── plotting.py           # Visualization utilities
├── config/               # Configuration constants
│   ├── const.py          # Core constants (BRAIN_BANDS, etc.)
│   └── plotlib.py        # Plotting configuration
└── utils/
    ├── corrmat/          # Correlation-based FC (main method)
    │   ├── base.py       # Core correlation matrix operations
    │   ├── bands.py      # Per-band correlation computation
    │   ├── thresholds.py # Percolation-based threshold selection
    │   ├── network.py    # Network processing
    │   └── structures.py # Batch processing for patients/phases
    ├── coherence/        # Coherence-based FC (alternative method)
    │   ├── __init__.py   # Main pipeline: coherence_fc_pipeline()
    │   ├── msc.py        # Magnitude-squared coherence via multitaper
    │   ├── bands.py      # Band validation
    │   ├── surrogates.py # Circular shift surrogates for null models
    │   └── sparsify.py   # Soft sparsification
    └── datamanag/        # Data loading
        ├── loaders.py    # Low-level MAT file loading
        └── patient.py    # PatientRecording dataclass, load_timeseries()
```

### Core data flow:

1. **Loading** (`utils/datamanag/`): Load `.mat` files from `data/stereoeeg_patients/Pat_XX/` into NumPy arrays
2. **Filtering**: Apply bandpass filter using `lrgsglib.utils.basic.signals.bandpass_sos()`
3. **FC computation** (`utils/corrmat/` or `utils/coherence/`):
   - **Correlation**: Pearson correlation → percolation-based thresholding
   - **Coherence**: Multitaper CSD → magnitude-squared coherence → optional soft sparsification
4. **Network analysis**: Convert to NetworkX graph, find connected components, compute ultrametric distances
5. **Visualization** (`plotting.py`): Correlation heatmaps, dendrograms, network graphs, entropy plots

### Two FC methods:

**Correlation-based** (default, `utils/corrmat/`):
- Bandpass filter time series → compute Pearson correlation → apply percolation threshold
- Threshold selection: Find "jumps" in number of components as edges are added
- Main functions: `build_corr_network()`, `find_threshold_jumps()`, `compute_band_connectivity()`

**Coherence-based** (newer, `utils/coherence/`):
- Multitaper spectral estimation → magnitude-squared coherence (MSC) → optional surrogate-based sparsification
- Soft weighting preserves edge weights (no binarization)
- Main function: `coherence_fc_pipeline(X, fs, bands=BRAIN_BANDS, sparsify="soft")`
- See `COHERENCE_FC_README.md` for details

### Key constants (`constants.py`, `config/const.py`):

```python
BRAIN_BANDS = {
    "delta": (0.53, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 80.0),
    "high_gamma": (80.0, 300.0),
}

PHASE_LABELS = ("rsPre", "taskLearn", "taskTest", "rsPost")
```

## Data Organization

Expected directory structure:
```
data/
└── stereoeeg_patients/
    ├── Pat_01/
    │   ├── rsPre.mat           # SEEG recordings (MATLAB format)
    │   ├── taskLearn.mat
    │   ├── taskTest.mat
    │   ├── rsPost.mat
    │   ├── Implant_pat_01.csv  # Channel metadata (optional)
    │   └── channel_labels.csv
    └── Pat_02/
        └── ...
```

Output artefacts written to `data/correlations/<patient>/`:
- `<band>_<phase>_corr.npy` - correlation matrices
- `<band>_<phase>_corr.png` - heatmaps
- `<band>_<phase>_dendrogram.png`
- `<band>_<phase>_graph.png`
- `<band>_<phase>_entropy.png`

## Important Implementation Details

### Type hints are mandatory:
All functions must have complete type hints. MyPy is configured with strict settings in `pyproject.toml`.

### Import style:
- Prefer explicit imports: `from module import name`
- Avoid wildcard imports except in `__init__.py` for re-exports
- `lrgsglib` and `emd` are marked as `ignore_missing_imports` in mypy config

### Percolation threshold selection:
The `find_threshold_jumps()` function identifies correlation thresholds where the number of graph components changes dramatically. `jump_index=0` selects the threshold for a single giant component.

### lrgsglib dependency:
This external library provides:
- `lrgsglib.utils.basic.signals.bandpass_sos()` - SOS bandpass filtering
- Laplacian-based graph renormalization tools
- Entropy and percolation utilities

Install separately: `pip install ./lrgsglib` (included as submodule)

### CLI caching:
`lrg-eegfc-corr` caches computed correlation matrices as `.npy` files and reuses them unless `--overwrite` is passed. This speeds up iterative plotting/analysis.

### Notebooks (`ipynb/`):
Extensive Jupyter notebooks for:
- Distance measure comparisons (`DSTCMP_*.ipynb`)
- Figure generation for papers (`FIGMNTGN*.ipynb`)
- Utility demonstrations (`UTILS-COHERENCE_NETWORKS.ipynb`, `UTILS-MUTUAL_INFORMATION_NETWORKS.ipynb`)
- Per-patient analyses (`TEST_per_patient_analysis.ipynb`)

## Code Style

- **Line length**: 88 characters (Black default)
- **Formatting**: Use Black + isort (configured in `pyproject.toml`)
- **Linting**: flake8 with E203, W503 ignored for Black compatibility
- **Type checking**: mypy with strict settings enabled
- **Docstrings**: Required for all public functions/classes
- **Testing**: pytest with markers for slow/integration tests

## Testing Notes

- Test files go in `tests/` directory
- Use markers: `@pytest.mark.slow`, `@pytest.mark.integration`
- Run fast tests only: `pytest -m "not slow"`
- Coverage configured to omit test files and `__main__` blocks

## Additional Documentation

- `README.md` - User-facing documentation, installation, API overview
- `COHERENCE_FC_README.md` - Detailed guide for coherence-based FC module
- `docs/overview.md` - Architecture overview (slightly outdated, references old module structure)
- `docs/developer-guide.md` - Development workflow, coding standards
- `lrgsglib/README.md` - External dependency documentation

## Common Pitfalls

1. **Don't forget to install lrgsglib**: It's a Git submodule, must be installed separately
2. **Data path assumptions**: Default paths expect `data/stereoeeg_patients/` structure
3. **Frequency band names**: Use exact keys from `BRAIN_BANDS` dict (e.g., "low_gamma" not "lowgamma")
4. **Phase names**: Must match `PHASE_LABELS` exactly (case-sensitive)
5. **MAT file format**: Expected variables: main data array + optional `Parameters` struct
6. **Coherence module**: New addition, not yet integrated into CLI (use Python API directly)
