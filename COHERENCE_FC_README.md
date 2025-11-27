# Coherence-Based Functional Connectivity Module

## Overview

This module implements a fast, standardized pipeline for constructing functional connectivity (FC) networks from SEEG time series using **magnitude-squared coherence (MSC)** with optional soft sparsification via surrogate-based null models.

**Key Features:**
- Multitaper spectral estimation (DPSS tapers) for robust cross-spectral density computation
- Band-averaged MSC adjacency matrices
- Optional soft sparsification using circular shift surrogates (preserves weight geometry)
- Output format compatible with correlation-based FC pipeline
- No changes required to downstream LRG analysis code

---

## Module Structure

```
src/lrg_eegfc/utils/coherence/
├── __init__.py           # Main pipeline: coherence_fc_pipeline()
├── msc.py               # CSD and MSC computation
├── bands.py             # Band validation helpers
├── surrogates.py        # Circular shift surrogate generation
└── sparsify.py          # Soft sparsification
```

---

## Quick Start

### Basic Usage

```python
from lrg_eegfc.utils.coherence import coherence_fc_pipeline
from lrg_eegfc.config.const import BRAIN_BANDS
import numpy as np

# Load your data (N channels × L samples)
X = ...  # shape (N, L)
fs = 512.0  # sampling frequency in Hz

# Compute coherence-based FC networks
adj_matrices = coherence_fc_pipeline(X, fs, bands=BRAIN_BANDS)

# Returns dict: {band_name: adjacency_matrix}
# e.g., adj_matrices['alpha'] is (N, N) adjacency for alpha band
```

### Without Sparsification (Dense MSC)

```python
# Get dense MSC matrices (no surrogate-based sparsification)
msc_matrices = coherence_fc_pipeline(
    X,
    fs,
    sparsify="none"
)
```

### Custom Frequency Bands

```python
# Define custom bands
custom_bands = {
    'slow': (0.5, 4.0),
    'medium': (4.0, 30.0),
    'fast': (30.0, 100.0),
}

# Compute for custom bands
adj_matrices = coherence_fc_pipeline(
    X,
    fs,
    bands=custom_bands,
    n_surrogates=200  # increase for final analysis
)
```

---

## API Reference

### Main Function: `coherence_fc_pipeline()`

```python
def coherence_fc_pipeline(
    X: NDArray,                                    # Input: (N, L) timeseries
    fs: float,                                     # Sampling frequency (Hz)
    bands: Dict[str, Tuple[float, float]] = None,  # Frequency bands
    n_surrogates: int = None,                      # Number of surrogates (default: 200)
    sparsify: str = "soft",                        # "soft" or "none"
    nperseg: int = 256,                            # Window length for CSD
    noverlap: int = None,                          # Overlap (default: nperseg // 2)
    nw: float = None,                              # Time-bandwidth product (default: 3.0)
    kmax: int = None,                              # Number of tapers (default: 5)
    zero_diagonal: bool = True,                    # Zero diagonal
    rng: np.random.Generator = None,               # RNG for reproducibility
) -> Dict[str, NDArray]:
    """
    Returns: Dict mapping band names to adjacency matrices (N, N)
    """
```

**Parameters:**

- **`X`**: Input time series, shape `(N, L)` where `N` is number of channels and `L` is number of samples
- **`fs`**: Sampling frequency in Hz
- **`bands`**: Dictionary `{band_name: (fmin, fmax)}`. If `None`, uses `BRAIN_BANDS` from config
- **`n_surrogates`**: Number of circular shift surrogates for null model. If `None`, uses `DEFAULT_N_SURROGATES` (200). Set to 0 to skip surrogates
- **`sparsify`**:
  - `"soft"`: Apply soft sparsification using surrogates
  - `"none"`: Return dense MSC matrices
- **`nperseg`**: Window length for spectral estimation (default: 256)
- **`noverlap`**: Number of overlapping samples (default: `nperseg // 2`)
- **`nw`**: Time-bandwidth product for multitaper (default: 3.0)
- **`kmax`**: Number of DPSS tapers (default: 5)
- **`zero_diagonal`**: Set diagonal to zero (no self-loops)
- **`rng`**: Random number generator for reproducibility

**Returns:**
- Dictionary mapping band names to adjacency matrices of shape `(N, N)`

---

## Algorithm Details

### 1. Cross-Spectral Density (CSD) via Multitaper

For each window `w` and taper `m`:
```
X_tapered = h_m(t) * x_i^(w)(t)
X_fft = FFT(X_tapered)
S_ij(f) = (1/(W*M)) * Σ_w Σ_m [X_i^(w,m)(f) * conj(X_j^(w,m)(f))]
```

Where:
- `h_m(t)` are DPSS (discrete prolate spheroidal sequences) tapers
- `W` is number of windows
- `M` is number of tapers

### 2. Magnitude-Squared Coherence (MSC)

```
Coh_ij(f) = |S_ij(f)|^2 / (S_ii(f) * S_jj(f))
```

Properties:
- `Coh_ij(f) ∈ [0, 1]`
- Symmetric
- Measures linear oscillatory coupling (amplitude + phase consistency)

### 3. Band Averaging

```
W_ij^(band) = mean(Coh_ij(f) for f in [fmin, fmax])
```

### 4. Soft Sparsification (Optional)

Generate `R` circular shift surrogates:
```
x_i^(r)(t) = x_i((t + Δ_i^(r)) mod L)
```

Compute empirical p-values:
```
p_ij = (#surrogates with W_surr >= W_obs) / R
```

Apply soft weighting:
```
A_ij = (1 - p_ij) * W_ij
```

This downweights non-significant edges while preserving the full weight geometry (no binarization).

---

## Configuration Constants

Add to `src/lrg_eegfc/config/const.py`:

```python
DEFAULT_N_SURROGATES = 200        # Number of surrogates for null model
DEFAULT_MULTITAPER_NW = 3.0       # Time-bandwidth product
DEFAULT_MULTITAPER_KMAX = 5       # Number of DPSS tapers
```

---

## Example Notebook

See [ipynb/UTILS-COHERENCE_NETWORKS.ipynb](../ipynb/UTILS-COHERENCE_NETWORKS.ipynb) for a complete example including:

1. Loading SEEG data
2. Computing dense MSC matrices
3. Soft sparsification with surrogates
4. Comparing dense vs sparsified distributions
5. Computing network statistics
6. Using custom frequency bands
7. Integration with LRG analysis

---

## Integration with Existing Pipeline

The coherence module is designed to be a **drop-in replacement** for correlation-based FC:

### Before (Correlation-based):
```python
from lrg_eegfc.utils.corrmat import build_corrmat_perband

corr_matrices = build_corrmat_perband(X, fs, ...)
```

### After (Coherence-based):
```python
from lrg_eegfc.utils.coherence import coherence_fc_pipeline

adj_matrices = coherence_fc_pipeline(X, fs, ...)
```

**Output format is identical**, so downstream LRG code (clustering, ultrametric analysis, etc.) works unchanged.

---

## Performance Notes

- **Dense MSC** (no surrogates): Fast, ~1-2 seconds for typical SEEG data
- **Soft sparsification** with 200 surrogates: ~2-5 minutes depending on data size
- For quick testing, use `n_surrogates=50`, then increase to 200 for final analysis
- Multitaper parameters affect computational cost:
  - Larger `nperseg` → better frequency resolution, slower
  - More tapers (`kmax`) → more robust, slower

---

## Advantages Over Correlation-Based FC

1. **Frequency-specific coupling**: Captures band-specific linear oscillatory interactions
2. **Robust spectral estimation**: Multitaper method reduces variance
3. **Phase + amplitude information**: MSC combines both (correlation only uses amplitude)
4. **Soft sparsification**: Preserves weight geometry without hard thresholding
5. **No bandpass filtering artifacts**: Works directly in frequency domain

---

## Limitations & Considerations

1. **Linear coupling only**: MSC captures linear relationships (like correlation)
2. **Computational cost**: Surrogate generation is slower than correlation
3. **Parameter choices**:
   - `nperseg` affects frequency resolution vs. time localization
   - Number of surrogates affects statistical power vs. computation time
4. **Frequency band validity**: Ensure `fmax < Nyquist frequency` (automatically validated)

---

## Testing

Run basic tests:
```bash
python -c "
from src.lrg_eegfc.utils.coherence import coherence_fc_pipeline
import numpy as np

X = np.random.randn(10, 1000)  # 10 channels, 1000 samples
fs = 512.0

# Test dense MSC
msc = coherence_fc_pipeline(X, fs, sparsify='none')
print(f'✓ Dense MSC: {len(msc)} bands')

# Test soft sparsification
adj = coherence_fc_pipeline(X, fs, n_surrogates=5, sparsify='soft')
print(f'✓ Soft-sparsified: {len(adj)} bands')
"
```

---

## References

1. **Multitaper spectral estimation**: Thomson, D. J. (1982). Spectrum estimation and harmonic analysis. *Proceedings of the IEEE*, 70(9), 1055-1096.
2. **Coherence analysis**: Bastos, A. M., & Schoffelen, J. M. (2016). A tutorial review of functional connectivity analysis methods and their interpretational pitfalls. *Frontiers in Systems Neuroscience*, 9, 175.
3. **Surrogate methods**: Prichard, D., & Theiler, J. (1994). Generating surrogate data for time series with several simultaneously measured variables. *Physical Review Letters*, 73(7), 951.

---

## Support

For questions or issues:
- Check the example notebook: `ipynb/UTILS-COHERENCE_NETWORKS.ipynb`
- Review the API documentation above
- Examine the source code in `src/lrg_eegfc/utils/coherence/`

---

**Version**: 1.0
**Last Updated**: 2025-11-25
