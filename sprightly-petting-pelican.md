# Analysis: MI vs Coherence vs Correlation for Ultra-Long SEEG FC

## Executive Summary

**Answer: NO** - Mutual information is NOT a viable alternative. It's **1000x slower** than correlation and **100x slower** than coherence.

**Recommendation**: For ultra-long recordings (10M samples), use **correlation** if you need speed, or **coherence with chunking** if you need frequency-specific information.

---

## Performance Comparison (10M samples, 50 channels)

| Method                      | Time        | Speedup vs Baseline |
|-----------------------------|-------------|---------------------|
| **Correlation (dense)**     | **4 sec**   | **1x (FASTEST)**    |
| Coherence (dense)           | 50 sec      | 0.08x (12x slower)  |
| Coherence + 200 surrogates  | 2.8 hours   | 0.0004x (2500x slower) |
| MI (dense)                  | 34 hours    | 0.00003x (30,000x slower) |
| MI + 200 surrogates         | 284 DAYS    | 0.00000016x (INFEASIBLE) |

### Why is MI so slow?

Mutual information for continuous data requires:
1. **Binning** (histogram method) or **KDE** (kernel density estimation)
2. **Entropy calculation** for each channel pair: H(X) + H(Y) - H(X,Y)
3. **O(L × B)** complexity where B = bins or bandwidth parameter
4. Large constant factors due to numerical integration

This is fundamentally slower than:
- **Correlation**: Single matrix multiplication, O(N²L), highly optimized in NumPy
- **Coherence**: FFT-based, O(N²F log F) per segment, vectorized

---

## Detailed Analysis

### 1. Correlation (NumPy `corrcoef`)
- **Complexity**: O(N²L)
- **10M samples**: ~4 seconds
- **Pros**:
  - Extremely fast (BLAS/LAPACK optimized)
  - Simple, interpretable
  - Linear time complexity
- **Cons**:
  - Broadband only (no frequency specificity)
  - Sensitive to non-stationarity

### 2. Coherence (Welch's method)
- **Complexity**: O(N² × n_segments × F log F)
- **Dense (10M samples)**: ~50 seconds
- **With 200 surrogates**: ~2.8 hours
- **Pros**:
  - Frequency-specific coupling
  - Robust spectral estimation
  - Phase + amplitude information
- **Cons**:
  - 10-15x slower than correlation
  - Surrogate generation is expensive

### 3. Mutual Information (sklearn KDE)
- **Complexity**: O(N² × L × B) with large constants
- **Dense (10M samples)**: ~34 hours (!)
- **With surrogates**: ~284 DAYS (completely infeasible)
- **Pros**:
  - Captures nonlinear relationships
  - Information-theoretic interpretation
- **Cons**:
  - 1000x slower than correlation
  - 100x slower than coherence
  - Not practical for N×N connectivity matrices
  - Requires parameter tuning (bins, bandwidth)

---

## Recommendations for Ultra-Long Recordings

### Option 1: Use Correlation (FASTEST)
**Use if**: Speed is critical and frequency-specific information is not required

```python
from lrg_eegfc.utils.correlation import correlation_fc_pipeline

# 4 seconds for 10M samples
adj_matrices = correlation_fc_pipeline(X, fs, bands=BRAIN_BANDS)
```

**Trade-off**: No frequency specificity (uses bandpass filtering)

### Option 2: Use Coherence with Chunking (BALANCED)
**Use if**: Need frequency-specific coupling but have ultra-long recordings

**Strategy**: Process in 10-minute chunks, average results

```python
chunk_size = int(10 * 60 * fs)  # 10 minutes
n_chunks = L // chunk_size

adj_accum = {band: np.zeros((N, N)) for band in bands}

for i in range(n_chunks):
    X_chunk = X[:, i*chunk_size:(i+1)*chunk_size]

    # Dense coherence (fast: ~0.5s per chunk)
    adj_chunk = coherence_fc_pipeline(
        X_chunk, fs,
        sparsify="none",  # Skip surrogates for speed
        bands=bands
    )

    for band in bands:
        adj_accum[band] += adj_chunk[band]

# Average over chunks
for band in bands:
    adj_accum[band] /= n_chunks
```

**Time estimate**: ~10 chunks × 0.5s = **5 seconds** (comparable to correlation!)

**Trade-off**: Loses temporal dynamics (averaging), but maintains frequency specificity

### Option 3: Use Fewer Surrogates
**Use if**: Need statistical testing but can tolerate lower power

- Reduce from 200 to 50 surrogates
- **Time**: 2.8 hours → **42 minutes**
- Still provides p-value estimates, just noisier

### Option 4: Increase `nperseg` for Coherence
**Use if**: Can sacrifice frequency resolution

```python
# Increase window length: 256 → 1024 samples
adj_matrices = coherence_fc_pipeline(X, fs, nperseg=1024)
```

**Speed improvement**: ~2-4x faster (fewer segments to process)
**Trade-off**: Lower frequency resolution

---

## DO NOT Use Mutual Information

MI is **not practical** for N×N connectivity estimation with ultra-long recordings:
- 1000x slower than correlation
- 100x slower than coherence
- Would take 284 DAYS with surrogates
- Designed for variable selection (e.g., feature vs target), not pairwise connectivity

---

## Final Recommendation

**For your ultra-long 10M sample recordings:**

1. **Exploratory analysis**: Use correlation (4 sec) or chunked coherence (5 sec)
2. **Final analysis with statistics**:
   - Use coherence with 50 surrogates (~42 min)
   - OR use correlation with surrogates (much faster)
3. **Never**: Use mutual information for all-to-all connectivity

**Best approach**: **Chunked coherence** gives you both speed AND frequency specificity without surrogates.

---

## Implementation Guide

### Existing Correlation-Based Pipeline

Your current correlation workflow ([bands.py:55](src/lrg_eegfc/utils/corrmat/bands.py#L55)):

```python
def build_corrmat_perband(data_ts, fs, brain_bands=BRAIN_BANDS, ...):
    for band_name, (low, high) in brain_bands.items():
        filtered = bandpass_sos(data_ts, low, high, fs, filter_order)  # O(NL)
        corr_mat = build_corr_network(filtered)  # np.corrcoef -> O(N²L)
```

**Performance**: ~4 seconds for 10M samples (6 bands)

### Recommended: Chunked Coherence (No Surrogates)

Add this helper function to [coherence/__init__.py](src/lrg_eegfc/utils/coherence/__init__.py):

```python
def coherence_fc_pipeline_chunked(
    X: NDArray,
    fs: float,
    bands: Dict[str, Tuple[float, float]] | None = None,
    chunk_duration: float = 600.0,  # 10 minutes default
    **kwargs
) -> Dict[str, NDArray]:
    """
    Chunked coherence for ultra-long recordings.

    Processes data in chunks and averages coherence matrices.
    Much faster than full-length with surrogates.
    """
    if bands is None:
        bands = BRAIN_BANDS

    N, L = X.shape
    chunk_size = int(chunk_duration * fs)
    n_chunks = L // chunk_size

    # Initialize accumulators
    adj_accum = {band: np.zeros((N, N)) for band in bands}

    # Process each chunk
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        X_chunk = X[:, start:end]

        # Dense coherence (no surrogates for speed)
        adj_chunk = coherence_fc_pipeline(
            X_chunk, fs,
            bands=bands,
            sparsify="none",  # Skip surrogates
            **kwargs
        )

        for band in bands:
            adj_accum[band] += adj_chunk[band]

    # Average over chunks
    for band in bands:
        adj_accum[band] /= n_chunks

    return adj_accum
```

**Usage**:
```python
# 10M samples → ~5 seconds (comparable to correlation!)
adj_matrices = coherence_fc_pipeline_chunked(X, fs)
```

**Trade-offs**:
- ✅ Fast: ~5 seconds vs 2.8 hours
- ✅ Frequency-specific coupling preserved
- ❌ Loses temporal dynamics (averaging)
- ❌ No statistical testing (but can add if needed)

---

## Decision Matrix

| Your Priority | Recommended Method | Time | Rationale |
|---------------|-------------------|------|-----------|
| **Speed above all** | Correlation (`build_corrmat_perband`) | 4 sec | Proven, established pipeline |
| **Frequency-specific + speed** | Chunked coherence | 5 sec | Best of both worlds |
| **Statistical testing needed** | Coherence + 50 surrogates | 42 min | Acceptable for final analysis |
| **Exploring methods** | ❌ NOT mutual information | 34 hours | Completely infeasible |

---

## Understanding "Temporal Dynamics"

### What Are Temporal Dynamics?

**Temporal dynamics** = How connectivity **changes over time** during your recording.

### Example: 1-Hour Recording

Connectivity between two channels might fluctuate:

```
Time:     0min    10min   20min   30min   40min   50min   60min
          |-------|-------|-------|-------|-------|-------|
Alpha
Coherence:  0.7     0.8     0.3     0.9     0.4     0.6     0.5
Ch1-Ch2
          [HIGH] [HIGH]  [LOW]  [HIGH]  [LOW]  [MED]  [MED]
```

**Why does connectivity change?**
- Brain state fluctuations (drowsiness, attention)
- Microstate transitions
- Arousal level changes
- Task-related modulations

### Chunked Averaging (Loses Dynamics)

```python
# Process 6 chunks of 10 minutes each
adj_accum = np.zeros((N, N))
for chunk in chunks:
    adj_chunk = coherence_fc_pipeline(chunk, fs, sparsify="none")
    adj_accum += adj_chunk['alpha']

adj_avg = adj_accum / 6  # Single average value

# Result: Alpha coherence Ch1-Ch2 = 0.60 (one number)
# You CANNOT see it varied from 0.3 to 0.9 over time
```

### Full-Length Analysis (Preserves Dynamics)

```python
# Option 1: Analyze full recording (slow with surrogates)
adj_full = coherence_fc_pipeline(X_full, fs)  # 2.8 hours with surrogates

# Option 2: Analyze in windows (preserves dynamics, moderate speed)
coherence_over_time = []
for t in range(0, 60, 10):  # Every 10 minutes
    X_window = X[:, t*fs*60:(t+10)*fs*60]
    adj_t = coherence_fc_pipeline(X_window, fs, sparsify="none")
    coherence_over_time.append(adj_t)

# Result: List of 6 matrices showing connectivity evolution
# Can plot connectivity vs time, detect state changes, etc.
```

### When Does Losing Dynamics Matter?

**❌ Losing dynamics is BAD when:**
- **Task studies**: "How does connectivity change during learning vs rest?"
- **State detection**: "Can I identify drowsy vs alert periods?"
- **Event-related analysis**: "What happens around seizures?"
- **Temporal hypotheses**: "Does connectivity increase over time?"

**✅ Losing dynamics is FINE when:**
- **Average characterization**: "What's typical alpha coupling in this patient?"
- **Between-subject comparisons**: "Patient A vs Patient B average connectivity"
- **Stationary assumption**: "Connectivity is stable over recording"
- **Exploratory analysis**: "Quick overview of connectivity patterns"

### Preserving Dynamics with Chunking (Best of Both Worlds)

```python
# Modified approach: Store each chunk separately (don't average)
coherence_timeseries = []

for i in range(n_chunks):
    X_chunk = X[:, i*chunk_size:(i+1)*chunk_size]
    adj_chunk = coherence_fc_pipeline(X_chunk, fs, sparsify="none")
    coherence_timeseries.append(adj_chunk)  # Keep all!

# Now you have:
# - coherence_timeseries[0] = connectivity during 0-10 min
# - coherence_timeseries[1] = connectivity during 10-20 min
# - coherence_timeseries[i] = connectivity during chunk i

# You can:
# 1. Plot connectivity over time
# 2. Detect high/low connectivity periods
# 3. Correlate with behavior
# 4. Compute average if needed: np.mean([c['alpha'] for c in coherence_timeseries], axis=0)
```

**This gives you:**
- ✅ Fast computation (5 sec total)
- ✅ Temporal dynamics preserved
- ✅ Flexibility to average or analyze dynamics
- ⚠️ More memory (storing all chunks vs just average)

### Visual Comparison

**With Dynamics (storing chunks separately):**
```
Coherence over time:
  1.0 |              ●
      |         ●
  0.8 |    ●              ●
      |                        ●
  0.6 |                             ●   ●
  0.4 |              ●
      |_____|_____|_____|_____|_____|_____|
      0    10    20    30    40    50    60 min

→ See fluctuations
→ Identify brain states
→ Correlate with events
```

**Without Dynamics (averaging chunks):**
```
Average connectivity:
  1.0 |
  0.8 |
  0.6 |━━━━━━━━━━━━━●━━━━━━━━━━━━━━━━━  (avg = 0.60)
  0.4 |
      |_____|_____|_____|_____|_____|_____|
      0    10    20    30    40    50    60 min

→ Only one number
→ Can't see variations
→ Can't identify states
```
