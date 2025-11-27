"""Magnitude-squared coherence (MSC) computation using vectorized Welch's method.

This module implements a fast, vectorized version of Welch's method that computes
coherence for all channel pairs at once, similar to how np.corrcoef works.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.signal import get_window


__all__ = [
    'compute_msc_welch',
    'band_average_msc',
]


def compute_msc_welch(
    X: NDArray,
    fs: float,
    nperseg: int = 256,
    noverlap: int | None = None,
) -> Tuple[NDArray, NDArray]:
    """
    Compute magnitude-squared coherence using vectorized Welch's method (fast).

    This function uses a vectorized implementation of Welch's method that computes
    coherence for all channel pairs simultaneously, similar to how np.corrcoef works.
    This is much faster than calling scipy.signal.coherence for each pair.

    Parameters
    ----------
    X : NDArray
        Input data of shape (N, L) where N is the number of channels and L is the
        number of samples
    fs : float
        Sampling frequency in Hz
    nperseg : int, optional
        Length of each segment (window) in samples (default: 256)
    noverlap : int, optional
        Number of overlapping samples between segments. If None, uses nperseg // 2
        (default: None)

    Returns
    -------
    freqs : NDArray
        Array of frequencies of shape (F,)
    Coh : NDArray
        Magnitude-squared coherence of shape (N, N, F) where Coh[i, j, f] is
        the coherence between channels i and j at frequency f

    Notes
    -----
    Vectorized Welch's method:
    1. Divide data into overlapping segments
    2. Apply Hanning window to all channels simultaneously
    3. FFT all channels for each segment
    4. Compute cross-spectral densities via matrix multiplication
    5. Average over segments to get robust CSD estimates
    6. Compute MSC from CSD matrix

    This is 10-100x faster than calling scipy.signal.coherence per pair
    and achieves similar speed to np.corrcoef.
    """
    N, L = X.shape

    if noverlap is None:
        noverlap = nperseg // 2

    # Calculate number of segments
    step = nperseg - noverlap
    n_segments = (L - noverlap) // step

    # Frequency array (one-sided for real signals)
    freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
    F = len(freqs)

    # Get Hanning window
    window = get_window('hann', nperseg)

    # Normalization factor for the window
    scale = 1.0 / (fs * (window * window).sum())

    # Initialize cross-spectral density matrix
    CSD = np.zeros((N, N, F), dtype=complex)

    # Process each segment
    for seg_idx in range(n_segments):
        start = seg_idx * step
        end = start + nperseg

        if end > L:
            break

        # Extract segment for all channels: shape (N, nperseg)
        segment = X[:, start:end]

        # Apply window to all channels: (N, nperseg) * (nperseg,) -> (N, nperseg)
        windowed = segment * window

        # FFT all channels: shape (N, F)
        fft_data = np.fft.rfft(windowed, n=nperseg, axis=1)

        # Compute cross-spectral density for all pairs
        # CSD[i,j,f] = FFT_i[f] * conj(FFT_j[f])
        # Use outer product: fft_data[:, None, :] (N,1,F) * conj(fft_data[None, :, :]) (1,N,F)
        # Result: (N, N, F)
        CSD += fft_data[:, None, :] * np.conj(fft_data[None, :, :])

    # Average over segments and apply scaling
    CSD *= scale / n_segments

    # Compute magnitude-squared coherence
    # Coh[i,j,f] = |CSD[i,j,f]|^2 / (CSD[i,i,f] * CSD[j,j,f])

    # Extract auto-spectra (PSD) along diagonal
    # CSD[i, i, :] for all i gives the PSD for each channel
    PSD = np.real(np.array([CSD[i, i, :] for i in range(N)]))  # shape: (N, F)

    # Compute MSC for all pairs using vectorized operations
    # PSD[i,f] * PSD[j,f] for all i,j pairs
    # PSD[:, None, :] (N,1,F) * PSD[None, :, :] (1,N,F) = (N,N,F)
    denom = PSD[:, None, :] * PSD[None, :, :]

    # Compute |CSD|^2
    CSD_mag_sq = np.abs(CSD) ** 2

    # Coherence: |CSD|^2 / (PSD_i * PSD_j)
    # Use np.divide with where to handle division by zero
    Coh = np.divide(CSD_mag_sq, denom, out=np.zeros((N, N, F)), where=denom > 0)

    # Set diagonal to 1 (coherence with self)
    for i in range(N):
        Coh[i, i, :] = 1.0

    return freqs, Coh


def band_average_msc(
    Coh: NDArray,
    freqs: NDArray,
    bands: Dict[str, Tuple[float, float]],
) -> Dict[str, NDArray]:
    """
    Average MSC over specified frequency bands.

    Parameters
    ----------
    Coh : NDArray
        Magnitude-squared coherence of shape (N, N, F)
    freqs : NDArray
        Array of frequencies of shape (F,)
    bands : Dict[str, Tuple[float, float]]
        Dictionary mapping band names to (fmin, fmax) tuples in Hz

    Returns
    -------
    W_bands : Dict[str, NDArray]
        Dictionary mapping band names to band-averaged MSC matrices of shape (N, N)

    Notes
    -----
    For each band b = [fmin, fmax]:

    W_ij^(b) = mean(Coh_ij(f) for f in [fmin, fmax])

    This yields a dense, non-negative, symmetric adjacency matrix for each band.
    """
    N = Coh.shape[0]
    W_bands = {}

    for band_name, (fmin, fmax) in bands.items():
        # Find frequency indices in band
        band_mask = (freqs >= fmin) & (freqs <= fmax)

        if not np.any(band_mask):
            # No frequencies in this band
            W_bands[band_name] = np.zeros((N, N))
            continue

        # Average coherence over band
        W_band = np.mean(Coh[:, :, band_mask], axis=2)
        W_bands[band_name] = W_band

    return W_bands
