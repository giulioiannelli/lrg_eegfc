"""Magnitude-squared coherence (MSC) computation using vectorized Welch's method.

This module implements a fast, vectorized version of Welch's method that computes
coherence for all channel pairs at once, similar to how np.corrcoef works.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.signal import get_window
from numpy.lib.stride_tricks import sliding_window_view


__all__ = [
    'compute_msc_welch',
    'band_average_msc',
]


def compute_msc_welch(
    X: NDArray,
    fs: float,
    nperseg: int = 256,
    noverlap: int | None = None,
    batch_size: int = 64,
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
    segments = sliding_window_view(X, window_shape=nperseg, axis=1)[:, ::step, :]
    n_segments = segments.shape[1]
    if n_segments == 0:
        raise ValueError("nperseg larger than signal length")

    # Frequency array (one-sided for real signals)
    freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
    F = len(freqs)

    # Get Hanning window
    window = get_window('hann', nperseg).astype(X.dtype, copy=False)

    # Normalization factor for the window (includes averaging over segments)
    scale = 1.0 / (fs * (window * window).sum() * n_segments)

    # Initialize cross-spectral density matrix
    CSD = np.zeros((N, N, F), dtype=complex)

    # Process segments in batches to reduce Python overhead
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    for start in range(0, n_segments, batch_size):
        stop = start + batch_size
        # (N, B, nperseg)
        block = segments[:, start:stop, :] * window[None, None, :]
        # FFT over the last axis -> (N, B, F)
        fft_block = np.fft.rfft(block, n=nperseg, axis=2)
        # Accumulate cross-spectral density for the batch
        CSD += np.einsum('nbf,mbf->nmf', fft_block, np.conj(fft_block), optimize='greedy')

    # Average over segments and apply scaling
    CSD *= scale

    # Compute magnitude-squared coherence
    PSD = np.real(np.array([CSD[i, i, :] for i in range(N)]))  # shape: (N, F)

    denom = PSD[:, None, :] * PSD[None, :, :]

    CSD_mag_sq = np.abs(CSD) ** 2

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
