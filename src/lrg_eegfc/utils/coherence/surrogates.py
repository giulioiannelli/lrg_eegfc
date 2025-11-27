"""Surrogate data generation for coherence null model estimation."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from .msc import compute_msc_welch, band_average_msc


__all__ = [
    'circular_shift_surrogates',
    'surrogate_msc_null',
]


def circular_shift_surrogates(
    X: NDArray,
    n_surrogates: int,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """
    Generate surrogate data via circular shifts of each channel.

    This method preserves each channel's spectrum and amplitude distribution but
    removes cross-channel coupling by shifting each channel by a random amount.

    Parameters
    ----------
    X : NDArray
        Input data of shape (N, L) where N is the number of channels and L is the
        number of samples
    n_surrogates : int
        Number of surrogate datasets to generate
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, uses default RNG.

    Returns
    -------
    X_surr : NDArray
        Surrogate data of shape (n_surrogates, N, L) where X_surr[r, i, :] is the
        r-th surrogate of channel i

    Notes
    -----
    For each surrogate r and channel i:

    x_i^(r)(t) = x_i((t + Δ_i^(r)) mod L)

    where Δ_i^(r) is a random integer in [0, L-1], independent per channel.
    This preserves each channel's spectrum and amplitude distribution but removes
    cross-channel coupling.
    """
    if rng is None:
        rng = np.random.default_rng()

    N, L = X.shape
    X_surr = np.zeros((n_surrogates, N, L), dtype=X.dtype)

    for r in range(n_surrogates):
        for i in range(N):
            # Random shift for this channel
            shift = rng.integers(0, L)

            # Circular shift
            X_surr[r, i, :] = np.roll(X[i, :], shift)

    return X_surr


def surrogate_msc_null(
    X: NDArray,
    fs: float,
    bands: Dict[str, Tuple[float, float]],
    n_surrogates: int,
    nperseg: int = 256,
    noverlap: int | None = None,
    rng: np.random.Generator | None = None,
) -> Dict[str, NDArray]:
    """
    Compute null distribution of band-averaged MSC using circular shift surrogates.

    Parameters
    ----------
    X : NDArray
        Input data of shape (N, L)
    fs : float
        Sampling frequency in Hz
    bands : Dict[str, Tuple[float, float]]
        Dictionary mapping band names to (fmin, fmax) tuples
    n_surrogates : int
        Number of surrogates to generate
    nperseg : int, optional
        Length of each segment for coherence computation (default: 256)
    noverlap : int, optional
        Number of overlapping samples (default: nperseg // 2)
    rng : np.random.Generator, optional
        Random number generator for reproducibility

    Returns
    -------
    W_null : Dict[str, NDArray]
        Dictionary mapping band names to null MSC matrices of shape (n_surrogates, N, N)
        where W_null[band][r, i, j] is the MSC between channels i and j in band for
        surrogate r

    Notes
    -----
    This function generates circular shift surrogates ONE AT A TIME to avoid excessive
    memory usage. For large datasets, this is critical to prevent OOM errors.

    Memory usage: O(N × L) instead of O(n_surrogates × N × L)
    """
    if rng is None:
        rng = np.random.default_rng()

    N, L = X.shape

    # Initialize null distribution dict
    W_null = {band_name: np.zeros((n_surrogates, N, N)) for band_name in bands}

    # Compute MSC for each surrogate (one at a time to save memory!)
    for r in range(n_surrogates):
        # Generate ONE surrogate at a time
        X_surr_r = np.empty_like(X)
        for i in range(N):
            shift = rng.integers(0, L)
            X_surr_r[i, :] = np.roll(X[i, :], shift)

        # Compute MSC for this surrogate
        freqs, Coh = compute_msc_welch(
            X_surr_r,
            fs,
            nperseg=nperseg,
            noverlap=noverlap,
        )

        # Band average
        W_bands_r = band_average_msc(Coh, freqs, bands)

        # Store in null distribution
        for band_name in bands:
            W_null[band_name][r] = W_bands_r[band_name]

        # X_surr_r goes out of scope and gets garbage collected

    return W_null
