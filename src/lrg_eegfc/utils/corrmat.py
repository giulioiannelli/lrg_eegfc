from ..shared import *

def build_corr_network(timeseries, filter_type=None, threshold=None, zero_diagonal=True):
    """
    Compute a thresholded correlation network from multivariate time series.

    Parameters
    ----------
    timeseries : array-like, shape (N, T)
        Time series data for N signals of length T.
    filter_type : {'abs', None}, optional
        If 'abs', use absolute values of correlations before thresholding.
        Default is None (no filtering).
    threshold : float, optional
        Minimum correlation value; entries below this are set to zero.
        Default is 0.0.
    zero_diagonal : bool, optional
        If True, set diagonal entries of the output matrix to zero.
        Default is True.

    Returns
    -------
    C : ndarray, shape (N, N)
        Thresholded correlation matrix (adjacency), with optional absolute
        filtering and zeroed diagonal.
    """
    C = np.corrcoef(timeseries)
    if filter_type == 'abs':
        C = np.abs(C)
    if threshold is not None:
        C[C < threshold] = 0
    if zero_diagonal:
        np.fill_diagonal(C, 0)
    return C

def clean_correlation_matrix(X: np.ndarray, rowvar: bool = True):
    """
    Given data matrix X (T×N), returns:
      - C_clean: denoised correlation matrix
      - eigvals: all eigenvalues of the empirical C
      - lambda_min, lambda_max: Marchenko–Pastur bounds
      - signal_mask: boolean mask of eigenvalues > lambda_max
    """
    N, T = X.shape
    C = np.corrcoef(X, rowvar=rowvar)
    eigvals, eigvecs = np.linalg.eigh(C)

    Q = T / N
    lambda_min = (1 - np.sqrt(1/Q))**2
    lambda_max = (1 + np.sqrt(1/Q))**2

    signal_mask = eigvals > lambda_max
    V = eigvecs[:, signal_mask]
    L = np.diag(eigvals[signal_mask])

    C_clean = V @ L @ V.T
    np.fill_diagonal(C_clean, 1.0)

    return C_clean, eigvals, eigvecs, lambda_min, lambda_max, signal_mask