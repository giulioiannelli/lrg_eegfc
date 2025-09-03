from ..shared import *

from lrgsglib.utils import compute_threshold_stats_fast

def find_exact_detachment_threshold(corr_mat):
    """
    Find the EXACT threshold where the first node detaches from giant component.
    Uses binary search on sorted edge weights for maximum efficiency.
    """
    # Get all edge weights (upper triangular, no self-loops)
    n = corr_mat.shape[0]
    triu_indices = np.triu_indices(n, k=1)
    edge_weights = np.abs(corr_mat[triu_indices])
    
    # Sort edge weights in ascending order
    sorted_weights = np.sort(edge_weights)
    
    # Binary search for the critical threshold
    left, right = 0, len(sorted_weights) - 1
    
    while left < right:
        mid = (left + right) // 2
        threshold = sorted_weights[mid]
        
        # Create adjacency matrix with threshold
        adj_matrix = np.abs(corr_mat) >= threshold
        np.fill_diagonal(adj_matrix, False)  # Remove self-loops
        
        # Quick connectivity check using matrix powers
        # If graph is connected, adj_matrix^n should be all non-zero
        reach_matrix = adj_matrix.copy().astype(int)
        for _ in range(n-1):
            reach_matrix = np.dot(reach_matrix, adj_matrix.astype(int))
            if np.any(reach_matrix):
                break
        
        # Check if all nodes can reach each other
        connected = np.all(reach_matrix + reach_matrix.T + np.eye(n) > 0)
        
        if connected:
            left = mid + 1
        else:
            right = mid
    
    return sorted_weights[left] if left < len(sorted_weights) else sorted_weights[-1]

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

def find_threshold_jumps(G_filt):
    Th, Einf, Pinf = compute_threshold_stats_fast(G_filt)
    Pinf_diff = np.diff(Pinf)
    jumps = np.where(Pinf_diff != 0)[0]
    return Th, jumps


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