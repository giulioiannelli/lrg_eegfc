from ..shared import *
from ..config.const import *
from typing import Optional, Tuple, Union
import networkx as nx

from lrgsglib.utils import compute_threshold_stats_fast, bandpass_sos

__all__ = [
    'apply_threshold_filter',
    'build_corr_network',
    'find_exact_detachment_threshold',
    'find_threshold_jumps',
    'build_corrmat_perband',
    'clean_correlation_matrix'
]


def apply_threshold_filter(matrix: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply threshold filtering to a matrix by setting values below threshold to zero.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix to filter.
    threshold : float
        Minimum value; entries below this are set to zero.
    
    Returns
    -------
    np.ndarray
        Filtered matrix with values below threshold set to zero.
    """
    filtered_matrix = matrix.copy()
    filtered_matrix[filtered_matrix < threshold] = 0
    return filtered_matrix


def build_corr_network(
        timeseries: np.ndarray, 
        filter_type: Optional[str] = None, 
        threshold: Optional[float] = None, 
        zero_diagonal: bool = True, 
        spectral_cleaning: bool = False) -> np.ndarray:
    """
    Compute a thresholded correlation network from multivariate time series.

    Parameters
    ----------
    timeseries : np.ndarray, shape (N, T)
        Time series data for N signals of length T.
    filter_type : {'abs'} or None, optional
        Apply absolute values if 'abs'. Default is None.
    threshold : float or None, optional
        Minimum correlation value; below this are set to zero. Default is None.
    zero_diagonal : bool, optional
        Whether to zero the diagonal entries. Default is True.
    spectral_cleaning : bool, optional
        Whether to apply RMT-based spectral denoising. Default is False.

    Returns
    -------
    np.ndarray, shape (N, N)
        Processed correlation matrix.
        
    Notes
    -----
    Processing order: correlation → filter_type → threshold → spectral_cleaning 
    → zero_diagonal. Spectral cleaning uses Marchenko-Pastur bounds to remove 
    noise eigenvalues.
    """
    # Compute correlation matrix
    C = np.corrcoef(timeseries)
    
    # Apply filters in sequence
    if filter_type == 'abs':
        C = np.abs(C)
    
    if threshold is not None:
        C = apply_threshold_filter(C, threshold)
    
    # Apply spectral cleaning before zeroing diagonal
    if spectral_cleaning:
        C_clean, _, _, _, _, _ = clean_correlation_matrix(C.T, rowvar=False)
        C = C_clean
    
    if zero_diagonal:
        np.fill_diagonal(C, 0)
    
    return C



























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



def find_threshold_jumps(
        G_filt: nx.Graph, 
        return_stats: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], 
                                             Tuple[np.ndarray, np.ndarray, 
                                                   np.ndarray, np.ndarray]]:
    """
    Find threshold jumps in network percolation statistics.
    
    Analyzes percolation behavior by computing threshold statistics and identifying
    discontinuities (jumps) in the giant component fraction.
    
    Parameters
    ----------
    G_filt : nx.Graph
        Weighted NetworkX graph with edge weights representing correlations.
    return_stats : bool, optional
        Whether to return full threshold statistics. Default is False.
    
    Returns
    -------
    Th : np.ndarray
        Logarithmically spaced threshold values between min and max edge weights.
    jumps : np.ndarray
        Indices where discontinuities occur in giant component size (phase transitions).
    Einf : np.ndarray, optional
        Fraction of edges in giant component vs threshold. Only if return_stats=True.
    Pinf : np.ndarray, optional
        Fraction of nodes in giant component vs threshold. Only if return_stats=True.
    
    Notes
    -----
    Uses fast Union-Find algorithm. First jump (jumps[0]) commonly used for threshold selection.
    Critical threshold = Th[jumps[0]].
    """
    # Compute percolation statistics using fast Union-Find algorithm
    Th, Einf, Pinf = compute_threshold_stats_fast(G_filt)
    
    # Find discontinuities in giant component size
    Pinf_diff = np.diff(Pinf)
    jumps = np.where(Pinf_diff != 0)[0]
    
    if return_stats:
        return Th, jumps, Einf, Pinf
    else:
        return Th, jumps


def _select_threshold_from_jumps(corr_mat_initial, jump_index=0, band_name=""):
    """
    Helper function to select threshold based on jump analysis of correlation matrix.
    
    Parameters
    ----------
    corr_mat_initial : np.ndarray
        Initial correlation matrix for threshold analysis
    jump_index : int, optional
        Index of the jump to use for threshold selection (0-based).
    band_name : str, optional
        Name of the band for warning messages.
    
    Returns
    -------
    dict
        Dictionary containing threshold selection results
    """
    import networkx as nx
    import warnings
    
    # Find detachment threshold
    Th, jumps = find_threshold_jumps(nx.from_numpy_array(corr_mat_initial))
    
    # Validate jump_index is within bounds
    if len(jumps) == 0:
        warnings.warn(
            f"Band '{band_name}': No jumps found in percolation analysis. "
            f"Using minimum threshold.",
            UserWarning
        )
        chosen_jump = 0
        effective_jump_index = 0
        chosen_threshold = Th[0] if len(Th) > 0 else 0.0
    elif jump_index >= len(jumps):
        warnings.warn(
            f"Band '{band_name}': jump_index={jump_index} exceeds available jumps "
            f"({len(jumps)} jumps found). Using last available jump (index {len(jumps)-1}).",
            UserWarning
        )
        chosen_jump = jumps[-1]
        effective_jump_index = len(jumps) - 1
        chosen_threshold = Th[chosen_jump]
    else:
        chosen_jump = jumps[jump_index]
        effective_jump_index = jump_index
        chosen_threshold = Th[chosen_jump]
    
    return {
        'jumps': jumps,
        'chosen_jump': chosen_jump,
        'chosen_threshold': chosen_threshold,
        'jump_index': effective_jump_index,
        'expected_components': effective_jump_index + 1
    }


def build_corrmat_perband(data_ts, fs, bandpass_func: Callable = bandpass_sos, brain_bands: dict = BRAIN_BANDS, 
                          return_jump_info=False, apply_threshold_filtering=True, 
                          corr_network_params: dict={'threshold': 0}, jump_index: int = 0):
    """
    Build correlation matrices per frequency band.
    
    Parameters:
    -----------
    data_ts : array-like
        Time series data
    fs : float
        Sampling frequency
    brain_bands : dict
        Dictionary of brain bands with (low, high) frequency pairs.
    bandpass_func : callable
        Bandpass filtering function (e.g., bandpass_sos)
    return_jump_info : bool, optional
        If True, also returns the jump information for threshold selection.
        Default is False.
    apply_threshold_filtering : bool, optional
        If True, applies automatic threshold filtering based on jump detection.
        If False, uses only the parameters from corr_network_params.
        Default is True.
    corr_network_params : dict, optional
        Parameters to pass to build_corr_network function.
        Default is {'threshold': 0} if not provided.
        Common parameters: threshold, filter_type, zero_diagonal.
    jump_index : int, optional
        Index of the jump to use for threshold selection (0-based).
        jump_index=0 means 1 giant component, jump_index=1 means 2 components, etc.
        Default is 0 (first jump, single giant component).
    
    Returns:
    --------
    corr_mat_band : dict
        Dictionary with band names as keys and correlation matrices as values
    band_jump_info : dict (optional)
        Dictionary with jump information for each band (only if return_jump_info=True)
        
    Notes:
    ------
    - When apply_threshold_filtering=False, threshold statistics are not computed,
      improving performance when jump information is not needed.
    - corr_network_params allows fine-tuning of correlation matrix computation.
    - If both apply_threshold_filtering=True and corr_network_params has 'threshold',
      the automatic threshold from jump detection overrides the manual threshold.
    - The function validates that the resulting network has the expected number of
      connected components (jump_index + 1). Issues a warning if validation fails.
    """
    import networkx as nx
    import warnings
    
    corr_mat_band = {}
    band_jump_info = {}
    
    for band_name, (low, high) in brain_bands.items():
        # Apply bandpass filter
        filter_data = bandpass_func(data_ts, low, high, fs, 1)
        
        if apply_threshold_filtering:
            # Build initial correlation network for threshold analysis
            corr_mat_initial = build_corr_network(filter_data, **corr_network_params)
            
            # Select threshold using jump analysis
            threshold_info = _select_threshold_from_jumps(corr_mat_initial, jump_index, band_name)
            
            # Store jump information
            band_jump_info[band_name] = threshold_info.copy()
            
            # Build final correlation network with chosen threshold
            # Override threshold parameter with automatically detected one
            final_params = corr_network_params.copy()
            final_params['threshold'] = threshold_info['chosen_threshold']
            corr_mat = build_corr_network(filter_data, **final_params)
            
            # Validate number of connected components
            G_final = nx.from_numpy_array(corr_mat)
            # Remove isolated nodes (nodes with no edges above threshold)
            G_final.remove_nodes_from(list(nx.isolates(G_final)))
            
            if G_final.number_of_nodes() > 0:  # Only check if graph has nodes
                n_components = nx.number_connected_components(G_final)
                expected_components = threshold_info['expected_components']
                
                if n_components != expected_components:
                    warnings.warn(
                        f"Band '{band_name}': Expected {expected_components} connected components "
                        f"(jump_index={threshold_info['jump_index']}) but found {n_components} components. "
                        f"Network topology may be unexpected at threshold {threshold_info['chosen_threshold']:.6f}.",
                        UserWarning
                    )
                
                # Store validation info
                band_jump_info[band_name]['actual_components'] = n_components
                band_jump_info[band_name]['validation_passed'] = (n_components == expected_components)
            else:
                warnings.warn(
                    f"Band '{band_name}': Threshold {threshold_info['chosen_threshold']:.6f} resulted in no connected nodes. "
                    f"All correlations below threshold.",
                    UserWarning
                )
                band_jump_info[band_name]['actual_components'] = 0
                band_jump_info[band_name]['validation_passed'] = False
                
        else:
            # Build correlation network with provided parameters only
            # No threshold statistics computation
            corr_mat = build_corr_network(filter_data, **corr_network_params)
            
            # Set empty jump info if needed for return consistency
            if return_jump_info:
                band_jump_info[band_name] = {
                    'jumps': None,
                    'chosen_jump': None,
                    'chosen_threshold': corr_network_params.get('threshold', 0),
                    'jump_index': None,
                    'expected_components': None,
                    'actual_components': None,
                    'validation_passed': None
                }
        
        corr_mat_band[band_name] = corr_mat
    
    if return_jump_info:
        return corr_mat_band, band_jump_info
    else:
        return corr_mat_band


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