from ..shared import *
from ..config.const import *
from typing import Optional, Tuple, Union
import networkx as nx
from scipy.spatial.distance import squareform

from lrgsglib.utils import compute_threshold_stats_fast, bandpass_sos, \
    get_giant_component_leftoff
from lrgsglib.utils.lrg import compute_laplacian_properties, \
    compute_optimal_threshold, compute_normalized_linkage

__all__ = [
    'apply_threshold_filter',
    'build_corr_network',
    'process_network_for_phase',
    'find_exact_detachment_threshold',
    'find_threshold_jumps',
    'build_corrmat_perband',
    'build_corrmat_single_band',
    'clean_correlation_matrix'
]
#
def apply_threshold_filter(matrix: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply threshold filtering to a matrix by setting values below threshold to 
    zero.
    
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
#
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
    
    # Check for non-finite values in correlation matrix
    if not np.all(np.isfinite(C)):
        import warnings
        warnings.warn(
            "Correlation matrix contains non-finite values. "
            "This may be due to constant or invalid time series. "
            "Replacing NaN/Inf with zeros.",
            RuntimeWarning
        )
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply spectral cleaning before zeroing diagonal
    if spectral_cleaning:
        C_clean, _, _, _, _, _ = clean_correlation_matrix(C.T, rowvar=False)
        C = C_clean

    # Apply filters in sequence
    if filter_type == 'abs':
        C = np.abs(C)
    
    if threshold is not None:
        C = apply_threshold_filter(C, threshold)
    
    if zero_diagonal:
        np.fill_diagonal(C, 0)
    
    return C
#
def process_network_for_phase(
        data_pat_phase_ts: np.ndarray,
        fs: float,
        band_name: str,
        correlation_protocol: dict,
        all_labels,
        jump_index_to_use: int = 0,
        scaling_factor: float = 0.98,
        linkage_method: str = 'ward',
        filter_order: int = 4) -> Tuple[Optional[nx.Graph], 
                                               Optional[dict], 
                                               Optional[np.ndarray], 
                                               Optional[float], 
                                               Optional[np.ndarray],
                                               Optional[np.ndarray]]:
    """
    Process network analysis for a single phase and frequency band.
    
    Builds correlation matrices, creates filtered networks, extracts the giant 
    component, and performs hierarchical clustering analysis with optimal 
    threshold detection.
    
    Parameters
    ----------
    data_pat_phase_ts : np.ndarray
        Time series data for the specific patient phase.
    fs : float
        Sampling frequency of the time series data.
    band_name : str
        Name of the frequency band to analyze.
    correlation_protocol : dict
        Parameters for correlation network construction (passed to 
        build_corrmat_perband).
    all_labels : pd.Series or dict-like
        Labels mapping for network nodes (typically brain region names).
    jump_index_to_use : int, optional
        Index of the threshold jump to use for network filtering (0-based). 
        Default is 0.
    scaling_factor : float, optional
        Scaling factor for optimal threshold computation. Default is 0.98.
    linkage_method : str, optional
        Method for hierarchical clustering linkage. Default is 'ward'.
    filter_order : int, optional
        Filter order for bandpass filtering. Default is 4.
    
    Returns
    -------
    G_giant : nx.Graph or None
        Giant component of the filtered network, or None if no valid network 
        found.
    labeldict : dict or None
        Dictionary mapping node indices to labels for nodes in giant component.
    lnkgM : np.ndarray or None
        Linkage matrix from hierarchical clustering.
    clTh : float or None
        Optimal clustering threshold.
    corr_mat : np.ndarray or None
        Correlation matrix for the specified frequency band.
    dists : np.ndarray or None
        Distance matrix computed from Laplacian properties.
        
    Notes
    -----
    Returns (None, None, None, None, None, None) if the network has no nodes 
    after filtering or if the giant component is empty. The function performs 
    automatic threshold filtering based on percolation analysis using the 
    specified jump index.
    """
    # Build correlation matrices per band
    corr_mat = build_corrmat_single_band(
        data_pat_phase_ts, fs, BRAIN_BANDS[band_name],
        jump_index=jump_index_to_use, corr_network_params=correlation_protocol,
        filter_order=filter_order
    )
    
    # Extract correlation matrix for the specified band
    
    # Create network and remove zero-weight edges and isolated nodes
    G = nx.from_numpy_array(corr_mat)
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) 
                         if d['weight'] == 0])
    G.remove_nodes_from(list(nx.isolates(G)))
    
    # Check if network has any nodes after filtering
    if len(G.nodes()) == 0:
        return None, None, None, None, None, None
    
    # Extract giant component
    G_giant, removed_nodes = get_giant_component_leftoff(G)
    
    # Check if giant component is empty
    if len(G_giant.nodes()) == 0:
        return None, None, None, None, None, None
    
    # Create filtered labels dictionary for nodes in giant component
    labeldict = {k: v for k, v in all_labels.to_dict().items() 
                 if k in G_giant.nodes()}
    
    # Compute Laplacian properties and hierarchical clustering
    spect, L, rho, Trho, tau = compute_laplacian_properties(G_giant, tau=None)
    
    # Check for non-finite values in the resistance distance matrix
    if not np.all(np.isfinite(Trho)):
        import warnings
        warnings.warn(
            f"Band '{band_name}': Non-finite values detected in resistance distance matrix. "
            f"This may indicate numerical instabilities in the correlation matrix or Laplacian computation. "
            f"Skipping this band for this phase.",
            RuntimeWarning
        )
        return None, None, None, None, None, None
    
    dists = squareform(Trho)
    
    # Additional check for the condensed distance matrix
    if not np.all(np.isfinite(dists)):
        import warnings
        warnings.warn(
            f"Band '{band_name}': Non-finite values detected in condensed distance matrix. "
            f"Skipping this band for this phase.",
            RuntimeWarning
        )
        return None, None, None, None, None, None
    
    lnkgM, label_list, _ = compute_normalized_linkage(dists, G_giant, 
                                                      method=linkage_method)
    clTh, *_ = compute_optimal_threshold(lnkgM, 
                                         scaling_factor=scaling_factor)
    
    return G_giant, labeldict, lnkgM, clTh, corr_mat, dists


























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
                          corr_network_params: dict={'threshold': 0}, jump_index: int = 0, filter_order: int = 4):
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
    filter_order : int, optional
        Filter order for bandpass filtering. Default is 4.
    
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
        # Adjust filter order for narrow bands to avoid numerical instabilities
        # For very narrow bands (< 1% of Nyquist), use order 1
        nyquist = fs / 2.0
        bandwidth_ratio = (high - low) / nyquist
        if bandwidth_ratio < 0.01 and filter_order > 1:
            adjusted_order = 1
            warnings.warn(
                f"Band '{band_name}': Reducing filter order from {filter_order} to {adjusted_order} "
                f"for narrow band ({low}-{high} Hz, {bandwidth_ratio:.4f} of Nyquist) "
                f"to avoid numerical instabilities.",
                RuntimeWarning
            )
        else:
            adjusted_order = filter_order
        
        # Apply bandpass filter
        filter_data = bandpass_func(data_ts, low, high, fs, adjusted_order)
        
        # Check if filtered data contains non-finite values
        if not np.all(np.isfinite(filter_data)):
            warnings.warn(
                f"Band '{band_name}': Bandpass filtering produced non-finite values. "
                f"Skipping this band.",
                RuntimeWarning
            )
            continue
        
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


def build_corrmat_single_band(
        data_ts: np.ndarray, 
        fs: float, 
        band: Tuple[float, float],
        bandpass_func: Callable = bandpass_sos,
        return_jump_info: bool = False,
        apply_threshold_filtering: bool = True,
        corr_network_params: Optional[dict] = None,
        jump_index: int = 0,
        band_name: str = "",
        filter_order: int = 4) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Build correlation matrix for a single frequency band.
    
    Parameters
    ----------
    data_ts : np.ndarray
        Time series data.
    fs : float
        Sampling frequency.
    band : tuple of (float, float)
        Frequency band as (low, high) frequency pair.
    bandpass_func : callable, optional
        Bandpass filtering function (e.g., bandpass_sos). Default is bandpass_sos.
    return_jump_info : bool, optional
        If True, also returns the jump information for threshold selection.
        Default is False.
    apply_threshold_filtering : bool, optional
        If True, applies automatic threshold filtering based on jump detection.
        If False, uses only the parameters from corr_network_params.
        Default is True.
    corr_network_params : dict or None, optional
        Parameters to pass to build_corr_network function.
        Default is {'threshold': 0} if not provided.
        Common parameters: threshold, filter_type, zero_diagonal.
    jump_index : int, optional
        Index of the jump to use for threshold selection (0-based).
        jump_index=0 means 1 giant component, jump_index=1 means 2 components, etc.
        Default is 0 (first jump, single giant component).
    band_name : str, optional
        Name of the band for warning messages. Default is empty string.
    filter_order : int, optional
        Filter order for bandpass filtering. Default is 4.
    
    Returns
    -------
    corr_mat : np.ndarray
        Correlation matrix for the specified frequency band.
    jump_info : dict, optional
        Dictionary with jump information (only if return_jump_info=True).
        
    Notes
    -----
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
    
    # Set default parameters if not provided
    if corr_network_params is None:
        corr_network_params = {'threshold': 0}
    
    low, high = band
    
    # Adjust filter order for narrow bands to avoid numerical instabilities
    # For very narrow bands (< 1% of Nyquist), use order 1
    nyquist = fs / 2.0
    bandwidth_ratio = (high - low) / nyquist
    if bandwidth_ratio < 0.01 and filter_order > 1:
        adjusted_order = 1
        import warnings
        warnings.warn(
            f"Reducing filter order from {filter_order} to {adjusted_order} "
            f"for narrow band ({low}-{high} Hz, {bandwidth_ratio:.4f} of Nyquist) "
            f"to avoid numerical instabilities.",
            RuntimeWarning
        )
    else:
        adjusted_order = filter_order
    
    # Apply bandpass filter
    filter_data = bandpass_func(data_ts, low, high, fs, adjusted_order)
    
    # Check if filtered data contains non-finite values
    if not np.all(np.isfinite(filter_data)):
        warnings.warn(
            f"Band ({low}-{high} Hz): Bandpass filtering produced non-finite values. "
            f"This indicates severe numerical instabilities. Returning None.",
            RuntimeWarning
        )
        if return_jump_info:
            return None, None
        return None
    
    if apply_threshold_filtering:
        # Build initial correlation network for threshold analysis
        corr_mat_initial = build_corr_network(filter_data, 
                                              **corr_network_params)
        
        # Select threshold using jump analysis
        threshold_info = _select_threshold_from_jumps(corr_mat_initial, 
                                                      jump_index, band_name)
        
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
                    f"Band '{band_name}' ({low}-{high} Hz): Expected "
                    f"{expected_components} connected components "
                    f"(jump_index={threshold_info['jump_index']}) but found "
                    f"{n_components} components. Network topology may be "
                    f"unexpected at threshold "
                    f"{threshold_info['chosen_threshold']:.6f}.",
                    UserWarning
                )
            
            # Store validation info
            threshold_info['actual_components'] = n_components
            threshold_info['validation_passed'] = (n_components == 
                                                   expected_components)
        else:
            warnings.warn(
                f"Band '{band_name}' ({low}-{high} Hz): Threshold "
                f"{threshold_info['chosen_threshold']:.6f} resulted in no "
                f"connected nodes. All correlations below threshold.",
                UserWarning
            )
            threshold_info['actual_components'] = 0
            threshold_info['validation_passed'] = False
    else:
        # Build correlation network with provided parameters only
        # No threshold statistics computation
        corr_mat = build_corr_network(filter_data, **corr_network_params)
        
        # Set empty jump info if needed for return consistency
        if return_jump_info:
            threshold_info = {
                'jumps': None,
                'chosen_jump': None,
                'chosen_threshold': corr_network_params.get('threshold', 0),
                'jump_index': None,
                'expected_components': None,
                'actual_components': None,
                'validation_passed': None
            }
    
    if return_jump_info:
        return corr_mat, threshold_info
    else:
        return corr_mat


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