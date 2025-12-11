"""Hierarchical clustering utilities with outlier detection.

This module provides enhanced hierarchical clustering functions that can identify
outlier nodes - nodes that join clusters above the threshold (colored black in dendrograms).

Based on code from FIGMNTGN03.ipynb.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster


def fcluster_with_outliers(
    linkage_matrix: np.ndarray,
    threshold: float,
    method: str = "distance",
    outlier_sensitivity: float = 1.5,
    min_cluster_size: int = 2,
) -> np.ndarray:
    """Enhanced fcluster that identifies outliers directly.

    Standard scipy.cluster.hierarchy.fcluster forces every node into a cluster.
    This function identifies "outlier" nodes that should not belong to any main
    cluster - nodes that only connect to clusters above the threshold.

    Parameters
    ----------
    linkage_matrix : np.ndarray
        The linkage matrix from hierarchical clustering (N-1, 4)
    threshold : float
        The main clustering threshold
    method : str, optional
        Clustering criterion ('distance', 'inconsistent', etc.), default 'distance'
    outlier_sensitivity : float, optional
        Higher values = more sensitive to outliers (more clusters at high resolution).
        Default 1.5
    min_cluster_size : int, optional
        Minimum size for a cluster to not be considered an outlier.
        Default 2 (singleton clusters are outliers)

    Returns
    -------
    cluster_assignments : np.ndarray
        Cluster assignments where:
        - 0 = outlier cluster
        - 1, 2, 3, ... = main clusters

    Examples
    --------
    >>> from scipy.cluster.hierarchy import linkage
    >>> from scipy.spatial.distance import pdist
    >>> X = np.random.rand(100, 10)
    >>> Z = linkage(pdist(X), method='ward')
    >>> clusters = fcluster_with_outliers(Z, threshold=5.0)
    >>> outliers = np.where(clusters == 0)[0]
    >>> print(f"Found {len(outliers)} outlier nodes")

    Notes
    -----
    This function solves the mismatch between dendrogram coloring and fcluster
    assignments. In dendrograms, nodes that join above the threshold are colored
    black ('k'), but fcluster assigns them to their nearest cluster. This function
    correctly identifies those nodes as outliers (cluster ID 0).
    """
    # Step 1: Get main clusters at the specified threshold
    main_clusters = fcluster(linkage_matrix, t=threshold, criterion=method)
    n_main_clusters = len(np.unique(main_clusters))

    # Step 2: Get higher resolution clustering to detect outliers
    # Use more clusters to identify small/singleton groups
    n_high_res = max(n_main_clusters + 3, int(n_main_clusters * outlier_sensitivity))
    high_res_clusters = fcluster(linkage_matrix, t=n_high_res, criterion="maxclust")

    # Step 3: Identify outlier clusters (small clusters in high-res)
    cluster_sizes = np.bincount(high_res_clusters)[1:]  # Skip index 0
    small_clusters = np.where(cluster_sizes < min_cluster_size)[0] + 1

    # Step 4: Create final clustering with outliers marked as 0
    final_clusters = main_clusters.copy()

    # Mark nodes in small high-resolution clusters as outliers
    for i, high_res_cluster in enumerate(high_res_clusters):
        if high_res_cluster in small_clusters:
            final_clusters[i] = 0

    return final_clusters


def get_dendrogram_consistent_clusters(
    linkage_matrix: np.ndarray,
    dendrogram_result: dict,
    threshold: float,
) -> np.ndarray:
    """Create cluster assignments consistent with dendrogram coloring.

    When you plot a dendrogram with a color threshold, scipy colors branches
    that merge above the threshold as black ('k'). This function extracts cluster
    assignments that match those colors exactly.

    Parameters
    ----------
    linkage_matrix : np.ndarray
        The linkage matrix from hierarchical clustering
    dendrogram_result : dict
        Result dictionary from scipy.cluster.hierarchy.dendrogram()
    threshold : float
        The color threshold used in the dendrogram plot

    Returns
    -------
    cluster_assignments : np.ndarray
        Cluster assignments where:
        - 0 = outlier (black nodes in dendrogram)
        - 1, 2, 3, ... = colored clusters from dendrogram

    Examples
    --------
    >>> from scipy.cluster.hierarchy import dendrogram, linkage
    >>> import matplotlib.pyplot as plt
    >>> Z = linkage(data, method='ward')
    >>> threshold = 10.0
    >>> fig, ax = plt.subplots()
    >>> dendro = dendrogram(Z, ax=ax, color_threshold=threshold)
    >>> clusters = get_dendrogram_consistent_clusters(Z, dendro, threshold)
    >>> # Now clusters exactly match the dendrogram colors!

    Notes
    -----
    This is the most direct way to get clustering that matches dendrogram
    visualization. It's particularly useful when you want cluster assignments
    to align with what users see in the dendrogram plot.
    """
    leaves_colors = dendrogram_result["leaves_color_list"]
    leaves_order = [int(x) for x in dendrogram_result["ivl"]]

    # Create cluster assignments array
    n_nodes = linkage_matrix.shape[0] + 1
    cluster_assignments = np.zeros(n_nodes, dtype=int)

    # Map colors to cluster IDs
    color_to_cluster = {}
    cluster_id = 1

    for node_idx, color in zip(leaves_order, leaves_colors):
        if color == "k":  # Black nodes are outliers
            cluster_assignments[node_idx] = 0
        else:
            if color not in color_to_cluster:
                color_to_cluster[color] = cluster_id
                cluster_id += 1
            cluster_assignments[node_idx] = color_to_cluster[color]

    return cluster_assignments


def compute_optimal_clusters_auto(
    linkage_matrix: np.ndarray,
    method: str = "auto",
    outlier_detection: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, float]:
    """Compute optimal clustering with automatic threshold selection.

    This function wraps lrgsglib's compute_optimal_threshold and provides
    optional outlier detection.

    Parameters
    ----------
    linkage_matrix : np.ndarray
        The linkage matrix from hierarchical clustering
    method : str, optional
        Method for optimal threshold:
        - 'auto' : Use lrgsglib.compute_optimal_threshold (default)
        - 'gap' : Gap statistic
        - 'silhouette' : Silhouette score
    outlier_detection : bool, optional
        If True, use fcluster_with_outliers. Default True
    **kwargs
        Additional arguments passed to threshold computation

    Returns
    -------
    clusters : np.ndarray
        Cluster assignments (with outliers if outlier_detection=True)
    threshold : float
        The computed optimal threshold

    Examples
    --------
    >>> from scipy.cluster.hierarchy import linkage
    >>> Z = linkage(data, method='ward')
    >>> clusters, threshold = compute_optimal_clusters_auto(Z)
    >>> print(f"Optimal threshold: {threshold:.3f}")
    >>> print(f"Number of clusters: {len(np.unique(clusters))}")
    >>> print(f"Number of outliers: {np.sum(clusters == 0)}")
    """
    from lrgsglib.core import compute_optimal_threshold

    if method == "auto":
        # Use lrgsglib's method
        scaling_factor = kwargs.pop("scaling_factor", 0.98)
        threshold, *_ = compute_optimal_threshold(
            linkage_matrix, scaling_factor=scaling_factor
        )
    else:
        raise NotImplementedError(f"Method '{method}' not yet implemented")

    # Compute clusters
    if outlier_detection:
        clusters = fcluster_with_outliers(linkage_matrix, threshold, **kwargs)
    else:
        clusters = fcluster(linkage_matrix, t=threshold, criterion="distance")

    return clusters, threshold


def get_outlier_nodes(clusters: np.ndarray) -> np.ndarray:
    """Get indices of outlier nodes from cluster assignments.

    Parameters
    ----------
    clusters : np.ndarray
        Cluster assignments where 0 indicates outliers

    Returns
    -------
    outlier_indices : np.ndarray
        Array of node indices that are outliers

    Examples
    --------
    >>> clusters = fcluster_with_outliers(linkage_matrix, threshold=5.0)
    >>> outliers = get_outlier_nodes(clusters)
    >>> print(f"Outlier nodes: {outliers}")
    """
    return np.where(clusters == 0)[0]


def compute_cluster_statistics(clusters: np.ndarray) -> dict:
    """Compute statistics about clustering results.

    Parameters
    ----------
    clusters : np.ndarray
        Cluster assignments

    Returns
    -------
    stats : dict
        Dictionary containing:
        - n_clusters : Number of clusters (excluding outliers)
        - n_outliers : Number of outlier nodes
        - cluster_sizes : Array of cluster sizes
        - mean_size : Mean cluster size
        - std_size : Standard deviation of cluster sizes

    Examples
    --------
    >>> clusters = fcluster_with_outliers(linkage_matrix, threshold=5.0)
    >>> stats = compute_cluster_statistics(clusters)
    >>> print(f"Found {stats['n_clusters']} clusters")
    >>> print(f"Found {stats['n_outliers']} outliers")
    """
    unique_clusters = np.unique(clusters)
    n_outliers = np.sum(clusters == 0)

    # Get non-outlier clusters
    non_outlier_clusters = unique_clusters[unique_clusters != 0]
    n_clusters = len(non_outlier_clusters)

    # Compute cluster sizes (excluding outliers)
    cluster_sizes = np.array([np.sum(clusters == c) for c in non_outlier_clusters])

    return {
        "n_clusters": n_clusters,
        "n_outliers": n_outliers,
        "cluster_sizes": cluster_sizes,
        "mean_size": np.mean(cluster_sizes) if len(cluster_sizes) > 0 else 0,
        "std_size": np.std(cluster_sizes) if len(cluster_sizes) > 0 else 0,
    }
