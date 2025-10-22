"""Ultrametric tree comparison utilities.

This module implements a selection of metrics for comparing ultrametric
matrices and the hierarchical clusterings they represent.  The functions
operate directly on SciPy linkage matrices or their corresponding
ultrametric distance matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.cluster.hierarchy import cophenet, fcluster, to_tree
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau, spearmanr


def _validate_square_matrix(matrix: ArrayLike) -> NDArray[np.float64]:
    """Return ``matrix`` as a float64 NumPy array after validation.

    Parameters
    ----------
    matrix:
        Square, symmetric array encoding pairwise distances.

    Returns
    -------
    numpy.ndarray
        Copy of ``matrix`` converted to ``float64``.
    """

    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        msg = "matrix must be a square two dimensional array"
        raise ValueError(msg)
    return array


def _upper_triangular_vector(
    matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the strictly upper triangular entries of ``matrix``.

    The entries are returned as a contiguous vector.
    """

    return matrix[np.triu_indices_from(matrix, k=1)]


def _ultrametric_matrix_from_linkage(
    linkage: ArrayLike,
) -> NDArray[np.float64]:
    """Return the ultrametric distance matrix derived from ``linkage``.

    Parameters
    ----------
    linkage:
        Matrix with shape ``(n - 1, 4)`` as produced by
        :func:`scipy.cluster.hierarchy.linkage`.

    Returns
    -------
    numpy.ndarray
        Square matrix containing cophenetic distances.
    """

    linkage = np.asarray(linkage, dtype=np.float64)
    if linkage.ndim != 2 or linkage.shape[1] != 4:
        msg = "linkage matrix must have shape (n - 1, 4)"
        raise ValueError(msg)
    _, condensed = cophenet(linkage)
    return squareform(condensed)


def ultrametric_matrix_distance(
    matrix_a: ArrayLike,
    matrix_b: ArrayLike,
    ord: str | int | None = "fro",
) -> float:
    """Return the norm of the difference between two ultrametric matrices.

    Parameters
    ----------
    matrix_a, matrix_b:
        Square matrices of equal size.
    ord:
        Norm type accepted by :func:`numpy.linalg.norm`.  Defaults to the
        Frobenius norm.

    Returns
    -------
    float
        Norm of the difference ``matrix_a - matrix_b``.
    """

    a = _validate_square_matrix(matrix_a)
    b = _validate_square_matrix(matrix_b)
    if a.shape != b.shape:
        msg = "matrices must share the same shape"
        raise ValueError(msg)
    return float(np.linalg.norm(a - b, ord=ord))


def ultrametric_scaled_distance(
    matrix_a: ArrayLike,
    matrix_b: ArrayLike,
    ord: str | int | None = "fro",
) -> float:
    """Return a scale-invariant distance between ultrametric matrices.

    Parameters
    ----------
    matrix_a, matrix_b:
        Square matrices of equal size.
    ord:
        Norm type accepted by :func:`numpy.linalg.norm`.

    Returns
    -------
    float
        Normalised distance ``||A - B|| / (||A|| + ||B||)`` in ``[0, 1]``.
    """

    a = _validate_square_matrix(matrix_a)
    b = _validate_square_matrix(matrix_b)
    norm_a = np.linalg.norm(a, ord=ord)
    norm_b = np.linalg.norm(b, ord=ord)
    denom = norm_a + norm_b
    if denom == 0.0:
        return 0.0
    diff = np.linalg.norm(a - b, ord=ord)
    return float(diff / denom)


def ultrametric_rank_correlation(
    matrix_a: ArrayLike,
    matrix_b: ArrayLike,
) -> float:
    """Return the Spearman rank correlation of two ultrametric matrices.

    Parameters
    ----------
    matrix_a, matrix_b:
        Square matrices of equal size.

    Returns
    -------
    float
        Spearman correlation of the upper triangular entries.
    """

    a = _upper_triangular_vector(_validate_square_matrix(matrix_a))
    b = _upper_triangular_vector(_validate_square_matrix(matrix_b))
    if a.size != b.size:
        msg = "matrices must share the same size"
        raise ValueError(msg)
    if a.size == 0:
        return 1.0
    correlation, _ = spearmanr(a, b)
    return float(correlation)


def ultrametric_quantile_rmse(
    matrix_a: ArrayLike,
    matrix_b: ArrayLike,
    quantiles: Sequence[float] | None = None,
) -> float:
    """Return the root mean square error between ultrametric quantiles.

    Parameters
    ----------
    matrix_a, matrix_b:
        Square matrices of equal size.
    quantiles:
        Sequence of quantiles in ``[0, 1]``.  Defaults to common quintiles.

    Returns
    -------
    float
        Root mean square error between the quantiles of each matrix.
    """

    if quantiles is None:
        quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)
    values_a = np.quantile(
        _upper_triangular_vector(_validate_square_matrix(matrix_a)),
        quantiles,
    )
    values_b = np.quantile(
        _upper_triangular_vector(_validate_square_matrix(matrix_b)),
        quantiles,
    )
    error = values_a - values_b
    return float(np.sqrt(np.mean(error * error)))


def ultrametric_distance_permutation_robust(
    matrix_a: ArrayLike,
    matrix_b: ArrayLike,
    ord: str | int | None = "fro",
    permutations: int = 256,
    random_state: int | None = None,
) -> float:
    """Return the minimal matrix distance across random leaf permutations.

    Parameters
    ----------
    matrix_a, matrix_b:
        Square matrices of equal size.
    ord:
        Norm type accepted by :func:`numpy.linalg.norm`.
    permutations:
        Number of random permutations evaluated.
    random_state:
        Seed used to initialise the pseudo random generator.

    Returns
    -------
    float
        Smallest distance obtained after permuting the labels of ``matrix_b``.
    """

    rng = np.random.default_rng(random_state)
    a = _validate_square_matrix(matrix_a)
    b = _validate_square_matrix(matrix_b)
    if a.shape != b.shape:
        msg = "matrices must share the same shape"
        raise ValueError(msg)
    if permutations <= 0:
        msg = "permutations must be a positive integer"
        raise ValueError(msg)
    n = a.shape[0]
    indices = np.arange(n)
    best = np.linalg.norm(a - b, ord=ord)
    for _ in range(permutations):
        rng.shuffle(indices)
        permuted = b[np.ix_(indices, indices)]
        distance = np.linalg.norm(a - permuted, ord=ord)
        if distance < best:
            best = distance
    return float(best)


def _collect_leaf_ids(node) -> frozenset[int]:
    """Return the identifiers of all leaves under ``node``."""

    stack = [node]
    leaves: list[int] = []
    while stack:
        current = stack.pop()
        if current.is_leaf():
            leaves.append(current.id)
            continue
        stack.append(current.left)
        stack.append(current.right)
    return frozenset(leaves)


def _leaf_sets(node) -> set[frozenset[int]]:
    """Return the set of leaf identifiers for all internal nodes."""

    n_leaves = node.count
    all_leaves = frozenset(range(n_leaves))
    splits: set[frozenset[int]] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current.is_leaf():
            continue
        leaves = _collect_leaf_ids(current)
        size = len(leaves)
        if 0 < size < n_leaves:
            complement = all_leaves - leaves
            splits.add(leaves if size <= n_leaves / 2 else complement)
        stack.append(current.left)
        stack.append(current.right)
    return splits


def tree_robinson_foulds_distance(
    linkage_a: ArrayLike,
    linkage_b: ArrayLike,
) -> int:
    """Return the Robinson–Foulds distance between two dendrograms.

    Parameters
    ----------
    linkage_a, linkage_b:
        Linkage matrices describing binary rooted trees.

    Returns
    -------
    int
        Number of bipartitions present in one tree but not the other.
    """

    tree_a = to_tree(np.asarray(linkage_a, dtype=np.float64), rd=False)
    tree_b = to_tree(np.asarray(linkage_b, dtype=np.float64), rd=False)
    splits_a = _leaf_sets(tree_a)
    splits_b = _leaf_sets(tree_b)
    return int(len(splits_a ^ splits_b))


def tree_cophenetic_correlation(
    linkage_a: ArrayLike,
    linkage_b: ArrayLike,
) -> float:
    """Return the Pearson correlation of cophenetic distances.

    Parameters
    ----------
    linkage_a, linkage_b:
        Linkage matrices describing binary rooted trees.

    Returns
    -------
    float
        Pearson correlation of the ultrametric distances.
    """

    matrix_a = _ultrametric_matrix_from_linkage(linkage_a)
    matrix_b = _ultrametric_matrix_from_linkage(linkage_b)
    vector_a = _upper_triangular_vector(matrix_a)
    vector_b = _upper_triangular_vector(matrix_b)
    if vector_a.size == 0:
        return 1.0
    covariance = np.cov(vector_a, vector_b, bias=True)
    denom = np.sqrt(covariance[0, 0] * covariance[1, 1])
    if denom == 0.0:
        return 0.0
    return float(covariance[0, 1] / denom)


def tree_baker_gamma(
    linkage_a: ArrayLike,
    linkage_b: ArrayLike,
) -> float:
    """Return Baker's Gamma index for two dendrograms.

    Parameters
    ----------
    linkage_a, linkage_b:
        Linkage matrices describing binary rooted trees.

    Returns
    -------
    float
        Baker's Gamma correlation coefficient.
    """

    matrix_a = _ultrametric_matrix_from_linkage(linkage_a)
    matrix_b = _ultrametric_matrix_from_linkage(linkage_b)
    values_a = _upper_triangular_vector(matrix_a)
    values_b = _upper_triangular_vector(matrix_b)
    if values_a.size == 0:
        return 1.0
    tau, _ = kendalltau(values_a, values_b)
    return float(tau)


def _choose_two(counts: Iterable[int]) -> float:
    """Return the sum over counts of ``n * (n - 1) / 2``."""

    counts = np.asarray(list(counts), dtype=np.float64)
    return float(np.sum(counts * (counts - 1.0) * 0.5))


def _fowlkes_mallows(
    labels_a: NDArray[np.int64],
    labels_b: NDArray[np.int64],
) -> float:
    """Return the Fowlkes–Mallows index for two labelings."""

    if labels_a.shape != labels_b.shape:
        msg = "label arrays must share the same shape"
        raise ValueError(msg)
    labels_a = labels_a.astype(np.int64, copy=False)
    labels_b = labels_b.astype(np.int64, copy=False)
    contingency = {}
    counts_a = {}
    counts_b = {}
    for a, b in zip(labels_a, labels_b, strict=True):
        counts_a[a] = counts_a.get(a, 0) + 1
        counts_b[b] = counts_b.get(b, 0) + 1
        contingency[(a, b)] = contingency.get((a, b), 0) + 1
    tp = _choose_two(contingency.values())
    fp = _choose_two(counts_a.values())
    fn = _choose_two(counts_b.values())
    denom = np.sqrt(fp * fn)
    if denom == 0.0:
        return 0.0
    return float(tp / denom)


def tree_fowlkes_mallows_index(
    linkage_a: ArrayLike,
    linkage_b: ArrayLike,
    n_clusters: int,
) -> float:
    """Return the Fowlkes–Mallows index for two dendrograms.

    Parameters
    ----------
    linkage_a, linkage_b:
        Linkage matrices in the format produced by
        :func:`scipy.cluster.hierarchy.linkage`.
    n_clusters:
        Number of clusters extracted from each dendrogram using the
        ``maxclust`` criterion.

    Returns
    -------
    float
        Fowlkes–Mallows index between the cluster labels.
    """

    if n_clusters <= 0:
        msg = "n_clusters must be a positive integer"
        raise ValueError(msg)
    labels_a = fcluster(linkage_a, n_clusters, criterion="maxclust")
    labels_b = fcluster(linkage_b, n_clusters, criterion="maxclust")
    return _fowlkes_mallows(labels_a, labels_b)


@dataclass(slots=True)
class TreeComparisonResult:
    """Container holding the most common tree comparison metrics.

    Attributes
    ----------
    matrix_distance, scaled_distance:
        Matrix based distances between the ultrametric embeddings.
    rank_correlation:
        Spearman correlation of the cophenetic distances.
    quantile_rmse:
        Root mean square error between selected quantiles.
    permutation_distance:
        Robust matrix distance across random permutations.
    robinson_foulds:
        Robinson–Foulds distance between the trees.
    cophenetic_correlation:
        Pearson correlation of the cophenetic distances.
    baker_gamma:
        Baker's Gamma index for the two trees.
    fowlkes_mallows:
        Fowlkes–Mallows index between the induced clusterings.
    """

    matrix_distance: float
    scaled_distance: float
    rank_correlation: float
    quantile_rmse: float
    permutation_distance: float
    robinson_foulds: int
    cophenetic_correlation: float
    baker_gamma: float
    fowlkes_mallows: float


def compare_ultrametric_trees(
    linkage_a: ArrayLike,
    linkage_b: ArrayLike,
    *,
    n_clusters: int,
    quantiles: Sequence[float] | None = None,
    ord: str | int | None = "fro",
    permutations: int = 256,
    random_state: int | None = None,
) -> TreeComparisonResult:
    """Return a suite of comparison metrics for two dendrograms.

    Parameters
    ----------
    linkage_a, linkage_b:
        Linkage matrices describing binary rooted trees.
    n_clusters:
        Number of clusters for the Fowlkes–Mallows comparison.
    quantiles:
        Optional sequence of quantiles used for the RMSE metric.
    ord:
        Norm used for matrix distance computations.
    permutations:
        Number of permutations sampled for the robust matrix distance.
    random_state:
        Seed used to initialise the pseudo random generator.

    Returns
    -------
    TreeComparisonResult
        Dataclass gathering the computed metrics.
    """

    matrix_a = _ultrametric_matrix_from_linkage(linkage_a)
    matrix_b = _ultrametric_matrix_from_linkage(linkage_b)
    return TreeComparisonResult(
        matrix_distance=ultrametric_matrix_distance(
            matrix_a, matrix_b, ord=ord
        ),
        scaled_distance=ultrametric_scaled_distance(
            matrix_a, matrix_b, ord=ord
        ),
        rank_correlation=ultrametric_rank_correlation(matrix_a, matrix_b),
        quantile_rmse=ultrametric_quantile_rmse(
            matrix_a, matrix_b, quantiles=quantiles
        ),
        permutation_distance=ultrametric_distance_permutation_robust(
            matrix_a,
            matrix_b,
            ord=ord,
            permutations=permutations,
            random_state=random_state,
        ),
        robinson_foulds=tree_robinson_foulds_distance(linkage_a, linkage_b),
        cophenetic_correlation=tree_cophenetic_correlation(
            linkage_a, linkage_b
        ),
        baker_gamma=tree_baker_gamma(linkage_a, linkage_b),
        fowlkes_mallows=tree_fowlkes_mallows_index(
            linkage_a, linkage_b, n_clusters=n_clusters
        ),
    )


__all__ = [
    "TreeComparisonResult",
    "compare_ultrametric_trees",
    "tree_baker_gamma",
    "tree_cophenetic_correlation",
    "tree_fowlkes_mallows_index",
    "tree_robinson_foulds_distance",
    "ultrametric_distance_permutation_robust",
    "ultrametric_matrix_distance",
    "ultrametric_quantile_rmse",
    "ultrametric_rank_correlation",
    "ultrametric_scaled_distance",
]
