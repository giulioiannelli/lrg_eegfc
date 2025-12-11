"""Metastable nodes visualization using Sankey diagrams.

This module implements cluster evolution tracking and Sankey diagram generation
to visualize how network nodes transition between communities across different
hierarchical scales (tau values).

Based on FIGMNTGN04.ipynb Sankey diagram analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.colors as colors
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform

__all__ = [
    "compute_clustering_across_tau",
    "analyze_node_trajectories",
    "track_cluster_changes",
    "create_sankey_diagram",
]


def compute_clustering_across_tau(
    linkage_matrix: np.ndarray,
    tau_values: np.ndarray,
    Gcc,
    method: str = "ward",
    scaling_factor: float = 0.99,
) -> Tuple[Dict, Dict]:
    """Compute clustering for different tau values.

    Parameters
    ----------
    linkage_matrix : np.ndarray
        Hierarchical linkage matrix
    tau_values : np.ndarray
        Array of tau (diffusion time) values to test
    Gcc : networkx.Graph
        Network graph
    method : str
        Linkage method (default: "ward")
    scaling_factor : float
        For optimal threshold computation

    Returns
    -------
    partitions : dict
        Mapping tau -> cluster labels array
    n_clusters : dict
        Mapping tau -> number of clusters
    """
    from lrgsglib.core import compute_laplacian_properties, compute_normalized_linkage, compute_optimal_threshold

    partitions = {}
    n_clusters = {}

    for tau in tau_values:
        spectrum, L, rho, Trho, tau_used = compute_laplacian_properties(Gcc, tau=tau)
        dists = squareform(Trho)
        linkage, label_list, _ = compute_normalized_linkage(dists, Gcc, method=method)
        clTh, *_ = compute_optimal_threshold(linkage, scaling_factor=scaling_factor)
        clusters = fcluster(linkage, clTh, criterion="distance")

        partitions[tau] = clusters
        n_clusters[tau] = len(np.unique(clusters))

    return partitions, n_clusters


def analyze_node_trajectories(
    partitions: Dict,
    tau_values: np.ndarray,
    node_labels: Optional[List[str]] = None,
) -> Tuple[Dict, Dict, Dict]:
    """Analyze node cluster membership trajectories.

    Parameters
    ----------
    partitions : dict
        Mapping tau -> cluster labels
    tau_values : ndarray
        Array of tau values
    node_labels : list, optional
        Node names/labels

    Returns
    -------
    node_trajectories : dict
        Mapping node_idx -> list of cluster assignments
    cluster_compositions : dict
        Mapping (tau, cluster) -> list of node indices
    trajectory_patterns : dict
        Unique trajectory patterns and their nodes
    """
    if node_labels is None:
        node_labels = [f"Node_{i}" for i in range(len(partitions[tau_values[0]]))]

    n_nodes = len(partitions[tau_values[0]])

    # Track each node's trajectory
    node_trajectories = {}
    for node_idx in range(n_nodes):
        trajectory = [partitions[tau][node_idx] for tau in tau_values]
        node_trajectories[node_idx] = trajectory

    # Track cluster compositions
    cluster_compositions = {}
    for tau in tau_values:
        clusters = partitions[tau]
        for cluster_id in np.unique(clusters):
            node_indices = np.where(clusters == cluster_id)[0]
            cluster_compositions[(tau, cluster_id)] = node_indices.tolist()

    # Find unique trajectory patterns
    trajectory_patterns = {}
    for node_idx, trajectory in node_trajectories.items():
        trajectory_tuple = tuple(trajectory)
        if trajectory_tuple not in trajectory_patterns:
            trajectory_patterns[trajectory_tuple] = []
        trajectory_patterns[trajectory_tuple].append(node_idx)

    return node_trajectories, cluster_compositions, trajectory_patterns


def track_cluster_changes(
    partitions: Dict,
    tau_values: np.ndarray,
    node_labels: Optional[List[str]] = None,
) -> Tuple[Dict, Dict, Dict]:
    """Track nodes that change clusters between consecutive tau steps.

    Parameters
    ----------
    partitions : dict
        Mapping tau -> cluster labels
    tau_values : ndarray
        Array of tau values
    node_labels : list, optional
        Node names

    Returns
    -------
    cluster_changes : dict
        Mapping (tau1, tau2) -> dict of node changes
    swapping_nodes : dict
        Nodes and their complete change history
    stability_analysis : dict
        Summary of node stability across tau values
    """
    if node_labels is None:
        node_labels = [f"Node_{i}" for i in range(len(partitions[tau_values[0]]))]

    cluster_changes = {}
    swapping_nodes = {node: [] for node in node_labels}

    # Track changes between consecutive tau steps
    for i in range(len(tau_values) - 1):
        tau1, tau2 = tau_values[i], tau_values[i + 1]
        clusters1, clusters2 = partitions[tau1], partitions[tau2]

        changes_at_step = {
            "tau_from": tau1,
            "tau_to": tau2,
            "nodes_changed": [],
            "nodes_stable": [],
            "change_details": {},
        }

        for node_idx, (c1, c2) in enumerate(zip(clusters1, clusters2)):
            node_name = node_labels[node_idx]

            if c1 != c2:
                change_info = {
                    "node": node_name,
                    "from_cluster": c1,
                    "to_cluster": c2,
                    "node_idx": node_idx,
                }
                changes_at_step["nodes_changed"].append(change_info)
                changes_at_step["change_details"][node_name] = change_info

                swapping_nodes[node_name].append(
                    {"tau_from": tau1, "tau_to": tau2, "from_cluster": c1, "to_cluster": c2}
                )
            else:
                changes_at_step["nodes_stable"].append(node_name)

        cluster_changes[(tau1, tau2)] = changes_at_step

    # Analyze stability
    stability_analysis = {}
    for node_name in node_labels:
        num_changes = len(swapping_nodes[node_name])
        stability_analysis[node_name] = {
            "total_changes": num_changes,
            "stability_score": 1 - (num_changes / (len(tau_values) - 1)),
            "change_history": swapping_nodes[node_name],
        }

    return cluster_changes, swapping_nodes, stability_analysis


def create_sankey_diagram(
    partitions: Dict,
    tau_values: np.ndarray,
    node_labels: Optional[List[str]] = None,
    title: str = "Cluster Evolution Across τ Values",
    width: int = 1400,
    height: int = 700,
) -> go.Figure:
    """Create interactive Sankey diagram showing cluster evolution.

    Parameters
    ----------
    partitions : dict
        Mapping tau -> cluster labels
    tau_values : ndarray
        Array of tau values
    node_labels : list, optional
        Node names
    title : str
        Figure title
    width : int
        Figure width
    height : int
        Figure height

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Sankey diagram
    """
    if node_labels is None:
        node_labels = [f"Node_{i}" for i in range(len(partitions[tau_values[0]]))]

    # Get trajectory analysis
    node_trajectories, cluster_compositions, _ = analyze_node_trajectories(
        partitions, tau_values, node_labels
    )

    # Create Sankey data
    sources, targets, values = [], [], []
    labels = []
    hover_texts = []
    node_count = 0

    # Create labels and hover texts
    tau_to_offset = {}
    for i, tau in enumerate(tau_values):
        tau_to_offset[tau] = node_count
        clusters = partitions[tau]
        unique_clusters = np.unique(clusters)

        for cluster in unique_clusters:
            node_indices = cluster_compositions[(tau, cluster)]
            cluster_nodes = [node_labels[idx] for idx in node_indices]

            labels.append(f"τ={tau:.2f}\nC{cluster}")

            # Create hover text with node information
            if len(cluster_nodes) <= 15:
                nodes_text = "<br>".join(cluster_nodes)
            else:
                nodes_text = "<br>".join(cluster_nodes[:15]) + f"<br>... and {len(cluster_nodes)-15} more"

            hover_text = f"τ = {tau:.3f}<br>Cluster {cluster}<br>{len(cluster_nodes)} nodes:<br>{nodes_text}"
            hover_texts.append(hover_text)

        node_count += len(unique_clusters)

    # Create flow connections
    for i in range(len(tau_values) - 1):
        tau1, tau2 = tau_values[i], tau_values[i + 1]
        clusters1, clusters2 = partitions[tau1], partitions[tau2]

        # Count transitions
        transition_matrix = {}
        for node_idx, (c1, c2) in enumerate(zip(clusters1, clusters2)):
            key = (c1, c2)
            transition_matrix[key] = transition_matrix.get(key, 0) + 1

        # Convert to source, target, value
        for (c1, c2), count in transition_matrix.items():
            source_idx = tau_to_offset[tau1] + (c1 - 1)
            target_idx = tau_to_offset[tau2] + (c2 - 1)
            sources.append(source_idx)
            targets.append(target_idx)
            values.append(count)

    # Generate colors
    color_palette = colors.qualitative.Set3 + colors.qualitative.Pastel + colors.qualitative.Dark2
    node_colors = []

    for tau in tau_values:
        clusters = partitions[tau]
        unique_clusters = np.unique(clusters)
        for cluster in unique_clusters:
            color_idx = (cluster - 1) % len(color_palette)
            node_colors.append(color_palette[color_idx])

    # Generate link colors
    link_colors = []
    for target_idx in targets:
        target_color = node_colors[target_idx]
        if target_color.startswith("rgb"):
            rgba_color = target_color.replace("rgb", "rgba").replace(")", ",0.6)")
        elif target_color.startswith("#"):
            hex_color = target_color.lstrip("#")
            r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
            rgba_color = f"rgba({r},{g},{b},0.6)"
        else:
            rgba_color = f"rgba({target_color},0.6)"
        link_colors.append(rgba_color)

    # Create figure
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=node_colors,
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=hover_texts,
                ),
                link=dict(source=sources, target=targets, value=values, color=link_colors),
            )
        ]
    )

    fig.update_layout(title_text=title, font_size=10, width=width, height=height)

    return fig
