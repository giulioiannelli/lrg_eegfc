"""Plotting utilities used by the command line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import squareform

from lrgsglib.core import (
    compute_laplacian_properties,
    compute_normalized_linkage,
    compute_optimal_threshold,
    entropy,
    get_giant_component,
)

__all__ = [
    "plot_correlation_matrix",
    "plot_entropy",
    "prepare_dendrogram",
    "plot_dendrogram",
    "plot_graph",
]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_correlation_matrix(
    matrix: np.ndarray,
    output_path: Path,
    *,
    title: str = "Correlation matrix",
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> Path:
    """Plot ``matrix`` and save the figure to ``output_path``."""

    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="coolwarm")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_entropy(
    graph: nx.Graph,
    output_path: Path,
    *,
    steps: int = 400,
    title: str = "Network entropy",
    log_scale: bool = True,
) -> Path:
    """Generate the entropy plot described in the LRG connectivity papers."""

    _ensure_parent(output_path)
    sm1, spec, *_, tau = entropy(graph, t1=-3, t2=5, steps=steps)
    sm1_norm = sm1 / sm1.max()
    spec_norm = spec / spec.max()

    fig, ax = plt.subplots()
    ax.plot(tau, 1 - sm1_norm, label=r"$1-S$")
    ax.plot(tau[:-1], spec_norm, label=r"$C$")
    if log_scale:
        ax.set_xscale("log")
    ax.set_xlabel("Scale (τ)")
    ax.set_ylabel("Normalised value")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def prepare_dendrogram(graph: nx.Graph):
    """Compute the dendrogram linkage matrix and metadata for ``graph``."""

    giant = get_giant_component(graph)
    _, _, _, Trho, _ = compute_laplacian_properties(giant)
    dists = squareform(Trho)
    linkage_matrix, labels, _ = compute_normalized_linkage(dists, giant)
    threshold, *_ = compute_optimal_threshold(linkage_matrix)
    return giant, linkage_matrix, labels, float(threshold)


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    labels: Sequence[int],
    channel_map: Mapping[int, str],
    threshold: float,
    output_path: Path,
    *,
    title: str = "Dendrogram",
) -> Path:
    """Plot the dendrogram corresponding to ``linkage_matrix``."""

    _ensure_parent(output_path)
    relabelled = [channel_map.get(label, str(label)) for label in labels]
    fig, ax = plt.subplots(figsize=(6, 10))
    dendro = dendrogram(
        linkage_matrix,
        ax=ax,
        labels=relabelled,
        color_threshold=threshold,
        above_threshold_color="k",
        leaf_font_size=9,
        orientation="right",
    )
    ax.axvline(threshold, color="b", linestyle="--", label=r"$\mathcal{D}_{\mathrm{th}}$")
    ax.legend()
    ax.set_xscale("log")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path, dendro


def plot_graph(
    graph: nx.Graph,
    dendro: Mapping[str, Sequence[str]],
    channel_map: Mapping[int, str],
    output_path: Path,
    *,
    title: str = "Graph",
) -> Path:
    """Draw ``graph`` using colours taken from ``dendro``."""

    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    leaf_colours = {
        label: colour for label, colour in zip(dendro["ivl"], dendro["leaves_color_list"])
    }
    node_colours = [leaf_colours.get(channel_map.get(node, str(node)), "tab:blue") for node in graph.nodes]
    widths = [graph[u][v].get("weight", 1.0) for u, v in graph.edges]
    nx.draw(
        graph,
        ax=ax,
        node_size=80,
        font_size=10,
        width=widths,
        node_color=node_colours,
        alpha=0.8,
        with_labels=True,
        labels={node: channel_map.get(node, str(node)) for node in graph.nodes},
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
