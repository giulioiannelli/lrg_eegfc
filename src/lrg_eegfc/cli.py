"""Command line entry points for :mod:`lrg_eegfc`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import networkx as nx
import numpy as np
from scipy.io import loadmat

from .constants import BRAIN_BANDS, PHASE_LABELS
from .io import load_patient_metadata
from .plotting import (
    plot_correlation_matrix,
    plot_entropy,
    plot_dendrogram,
    plot_graph,
    prepare_dendrogram,
)
from .workflow import compute_band_connectivity

__all__ = ["create_corr_matrix_band_parser", "corr_matrix_band"]


DEFAULT_DATASET_ROOT = Path("data") / "stereoeeg_patients"
DEFAULT_OUTPUT_ROOT = Path("data") / "correlations"


def _configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(asctime)s %(levelname)s %(message)s")


def _load_channel_names(path: Optional[Path], n_channels: int) -> Dict[int, str]:
    if path is None:
        return {i: f"Ch{i}" for i in range(n_channels)}

    mat = loadmat(path)
    names = mat.get("ChannelNames")
    if names is None:
        raise KeyError("The MAT file does not contain a 'ChannelNames' variable")
    decoded = [entry[0] if isinstance(entry, (list, tuple, np.ndarray)) else entry for entry in names[0]]
    return {i: str(name) for i, name in enumerate(decoded)}


def _metadata_channel_map(patient: str, dataset_root: Path, n_channels: int) -> Dict[int, str]:
    metadata = load_patient_metadata(patient, dataset_root)
    if metadata is None or "label" not in metadata.columns:
        return {i: f"Ch{i}" for i in range(n_channels)}
    labels = metadata["label"].tolist()
    if len(labels) < n_channels:
        labels.extend(f"Ch{idx}" for idx in range(len(labels), n_channels))
    return {i: str(label) for i, label in enumerate(labels)}


def create_corr_matrix_band_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute correlation matrices and optional plots for SEEG data",
    )
    parser.add_argument("--patient", "-p", required=True, help="Patient identifier (e.g. Pat_01)")
    parser.add_argument(
        "--phase",
        "-ph",
        required=True,
        choices=PHASE_LABELS,
        help="Recording phase",
    )
    parser.add_argument(
        "--band",
        "-b",
        required=True,
        choices=sorted(BRAIN_BANDS.keys()),
        help="Frequency band to analyse",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing patient sub-folders",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where artefacts will be written",
    )
    parser.add_argument(
        "--filter-time",
        type=int,
        default=None,
        help="Optional maximum number of samples to consider",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=2048.0,
        help="Sampling rate used for band-pass filtering",
    )
    parser.add_argument(
        "--filter-order",
        type=int,
        default=4,
        help="Filter order for the SOS band-pass filter",
    )
    parser.add_argument(
        "--jump-index",
        type=int,
        default=0,
        help="Percolation jump to use when selecting the threshold",
    )
    parser.add_argument(
        "--channel-names",
        type=Path,
        default=None,
        help="Optional path to a ChannelNames.mat file",
    )
    parser.add_argument(
        "--entropy-steps",
        type=int,
        default=400,
        help="Number of steps used when computing network entropy",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...)",
    )
    parser.add_argument("--plot-corr", action="store_true", help="Generate the correlation matrix plot")
    parser.add_argument("--plot-entropy", action="store_true", help="Generate the entropy plot")
    parser.add_argument("--plot-dendrogram", action="store_true", help="Generate the dendrogram plot")
    parser.add_argument("--plot-graph", action="store_true", help="Generate the network graph plot")
    parser.add_argument("--plot-all", action="store_true", help="Generate all available plots")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing correlation matrices instead of reusing them",
    )
    return parser


def corr_matrix_band(argv: Optional[Iterable[str]] = None) -> int:
    parser = create_corr_matrix_band_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    logging.info("Computing connectivity for patient=%s phase=%s band=%s", args.patient, args.phase, args.band)

    output_dir = args.output_root / args.patient
    output_dir.mkdir(parents=True, exist_ok=True)
    corr_path = output_dir / f"{args.band}_{args.phase}_corr.npy"

    if corr_path.exists() and not args.overwrite:
        logging.info("Reusing cached correlation matrix at %s", corr_path)
        corr_matrix = np.load(corr_path)
        graph = nx.from_numpy_array(corr_matrix)
        threshold = None
        jump_stats: Dict[str, object] = {}
    else:
        result = compute_band_connectivity(
            args.patient,
            args.phase,
            args.band,
            args.dataset_root,
            filter_time=args.filter_time,
            sample_rate=args.sample_rate,
            filter_order=args.filter_order,
            jump_index=args.jump_index,
        )
        corr_matrix = result.correlation_matrix
        graph = result.graph
        threshold = result.threshold
        jump_stats = result.jump_statistics
        np.save(corr_path, corr_matrix)
        logging.info("Stored correlation matrix at %s", corr_path)

    n_channels = corr_matrix.shape[0]
    if args.channel_names and args.channel_names.exists():
        channel_map = _load_channel_names(args.channel_names, n_channels)
    else:
        channel_map = _metadata_channel_map(args.patient, args.dataset_root, n_channels)

    if args.plot_all:
        args.plot_corr = args.plot_entropy = args.plot_dendrogram = args.plot_graph = True

    if args.plot_corr:
        plot_correlation_matrix(
            corr_matrix,
            output_dir / f"{args.band}_{args.phase}_corr.png",
            title=f"{args.patient} {args.phase} {args.band}",
        )
        logging.info("Correlation plot written to %s", output_dir)

    dendro = None
    giant = graph
    if args.plot_dendrogram or args.plot_graph:
        giant, linkage_matrix, labels, threshold_value = prepare_dendrogram(graph)
        dendro_path, dendro = plot_dendrogram(
            linkage_matrix,
            labels,
            channel_map,
            threshold_value,
            output_dir / f"{args.band}_{args.phase}_dendrogram.png",
            title=f"Dendrogram {args.patient} {args.phase} {args.band}",
        )
        logging.info("Dendrogram plot written to %s", dendro_path)

    if args.plot_entropy:
        plot_entropy(
            giant,
            output_dir / f"{args.band}_{args.phase}_entropy.png",
            steps=args.entropy_steps,
            title=f"Entropy {args.patient} {args.phase} {args.band}",
        )
        logging.info("Entropy plot written to %s", output_dir)

    if args.plot_graph and dendro is not None:
        plot_graph(
            giant,
            dendro,
            channel_map,
            output_dir / f"{args.band}_{args.phase}_graph.png",
            title=f"Graph {args.patient} {args.phase} {args.band}",
        )
        logging.info("Graph plot written to %s", output_dir)

    if threshold is not None:
        logging.info("Chosen threshold: %.5f", threshold)
        logging.debug("Jump statistics: %s", jump_stats)

    return 0


def main() -> int:  # pragma: no cover - convenience wrapper
    return corr_matrix_band()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
