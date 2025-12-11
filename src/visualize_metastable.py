"""CLI script for generating metastable nodes visualizations (Sankey diagrams).

Visualizes cluster evolution across hierarchical scales to identify stable vs unstable nodes.
"""

import argparse
from pathlib import Path

import numpy as np

from lrg_eegfc.config.const import BRAIN_BANDS, PHASE_LABELS
from lrg_eegfc.visuals.metastable import (
    analyze_node_trajectories,
    compute_clustering_across_tau,
    create_sankey_diagram,
    track_cluster_changes,
)
from lrg_eegfc.workflow_lrg import load_lrg_result


def main():
    parser = argparse.ArgumentParser(
        description="Generate metastable nodes visualization (Sankey diagram)"
    )
    parser.add_argument("--patient", required=True, help="Patient ID (e.g., Pat_03)")
    parser.add_argument("--phase", help="Phase (e.g., rsPre)")
    parser.add_argument("--band", help="Frequency band (e.g., beta)")
    parser.add_argument(
        "--fc-method",
        required=True,
        choices=["corr", "msc"],
        help="FC method: corr or msc",
    )
    parser.add_argument(
        "--n-tau",
        type=int,
        default=4,
        help="Number of tau values to test (default: 4)",
    )
    parser.add_argument(
        "--tau-min",
        type=float,
        default=1e-4,
        help="Minimum tau value (default: 1e-4)",
    )
    parser.add_argument(
        "--tau-max",
        type=float,
        default=1000,
        help="Maximum tau value (default: 1000)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all bands (and phases if --phase not specified)",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/lrg_cache"),
        help="LRG cache root directory",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/stereoeeg_patients"),
        help="Dataset root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/figures/metastable"),
        help="Output directory for Sankey diagrams",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Determine what to process
    if args.batch:
        bands = list(BRAIN_BANDS.keys())
        phases = list(PHASE_LABELS) if not args.phase else [args.phase]
    else:
        if not args.phase or not args.band:
            parser.error("--phase and --band required unless --batch is used")
        bands = [args.band]
        phases = [args.phase]

    # Create output directory
    patient_output_dir = args.output_dir / args.patient
    patient_output_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"\n=== Metastable nodes for {args.patient} ({args.fc_method}) ===")
        print(f"  bands: {', '.join(bands)}")
        print(f"  phases: {', '.join(phases)}")
        print(f"  cache: {args.cache_root}/{args.patient}/")
        print(f"  output: {patient_output_dir}/")

    # Process each combination
    for phase in phases:
        for band in bands:
            try:
                output_file = patient_output_dir / f"{band}_{phase}_sankey_{args.fc_method}.html"
                if output_file.exists():
                    if args.verbose:
                        print(f"• Skipping (exists): {output_file.name}")
                    continue

                # Load LRG result
                lrg_result = load_lrg_result(
                    args.patient, phase, band, args.fc_method, args.cache_root
                )

                if lrg_result is None:
                    print(
                        f"  ✗ LRG result not found for {args.patient} {phase} {band} ({args.fc_method})"
                    )
                    continue

                # Load channel labels
                try:
                    from lrg_eegfc.utils.datamanag.patient_robust import (
                        load_patient_dataset_robust,
                    )

                    dataset = load_patient_dataset_robust(
                        args.patient, args.dataset_root, phases=[phase]
                    )
                    recording = dataset[phase]

                    if hasattr(recording, "channel_labels") and recording.channel_labels:
                        node_labels = list(recording.channel_labels)
                    else:
                        node_labels = [f"Ch{i}" for i in range(lrg_result.n_nodes)]
                except Exception:
                    node_labels = [f"Ch{i}" for i in range(lrg_result.n_nodes)]

                # Generate tau values based on entropy curve
                # According to LRG theory: $\tau$ should range from $\tau' = 1/\lambda_{max}$ to $\tau^*$
                # $\tau^*$ is the peak of the specific heat curve $C(\tau)$

                # Find $\tau^*$ (peak of specific heat)
                tau_star_idx = np.argmax(lrg_result.entropy_C)
                tau_star = lrg_result.entropy_tau[tau_star_idx]

                # Find $\tau'$ ($1/\lambda_{max}$) - approximate as minimum meaningful tau
                # This is typically around where entropy starts to change significantly
                tau_prime = lrg_result.entropy_tau[0]  # Minimum tau from entropy computation

                # Generate tau values in the meaningful range
                # Space them logarithmically between $\tau'$ and slightly beyond $\tau^*$
                tau_max_effective = tau_star * 10  # Go a bit beyond the peak
                tau_values = np.logspace(
                    np.log10(tau_prime), np.log10(tau_max_effective), args.n_tau
                )[::-1]  # Reverse to go from large to small (coarse to fine)

                if args.verbose:
                    print(rf"  $\tau'$ (1/$\lambda_{{max}}$) ≈ {tau_prime:.4f}")
                    print(rf"  $\tau^*$ (peak of C) = {tau_star:.4f}")
                    print(f"  Computing clustering across {args.n_tau} tau values: {[f'{t:.4f}' for t in tau_values]}")

                # Compute clustering across tau values (requires network)
                # We need to reconstruct the network from the LRG result
                # For now, we'll create a placeholder that loads the FC matrix
                from lrg_eegfc.workflow_corr import load_corr_matrix
                from lrg_eegfc.workflow_msc import load_msc_matrix
                import networkx as nx

                if args.fc_method == "corr":
                    fc_matrix = load_corr_matrix(
                        args.patient,
                        phase,
                        band,
                        cache_root=Path("data/corr_cache"),
                        filter_type="abs",
                        zero_diagonal=True,
                    )
                else:
                    fc_matrix = load_msc_matrix(
                        args.patient,
                        phase,
                        band,
                        cache_root=Path("data/msc_cache"),
                        sparsify="none",
                        n_surrogates=0,
                    )

                if fc_matrix is None:
                    print(f"  ✗ FC matrix not found")
                    continue

                # Create network
                Gcc = nx.from_numpy_array(fc_matrix)

                # Compute clustering across tau
                partitions, n_clusters = compute_clustering_across_tau(
                    lrg_result.linkage_matrix, tau_values, Gcc
                )

                if args.verbose:
                    print(f"  Clusters per tau: {[n_clusters[tau] for tau in tau_values]}")

                # Analyze trajectories and changes
                node_trajectories, cluster_compositions, trajectory_patterns = (
                    analyze_node_trajectories(partitions, tau_values, node_labels)
                )

                cluster_changes, swapping_nodes, stability_analysis = track_cluster_changes(
                    partitions, tau_values, node_labels
                )

                # Create Sankey diagram
                title = f"Cluster Evolution - {args.patient} {phase} {band} ({args.fc_method})"
                fig = create_sankey_diagram(
                    partitions, tau_values, node_labels, title=title
                )

                # Save as HTML
                fig.write_html(str(output_file))

                if args.verbose:
                    print(f"✓ Sankey: {phase} {band} -> {output_file.name}")

                # Print stability summary
                if args.verbose:
                    stability_scores = [
                        analysis["stability_score"] for analysis in stability_analysis.values()
                    ]
                    mean_stability = np.mean(stability_scores)
                    print(f"  Mean node stability: {mean_stability:.3f}")

                    # Find most stable nodes
                    stable_nodes = sorted(
                        stability_analysis.items(),
                        key=lambda x: x[1]["stability_score"],
                        reverse=True,
                    )[:5]
                    print(f"  Top 5 stable nodes:")
                    for node, analysis in stable_nodes:
                        print(
                            f"    {node}: {analysis['stability_score']:.3f} ({analysis['total_changes']} changes)"
                        )

            except Exception as e:
                print(f"  ✗ Error processing {phase} {band}: {e}")
                if args.verbose:
                    import traceback

                    traceback.print_exc()

    print(f"\n✓ Metastable visualization complete!")
    print(f"Output directory: {patient_output_dir}")


if __name__ == "__main__":
    main()
