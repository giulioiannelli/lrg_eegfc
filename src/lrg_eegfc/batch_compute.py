#!/usr/bin/env python3
"""Batch computation of MSC and correlation matrices for all patients."""

from pathlib import Path
from typing import List, Optional

from .workflow_msc import compute_msc_for_patient
from .workflow_corr import compute_corr_for_patient
from .config.const import BRAIN_BANDS, PHASE_LABELS

__all__ = ["compute_all_matrices", "main"]


def compute_all_matrices(
    patients: Optional[List[str]] = None,
    verbose: bool = False
) -> dict:
    """Compute all MSC and correlation matrices for specified patients.

    Parameters
    ----------
    patients : List[str], optional
        List of patient IDs. If None, uses default list.
    verbose : bool, optional
        Print detailed progress information

    Returns
    -------
    dict
        Summary of computations with keys 'msc_results', 'corr_results',
        'total_msc', 'total_corr'
    """
    if patients is None:
        # Default: patients with complete data
        patients = ["Pat_02", "Pat_03", "Pat_05", "Pat_07", "Pat_08"]

    print("Computing all MSC and Correlation matrices...")
    print("=" * 70)
    print(f"Patients: {', '.join(patients)}")
    print(f"Bands: {', '.join(BRAIN_BANDS.keys())}")
    print(f"Phases: {', '.join(PHASE_LABELS)}")
    expected = len(patients) * len(BRAIN_BANDS) * len(PHASE_LABELS)
    print(f"Expected matrices per method: {expected}")
    print("=" * 70)

    all_msc_results = {}
    all_corr_results = {}
    total_msc = 0
    total_corr = 0

    for patient in patients:
        print(f"\nProcessing {patient}...")
        print("-" * 70)

        # Compute all MSC matrices for this patient
        print(f"  Computing MSC matrices...")
        msc_results = compute_msc_for_patient(patient, verbose=verbose)
        all_msc_results[patient] = msc_results

        # Compute all correlation matrices for this patient
        print(f"  Computing correlation matrices...")
        corr_results = compute_corr_for_patient(patient, verbose=verbose)
        all_corr_results[patient] = corr_results

        # Count successful computations
        msc_count = sum(
            1 for band in msc_results
            for phase in msc_results[band]
            if msc_results[band][phase] is not None
        )
        corr_count = sum(
            1 for band in corr_results
            for phase in corr_results[band]
            if corr_results[band][phase] is not None
        )

        total_msc += msc_count
        total_corr += corr_count

        print(f"  ✓ MSC: {msc_count}/{len(BRAIN_BANDS) * len(PHASE_LABELS)} matrices")
        print(f"  ✓ Correlation: {corr_count}/{len(BRAIN_BANDS) * len(PHASE_LABELS)} matrices")

    print("\n" + "=" * 70)
    print("✓ All matrices computed and cached!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Total MSC matrices: {total_msc}/{expected}")
    print(f"  Total Correlation matrices: {total_corr}/{expected}")
    print(f"\nCache locations:")
    print(f"  MSC: data/msc_cache/{{patient}}/")
    print(f"  Correlation: data/corr_cache/{{patient}}/")

    return {
        'msc_results': all_msc_results,
        'corr_results': all_corr_results,
        'total_msc': total_msc,
        'total_corr': total_corr,
        'patients': patients,
    }


def main() -> int:
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute all MSC and correlation matrices"
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=None,
        help="Patient IDs to process (default: Pat_02 Pat_03 Pat_05 Pat_07 Pat_08)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )

    args = parser.parse_args()

    compute_all_matrices(patients=args.patients, verbose=args.verbose)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
