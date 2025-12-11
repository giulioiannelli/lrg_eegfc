#!/bin/bash
# =============================================================================
# Complete Analysis Pipeline for LRG EEG Functional Connectivity
# =============================================================================
# This script runs the full analysis pipeline on all available patients:
#   1. Compute correlation matrices
#   2. Clean matrices (Marchenko-Pastur noise removal)
#   3. Generate correlation summary visualizations
#   4. Generate cleaned correlation visualizations (skipped)
#   5. Compute MSC matrices
#   5b. Generate MSC summary visualizations
#   5c. Generate FC comparison visualizations (Correlation vs MSC)
#   6. Compare FC methods (distance metrics)
#   7. Compute LRG analysis (Correlation-based)
#   8. Compute LRG analysis (MSC-based)
#   9. Generate LRG visualizations (Correlation)
#   10. Generate LRG visualizations (MSC)
#   11. Generate phase reorganization visualizations (Correlation)
#   12. Generate phase reorganization visualizations (MSC)
#   13. Generate metastable node visualizations (Sankey)
#
# Usage:
#   bash scripts/run_full_analysis.sh
#
# Output:
#   - Correlation matrices: data/corr_cache/Pat_XX/
#   - Cleaned matrices: data/corr_cache/Pat_XX/*_cleaned.npy
#   - Correlation figures: data/figures/correlation/Pat_XX/
#   - MSC matrices: data/msc_cache/Pat_XX/
#   - MSC figures: data/figures/msc/Pat_XX/
#   - FC comparison figures: data/figures/comparison/Pat_XX/
#   - LRG cache: data/lrg_cache/Pat_XX/
#   - LRG figures: data/figures/lrg/Pat_XX/
#   - FC distance metrics: data/comparisons/Pat_XX/
#   - Phase reorganization figures: data/figures/reorganization/Pat_XX/
# =============================================================================

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Available patients and phases
PATIENTS=(Pat_02 Pat_03)
PHASES=(rsPre taskLearn taskTest rsPost)

echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}LRG EEG FC Analysis Pipeline${NC}"
echo -e "${BLUE}=================================${NC}"
echo ""
echo -e "Patients to process: ${GREEN}${PATIENTS[@]}${NC}"
echo ""

# Track timing
START_TIME=$(date +%s)

# Process each patient
for PATIENT in "${PATIENTS[@]}"; do
  echo ""
  echo -e "${YELLOW}========================================${NC}"
  echo -e "${YELLOW}Processing: $PATIENT${NC}"
  echo -e "${YELLOW}========================================${NC}"

  PATIENT_START=$(date +%s)

  # -------------------------------------------------------------------------
  # Step 1: Compute correlation matrices
  # -------------------------------------------------------------------------
  echo -e "${BLUE}[1/12]${NC} Computing correlation matrices..."
  if python src/compute_corr_matrices.py --patient "$PATIENT" --verbose; then
    echo -e "${GREEN}✓${NC} Correlation matrices computed"
  else
    echo -e "${RED}✗${NC} Failed to compute correlation matrices for $PATIENT"
    continue
  fi

  # -------------------------------------------------------------------------
  # Step 2: Clean correlation matrices
  # -------------------------------------------------------------------------
  echo -e "${BLUE}[2/12]${NC} Cleaning correlation matrices (Marchenko-Pastur)..."
  if python src/clean_correlation_matrices.py --patient "$PATIENT" --verbose; then
    echo -e "${GREEN}✓${NC} Correlation matrices cleaned"
  else
    echo -e "${RED}✗${NC} Failed to clean correlation matrices for $PATIENT"
  fi

  # -------------------------------------------------------------------------
  # Step 3: Generate visualizations (uncleaned)
  # -------------------------------------------------------------------------
  echo -e "${BLUE}[3/12]${NC} Generating unified visualizations (uncleaned matrices)..."
  if python src/visualize_correlation.py --patient "$PATIENT" --batch --plot-type summary --verbose; then
    echo -e "${GREEN}✓${NC} Summary visualizations generated"
  else
    echo -e "${RED}✗${NC} Failed to generate summary visualizations for $PATIENT"
  fi

  # -------------------------------------------------------------------------
  # Step 4: Generate visualizations (cleaned)
  # -------------------------------------------------------------------------
  echo -e "${BLUE}[4/12]${NC} Skipping cleaned visualizations (summary currently uses raw/abs matrices)"

  # -------------------------------------------------------------------------
  # Step 5: Compute MSC matrices (if available)
  # -------------------------------------------------------------------------
  echo -e "${BLUE}[5/12]${NC} Computing MSC matrices..."
  if [ -f "src/compute_msc_matrices.py" ]; then
    if python src/compute_msc_matrices.py --patient "$PATIENT" --verbose 2>/dev/null; then
      echo -e "${GREEN}✓${NC} MSC matrices computed"
    else
      echo -e "${YELLOW}⚠${NC} MSC computation skipped or failed for $PATIENT"
    fi
  else
    echo -e "${YELLOW}⚠${NC} MSC script not available yet"
  fi

  # -------------------------------------------------------------------------
  # Step 5b: Generate MSC summary visualizations (if available)
  # -------------------------------------------------------------------------
  echo -e "${BLUE}[5b/8]${NC} Generating MSC summary visualizations..."
  if [ -f "src/visualize_msc.py" ]; then
    if python src/visualize_msc.py --patient "$PATIENT" --batch --plot-type summary --verbose 2>/dev/null; then
      echo -e "${GREEN}✓${NC} MSC summary visualizations generated"
    else
      echo -e "${YELLOW}⚠${NC} MSC visualization skipped or failed for $PATIENT"
    fi
  else
    echo -e "${YELLOW}⚠${NC} MSC visualization script not available yet"
  fi

  # -------------------------------------------------------------------------
  # Step 5c: Generate FC comparison visualizations (Correlation vs MSC)
  # -------------------------------------------------------------------------
  echo -e "${BLUE}[5c/10]${NC} Generating FC comparison visualizations..."
  if [ -f "src/visualize_comparison.py" ]; then
    if python src/visualize_comparison.py --patient "$PATIENT" --batch --verbose 2>/dev/null; then
      echo -e "${GREEN}✓${NC} FC comparison visualizations generated"
    else
      echo -e "${YELLOW}⚠${NC} FC comparison visualization skipped or failed for $PATIENT"
    fi
  else
    echo -e "${YELLOW}⚠${NC} FC comparison visualization script not available yet"
  fi

  # -------------------------------------------------------------------------
  # Step 6: Compare FC methods (if available)
  # -------------------------------------------------------------------------
  echo -e "${BLUE}[6/12]${NC} Comparing FC methods..."
  if [ -f "src/compare_fc_methods.py" ]; then
    if python src/compare_fc_methods.py --patient "$PATIENT" --verbose 2>/dev/null; then
      echo -e "${GREEN}✓${NC} FC methods compared"
    else
      echo -e "${YELLOW}⚠${NC} FC comparison skipped or failed for $PATIENT"
    fi
  else
    echo -e "${YELLOW}⚠${NC} Comparison script not available yet"
  fi

  # -------------------------------------------------------------------------
  # Step 7: Compute LRG analysis (Correlation-based)
  # -------------------------------------------------------------------------
  echo -e "${BLUE}[7/12]${NC} Computing LRG analysis (Correlation)..."
  if [ -f "src/compute_lrg_analysis.py" ]; then
    if python src/compute_lrg_analysis.py --patient "$PATIENT" --fc-method corr --verbose 2>/dev/null; then
      echo -e "${GREEN}✓${NC} Correlation LRG analysis computed"
    else
      echo -e "${YELLOW}⚠${NC} Correlation LRG analysis skipped or failed for $PATIENT"
    fi
  else
    echo -e "${YELLOW}⚠${NC} LRG script not available yet"
  fi

  # -------------------------------------------------------------------------
  # Step 8: Compute LRG analysis (MSC-based)
  # -------------------------------------------------------------------------
  echo -e "${BLUE}[8/12]${NC} Computing LRG analysis (MSC)..."
  if [ -f "src/compute_lrg_analysis.py" ]; then
    if python src/compute_lrg_analysis.py --patient "$PATIENT" --fc-method msc --verbose 2>/dev/null; then
      echo -e "${GREEN}✓${NC} MSC LRG analysis computed"
    else
      echo -e "${YELLOW}⚠${NC} MSC LRG analysis skipped or failed for $PATIENT"
    fi
  fi

  # -------------------------------------------------------------------------
  # Step 9: Generate LRG visualizations (Correlation-based)
  # -------------------------------------------------------------------------
  echo -e "${BLUE}[9/12]${NC} Generating LRG visualizations (Correlation)..."
  if [ -f "src/visualize_lrg.py" ]; then
    if python src/visualize_lrg.py --patient "$PATIENT" --fc-method corr --batch --plot-type full --verbose 2>/dev/null; then
      echo -e "${GREEN}✓${NC} Correlation LRG visualizations generated"
    else
      echo -e "${YELLOW}⚠${NC} Correlation LRG visualization skipped or failed for $PATIENT"
    fi
  else
    echo -e "${YELLOW}⚠${NC} LRG visualization script not available yet"
  fi

  # -------------------------------------------------------------------------
  # Step 10: Generate LRG visualizations (MSC-based)
  # -------------------------------------------------------------------------
  echo -e "${BLUE}[10/12]${NC} Generating LRG visualizations (MSC)..."
  if [ -f "src/visualize_lrg.py" ]; then
    if python src/visualize_lrg.py --patient "$PATIENT" --fc-method msc --batch --plot-type full --verbose 2>/dev/null; then
      echo -e "${GREEN}✓${NC} MSC LRG visualizations generated"
    else
      echo -e "${YELLOW}⚠${NC} MSC LRG visualization skipped or failed for $PATIENT"
    fi
  fi

  # -------------------------------------------------------------------------
  # Step 11: Generate phase reorganization visualizations (Correlation-based)
  # -------------------------------------------------------------------------
  echo -e "${BLUE}[11/12]${NC} Generating phase reorganization visualizations (Correlation)..."
  if [ -f "src/visualize_phase_reorganization.py" ]; then
    if python src/visualize_phase_reorganization.py --patient "$PATIENT" --fc-method corr --batch --plot-type all --verbose 2>/dev/null; then
      echo -e "${GREEN}✓${NC} Correlation phase reorganization visualizations generated"
    else
      echo -e "${YELLOW}⚠${NC} Correlation phase reorganization visualization skipped or failed for $PATIENT"
    fi
  else
    echo -e "${YELLOW}⚠${NC} Phase reorganization visualization script not available yet"
  fi

  # -------------------------------------------------------------------------
  # Step 12: Generate phase reorganization visualizations (MSC-based)
  # -------------------------------------------------------------------------
  echo -e "${BLUE}[12/13]${NC} Generating phase reorganization visualizations (MSC)..."
  if [ -f "src/visualize_phase_reorganization.py" ]; then
    if python src/visualize_phase_reorganization.py --patient "$PATIENT" --fc-method msc --batch --plot-type all --verbose 2>/dev/null; then
      echo -e "${GREEN}✓${NC} MSC phase reorganization visualizations generated"
    else
      echo -e "${YELLOW}⚠${NC} MSC phase reorganization visualization skipped or failed for $PATIENT"
    fi
  fi

  # -------------------------------------------------------------------------
  # Step 13: Metastable nodes visualization (Sankey)
  # -------------------------------------------------------------------------
  echo -e "${BLUE}[13/13]${NC} Generating metastable nodes visualizations (Sankey)..."
  if [ -f "src/visualize_metastable.py" ]; then
    if python src/visualize_metastable.py --patient "$PATIENT" --fc-method corr --batch --verbose 2>/dev/null; then
      echo -e "${GREEN}✓${NC} Metastable visualizations (corr) generated"
    else
      echo -e "${YELLOW}⚠${NC} Metastable visualization skipped or failed (corr) for $PATIENT"
    fi
    if python src/visualize_metastable.py --patient "$PATIENT" --fc-method msc --batch --verbose 2>/dev/null; then
      echo -e "${GREEN}✓${NC} Metastable visualizations (msc) generated"
    else
      echo -e "${YELLOW}⚠${NC} Metastable visualization skipped or failed (msc) for $PATIENT"
    fi
  else
    echo -e "${YELLOW}⚠${NC} Metastable visualization script not available yet"
  fi

  PATIENT_END=$(date +%s)
  PATIENT_TIME=$((PATIENT_END - PATIENT_START))
  echo -e "${GREEN}✓${NC} $PATIENT completed in ${PATIENT_TIME}s"
done

# Summary
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Analysis Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Total time: ${TOTAL_TIME}s ($(($TOTAL_TIME / 60))m $(($TOTAL_TIME % 60))s)"
echo ""
echo -e "${BLUE}Output locations:${NC}"
echo -e "  Correlation cache:         ${GREEN}data/corr_cache/${NC}"
echo -e "  Correlation figures:       ${GREEN}data/figures/correlation/${NC}"
echo -e "  MSC cache:                 ${GREEN}data/msc_cache/${NC}"
echo -e "  MSC figures:               ${GREEN}data/figures/msc/${NC}"
echo -e "  FC comparison figures:     ${GREEN}data/figures/comparison/${NC}"
echo -e "  LRG cache:                 ${GREEN}data/lrg_cache/${NC}"
echo -e "  LRG figures:               ${GREEN}data/figures/lrg/${NC}"
echo -e "  Phase reorganization:      ${GREEN}data/figures/reorganization/${NC}"
echo -e "  FC distance metrics:       ${GREEN}data/comparisons/${NC}"
echo ""
