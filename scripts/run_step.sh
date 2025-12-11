#!/bin/bash
# =============================================================================
# Modular Pipeline Step Runner
# =============================================================================
# Run individual pipeline steps with test or batch mode
#
# Usage:
#   # Quick test (single case: Pat_03/rsPre/beta)
#   bash scripts/run_step.sh --test --step 5b
#
#   # Test with custom parameters
#   bash scripts/run_step.sh --test --step 7 --patient Pat_02 --phase taskLearn --band alpha
#
#   # Run step in batch mode (all bands/phases for a patient)
#   bash scripts/run_step.sh --step 9 --patient Pat_03 --fc-method corr
#
#   # List available steps
#   bash scripts/run_step.sh --list
#
# Steps:
#   1   - Compute correlation matrices
#   2   - Clean correlation matrices
#   3   - Visualize correlations
#   5   - Compute MSC matrices
#   5b  - Visualize MSC (summary)
#   5c  - Visualize FC comparison (Corr vs MSC)
#   6   - Compare FC methods (distances)
#   7   - Compute LRG (Correlation)
#   8   - Compute LRG (MSC)
#   9   - Visualize LRG (Correlation)
#   10  - Visualize LRG (MSC)
#   11  - Visualize phase reorganization (Correlation)
#   12  - Visualize phase reorganization (MSC)
#   meta - Visualize metastable nodes (Sankey)
# =============================================================================

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# Default values
TEST_MODE=false
STEP=""
PATIENT=""
PHASE=""
BAND=""
FC_METHOD=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --test)
      TEST_MODE=true
      shift
      ;;
    --step)
      STEP="$2"
      shift 2
      ;;
    --patient)
      PATIENT="$2"
      shift 2
      ;;
    --phase)
      PHASE="$2"
      shift 2
      ;;
    --band)
      BAND="$2"
      shift 2
      ;;
    --fc-method)
      FC_METHOD="$2"
      shift 2
      ;;
    --list)
      echo -e "${BLUE}Available pipeline steps:${NC}"
      echo -e "  ${CYAN}1${NC}    - Compute correlation matrices"
      echo -e "  ${CYAN}2${NC}    - Clean correlation matrices"
      echo -e "  ${CYAN}3${NC}    - Visualize correlations (summary)"
      echo -e "  ${CYAN}5${NC}    - Compute MSC matrices"
      echo -e "  ${CYAN}5b${NC}   - Visualize MSC (summary)"
      echo -e "  ${CYAN}5c${NC}   - Visualize FC comparison (Corr vs MSC)"
      echo -e "  ${CYAN}6${NC}    - Compare FC methods (distance metrics)"
      echo -e "  ${CYAN}7${NC}    - Compute LRG analysis (use --fc-method)"
      echo -e "  ${CYAN}8${NC}    - Compute LRG analysis (use --fc-method)"
      echo -e "  ${CYAN}9${NC}    - Visualize LRG (use --fc-method)"
      echo -e "  ${CYAN}10${NC}   - Visualize LRG (use --fc-method)"
      echo -e "  ${CYAN}11${NC}   - Visualize phase reorganization (use --fc-method)"
      echo -e "  ${CYAN}12${NC}   - Visualize phase reorganization (use --fc-method)"
      echo -e "  ${CYAN}meta${NC} - Visualize metastable nodes (Sankey diagrams)"
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo "Use --list to see available steps"
      exit 1
      ;;
  esac
done

# Validate step is provided
if [ -z "$STEP" ]; then
  echo -e "${RED}Error: --step is required${NC}"
  echo "Use --list to see available steps"
  exit 1
fi

# Set defaults based on mode
if [ "$TEST_MODE" = true ]; then
  # Test mode: use single case defaults (overridable)
  PATIENT="${PATIENT:-$TEST_PATIENT}"
  PHASE="${PHASE:-$TEST_PHASE}"
  BAND="${BAND:-$TEST_BAND}"
  FC_METHOD="${FC_METHOD:-corr}"

  echo -e "${CYAN}========================================${NC}"
  echo -e "${CYAN}Running in TEST MODE${NC}"
  echo -e "${CYAN}========================================${NC}"
  echo -e "Patient: ${GREEN}$PATIENT${NC}"
  echo -e "Phase:   ${GREEN}$PHASE${NC}"
  echo -e "Band:    ${GREEN}$BAND${NC}"
  echo -e "FC:      ${GREEN}$FC_METHOD${NC}"
  echo -e "Step:    ${GREEN}$STEP${NC}"
  echo ""
else
  # Batch mode: require at least patient
  if [ -z "$PATIENT" ]; then
    echo -e "${RED}Error: --patient is required in batch mode${NC}"
    exit 1
  fi

  FC_METHOD="${FC_METHOD:-corr}"

  echo -e "${BLUE}========================================${NC}"
  echo -e "${BLUE}Running in BATCH MODE${NC}"
  echo -e "${BLUE}========================================${NC}"
  echo -e "Patient: ${GREEN}$PATIENT${NC}"
  [ -n "$PHASE" ] && echo -e "Phase:   ${GREEN}$PHASE${NC}" || echo -e "Phase:   ${GREEN}All${NC}"
  [ -n "$BAND" ] && echo -e "Band:    ${GREEN}$BAND${NC}" || echo -e "Band:    ${GREEN}All${NC}"
  echo -e "FC:      ${GREEN}$FC_METHOD${NC}"
  echo -e "Step:    ${GREEN}$STEP${NC}"
  echo ""
fi

# Execute step
START_TIME=$(date +%s)

run_step_1() {
  echo -e "${BLUE}[Step 1]${NC} Computing correlation matrices..."
  if [ "$TEST_MODE" = true ]; then
    python src/compute_corr_matrices.py --patient "$PATIENT" --phase "$PHASE" --band "$BAND" --verbose
  else
    python src/compute_corr_matrices.py --patient "$PATIENT" --verbose
  fi
}

run_step_2() {
  echo -e "${BLUE}[Step 2]${NC} Cleaning correlation matrices..."
  if [ "$TEST_MODE" = true ]; then
    python src/clean_correlation_matrices.py --patient "$PATIENT" --phase "$PHASE" --band "$BAND" --verbose
  else
    python src/clean_correlation_matrices.py --patient "$PATIENT" --verbose
  fi
}

run_step_3() {
  echo -e "${BLUE}[Step 3]${NC} Visualizing correlations..."
  if [ "$TEST_MODE" = true ]; then
    python src/visualize_correlation.py --patient "$PATIENT" --phase "$PHASE" --band "$BAND" --plot-type summary --verbose
  else
    python src/visualize_correlation.py --patient "$PATIENT" --batch --plot-type summary --verbose
  fi
}

run_step_5() {
  echo -e "${BLUE}[Step 5]${NC} Computing MSC matrices..."
  if [ "$TEST_MODE" = true ]; then
    python src/compute_msc_matrices.py --patient "$PATIENT" --phase "$PHASE" --band "$BAND" --verbose
  else
    python src/compute_msc_matrices.py --patient "$PATIENT" --verbose
  fi
}

run_step_5b() {
  echo -e "${BLUE}[Step 5b]${NC} Visualizing MSC (summary)..."
  if [ "$TEST_MODE" = true ]; then
    python src/visualize_msc.py --patient "$PATIENT" --phase "$PHASE" --band "$BAND" --plot-type summary --verbose
  else
    python src/visualize_msc.py --patient "$PATIENT" --batch --plot-type summary --verbose
  fi
}

run_step_5c() {
  echo -e "${BLUE}[Step 5c]${NC} Visualizing FC comparison (Corr vs MSC)..."
  if [ "$TEST_MODE" = true ]; then
    python src/visualize_comparison.py --patient "$PATIENT" --phase "$PHASE" --band "$BAND" --verbose
  else
    python src/visualize_comparison.py --patient "$PATIENT" --batch --verbose
  fi
}

run_step_6() {
  echo -e "${BLUE}[Step 6]${NC} Comparing FC methods (distance metrics)..."
  python src/compare_fc_methods.py --patient "$PATIENT" --verbose
}

run_step_7() {
  echo -e "${BLUE}[Step 7]${NC} Computing LRG analysis (FC method: $FC_METHOD)..."
  if [ "$TEST_MODE" = true ]; then
    python src/compute_lrg_analysis.py --patient "$PATIENT" --phase "$PHASE" --band "$BAND" --fc-method "$FC_METHOD" --verbose
  else
    python src/compute_lrg_analysis.py --patient "$PATIENT" --fc-method "$FC_METHOD" --verbose
  fi
}

run_step_8() {
  echo -e "${BLUE}[Step 8]${NC} Computing LRG analysis (FC method: $FC_METHOD)..."
  if [ "$TEST_MODE" = true ]; then
    python src/compute_lrg_analysis.py --patient "$PATIENT" --phase "$PHASE" --band "$BAND" --fc-method "$FC_METHOD" --verbose
  else
    python src/compute_lrg_analysis.py --patient "$PATIENT" --fc-method "$FC_METHOD" --verbose
  fi
}

run_step_9() {
  echo -e "${BLUE}[Step 9]${NC} Visualizing LRG (FC method: $FC_METHOD)..."
  if [ "$TEST_MODE" = true ]; then
    python src/visualize_lrg.py --patient "$PATIENT" --phase "$PHASE" --band "$BAND" --fc-method "$FC_METHOD" --plot-type full --verbose
  else
    python src/visualize_lrg.py --patient "$PATIENT" --fc-method "$FC_METHOD" --batch --plot-type full --verbose
  fi
}

run_step_10() {
  echo -e "${BLUE}[Step 10]${NC} Visualizing LRG (FC method: $FC_METHOD)..."
  if [ "$TEST_MODE" = true ]; then
    python src/visualize_lrg.py --patient "$PATIENT" --phase "$PHASE" --band "$BAND" --fc-method "$FC_METHOD" --plot-type full --verbose
  else
    python src/visualize_lrg.py --patient "$PATIENT" --fc-method "$FC_METHOD" --batch --plot-type full --verbose
  fi
}

run_step_11() {
  echo -e "${BLUE}[Step 11]${NC} Visualizing phase reorganization (FC method: $FC_METHOD)..."
  if [ "$TEST_MODE" = true ]; then
    python src/visualize_phase_reorganization.py --patient "$PATIENT" --band "$BAND" --fc-method "$FC_METHOD" --plot-type all --verbose
  else
    python src/visualize_phase_reorganization.py --patient "$PATIENT" --fc-method "$FC_METHOD" --batch --plot-type all --verbose
  fi
}

run_step_12() {
  echo -e "${BLUE}[Step 12]${NC} Visualizing phase reorganization (FC method: $FC_METHOD)..."
  if [ "$TEST_MODE" = true ]; then
    python src/visualize_phase_reorganization.py --patient "$PATIENT" --band "$BAND" --fc-method "$FC_METHOD" --plot-type all --verbose
  else
    python src/visualize_phase_reorganization.py --patient "$PATIENT" --fc-method "$FC_METHOD" --batch --plot-type all --verbose
  fi
}

run_step_meta() {
  echo -e "${BLUE}[Step meta]${NC} Visualizing metastable nodes (Sankey)..."
  if [ "$TEST_MODE" = true ]; then
    python src/visualize_metastable.py --patient "$PATIENT" --phase "$PHASE" --band "$BAND" --fc-method "$FC_METHOD" --verbose
  else
    python src/visualize_metastable.py --patient "$PATIENT" --fc-method "$FC_METHOD" --batch --verbose
  fi
}

# Execute the requested step
case $STEP in
  1)   run_step_1 ;;
  2)   run_step_2 ;;
  3)   run_step_3 ;;
  5)   run_step_5 ;;
  5b)  run_step_5b ;;
  5c)  run_step_5c ;;
  6)   run_step_6 ;;
  7)   run_step_7 ;;
  8)   run_step_8 ;;
  9)   run_step_9 ;;
  10)  run_step_10 ;;
  11)  run_step_11 ;;
  12)  run_step_12 ;;
  meta) run_step_meta ;;
  *)
    echo -e "${RED}Unknown step: $STEP${NC}"
    echo "Use --list to see available steps"
    exit 1
    ;;
esac

RESULT=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $RESULT -eq 0 ]; then
  echo ""
  echo -e "${GREEN}✓ Step $STEP completed successfully in ${ELAPSED}s${NC}"
else
  echo ""
  echo -e "${RED}✗ Step $STEP failed (exit code: $RESULT)${NC}"
  exit $RESULT
fi
