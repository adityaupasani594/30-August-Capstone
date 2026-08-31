#!/usr/bin/env bash
# ==============================================================================
# run_chart_generator.sh
# ==============================================================================
# Executes the publication-grade QAOA visualization generator using the
# project's Python virtual environment.
# ==============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_BIN="${SCRIPT_DIR}/venv/bin/python3"

if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

echo "[INFO] Running QAOA Algorithmic Chart Generator script..."
"$PYTHON_BIN" "${SCRIPT_DIR}/generate_qaoa_charts.py"

echo ""
echo "[INFO] Running Scenario-by-Scenario QAOA vs PSO Chart Generator script..."
"$PYTHON_BIN" "${SCRIPT_DIR}/generate_scenario_qaoa_vs_pso_charts.py"

echo ""
echo "[INFO] Generated Chart Files in ${SCRIPT_DIR}/charts:"
ls -lh "${SCRIPT_DIR}/charts"/*.png
