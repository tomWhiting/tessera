#!/bin/bash
# Launcher script for Tessera Marimo notebooks

set -e

NOTEBOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$NOTEBOOK_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

if [ $# -eq 0 ]; then
    echo "Tessera Marimo Notebooks Launcher"
    echo "=================================="
    echo ""
    echo "Usage: ./run.sh [notebook]"
    echo ""
    echo "Available notebooks:"
    echo "  1) embedding     - Embedding Paradigm Comparison"
    echo "  2) timeseries    - Legacy time-series quarantine note"
    echo "  3) both          - Open both notebooks"
    echo ""
    echo "Examples:"
    echo "  ./run.sh embedding"
    echo "  ./run.sh timeseries"
    echo "  ./run.sh both"
    echo ""
    exit 0
fi

case "$1" in
    embedding|1)
        echo "🚀 Launching Embedding Comparison notebook..."
        uv run marimo edit examples/notebooks/embedding_comparison.py
        ;;
    timeseries|2)
        echo "🚀 Opening the legacy time-series quarantine note..."
        uv run marimo edit examples/notebooks/timeseries_forecasting.py
        ;;
    both|3)
        echo "🚀 Launching both notebooks..."
        uv run marimo edit examples/notebooks/embedding_comparison.py examples/notebooks/timeseries_forecasting.py
        ;;
    *)
        echo "❌ Unknown notebook: $1"
        echo "Run './run.sh' without arguments to see available options"
        exit 1
        ;;
esac
