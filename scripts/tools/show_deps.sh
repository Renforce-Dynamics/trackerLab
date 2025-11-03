#!/bin/bash
set -e

# ============================================================
# show_deps.sh
# ------------------------------------------------------------
# Description:
#   List all pip-installed editable packages whose installation
#   paths contain a specific substring pattern (default: "trackerLab/source").
#
# Usage:
#   bash scripts/tools/show_deps.sh [pattern]
#
# Examples:
#   bash scripts/tools/show_deps.sh
#   bash scripts/tools/show_deps.sh IsaacLab/source
#   bash scripts/tools/show_deps.sh trackerLab/source
#
# Options:
#   -h, --help     Show this help message and exit
# ============================================================

# --- Help message ---
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    grep '^#' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

# --- Get pattern argument (default: trackerLab/source) ---
PATTERN=${1:-"trackerLab/source"}

echo "🔍 Collecting pip-installed editable packages matching pattern: '$PATTERN'"
echo "------------------------------------------------------------"

# --- Get current Python environment's pip executable ---
PIP_BIN=$(which pip)

# --- Check pip availability ---
if [ -z "$PIP_BIN" ]; then
    echo "❌ pip not found in current environment."
    exit 1
fi

# --- Extract package names and locations from pip list ---
# The script filters lines where the last column (location)
# contains the specified pattern, then prints both name and path.
$PIP_BIN list --format=columns | awk -v pattern="$PATTERN" '
NR > 2 && $0 ~ pattern {
    n = split($0, parts, /[[:space:]]+/)
    pkg = parts[1]
    path = parts[n]
    printf "✅ %-25s → %s\n", pkg, path
}
'

echo "------------------------------------------------------------"
echo "✅ Done."
