#!/bin/bash

# ========================================
# Download Unitree Robot Descriptions via Git Sparse-Checkout
# ========================================
# Downloads only specific folders (e.g., g1_description) without cloning the full repo.
# ========================================

REPO_URL="git@github.com:unitreerobotics/unitree_ros.git"
BRANCH="master"

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ASSET_DIR="${SCRIPT_DIR}/../../../data/assets"
USD_DIR="${ASSET_DIR}/usd"

mkdir -p "${ASSET_DIR}"
mkdir -p "${USD_DIR}"

show_help() {
    echo ""
    echo "Usage: $0 [robot_model ...]"
    echo ""
    echo "Downloads robot description folders (e.g. g1_description) into the script's directory."
    echo ""
    echo "Arguments:"
    echo "  robot_model         One or more robot types (e.g. g1 h1 a1)"
    echo ""
    echo "Options:"
    echo "  --help              Show this help message and exit"
    echo ""
    echo "Examples:"
    echo "  $0 g1"
    echo "  $0 g1 h1"
    echo ""
}

# Show help if needed
if [[ "$1" == "--help" || "$#" -eq 0 ]]; then
    show_help
    exit 0
fi

# Check for required tools
if ! command -v git &> /dev/null; then
    echo "❌ 'git' command not found. Please install git first."
    exit 1
fi

# Create temp working directory
TMP_DIR=$(mktemp -d)
echo "📁 Using temporary directory: $TMP_DIR"

cd "$TMP_DIR"
git init -q
git remote add origin "$REPO_URL"
git config core.sparseCheckout true
git config advice.detachedHead false

# Add sparse targets
SPARSE_FILE=".git/info/sparse-checkout"
> "$SPARSE_FILE"
for ROBOT in "$@"; do
    echo "robots/${ROBOT}_description/" >> "$SPARSE_FILE"
done

# Pull only needed content
git pull --depth 1 origin "$BRANCH" >/dev/null

# Copy and clean
for ROBOT in "$@"; do
    SRC="robots/${ROBOT}_description"
    DST="${ASSET_DIR}/${ROBOT}_description"
    echo "✅ Moving ${SRC} → ${DST}"
    rm -rf "$DST"
    cp -r "$SRC" "$DST"
done

# Clean up
cd "$SCRIPT_DIR"
rm -rf "$TMP_DIR"

echo ""
echo "🎉 All robot descriptions downloaded into: $SCRIPT_DIR"
