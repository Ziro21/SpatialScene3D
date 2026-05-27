#!/usr/bin/env bash
# ============================================================
# run.sh — One-command pipeline: video → 3D semantic scene
# ============================================================
# Usage:
#   bash run.sh path/to/video.mp4 [scene_name]
#
# Example:
#   bash run.sh assets/videos/bedroom.mp4 bedroom
#
# This script runs the LOCAL stages only (preprocessing,
# semantic lifting, CLIP embeddings, and launches the viewer).
# MASt3R-SLAM and gsplat training must be run on Colab first.
# See notebooks/colab_pipeline.py for the cloud steps.
# ============================================================

set -euo pipefail

# --- Parse Arguments ---
VIDEO_PATH="${1:?Usage: bash run.sh <video_path> [scene_name]}"
SCENE_NAME="${2:-scene1}"

echo ""
echo "============================================="
echo " scene3d — 3D Scene Reconstruction Pipeline"
echo "============================================="
echo " Video:  ${VIDEO_PATH}"
echo " Scene:  ${SCENE_NAME}"
echo "============================================="
echo ""

# --- Step 1: Preprocess (extract + filter frames) ---
echo "[1/6] Preprocessing video → frames..."
python -m preprocess.extract_frames \
    --input "${VIDEO_PATH}" \
    --output "data/${SCENE_NAME}/frames" \
    --fps 3 \
    --height 518
echo "  ✓ Frames saved to data/${SCENE_NAME}/frames/"
echo ""

# --- Step 2 & 3: Colab Checks ---
COLMAP_DIR="data/${SCENE_NAME}/colmap"
PLY_FILE="outputs/${SCENE_NAME}/splat.ply"
MASKS_DIR="data/${SCENE_NAME}/masks"

MISSING_DATA=0

if [ ! -d "${COLMAP_DIR}" ]; then
    echo "[2/6] ⚠ COLMAP workspace not found at ${COLMAP_DIR}/"
    MISSING_DATA=1
else
    echo "[2/6] ✓ COLMAP workspace found at ${COLMAP_DIR}/"
fi

if [ ! -f "${PLY_FILE}" ]; then
    echo "[3/6] ⚠ Gaussian splat not found at ${PLY_FILE}"
    MISSING_DATA=1
else
    echo "[3/6] ✓ Gaussian splat found at ${PLY_FILE}"
fi

if [ ! -d "${MASKS_DIR}" ]; then
    echo "[4/6] ⚠ Masks not found at ${MASKS_DIR}/"
    MISSING_DATA=1
fi

if [ $MISSING_DATA -eq 1 ]; then
    echo ""
    echo "  → Required Colab outputs are missing."
    echo "  → Run the Colab pipeline first (see notebooks/colab_pipeline.py)"
    echo "  → Download splat.ply, colmap/ and masks/ into the project"
    echo "  → Then re-run this script."
    exit 1
fi
echo ""

# --- Step 4: Semantic Lifting ---
SEMANTIC_PLY="outputs/${SCENE_NAME}/splat_semantic.ply"
if [ ! -f "${SEMANTIC_PLY}" ]; then
    echo "[4/6] Lifting 2D masks → 3D Gaussian labels..."
    python -m semantics.lift_to_3d \
        --splat "${PLY_FILE}" \
        --masks "${MASKS_DIR}" \
        --colmap "${COLMAP_DIR}" \
        --output "${SEMANTIC_PLY}"
    echo "  ✓ Semantic PLY saved to ${SEMANTIC_PLY}"
else
    echo "[4/6] ✓ Semantic PLY already computed at ${SEMANTIC_PLY}"
fi
echo ""

# --- Step 5: CLIP Embeddings ---
EMBEDDINGS_FILE="outputs/${SCENE_NAME}/embeddings.npz"
if [ ! -f "${EMBEDDINGS_FILE}" ]; then
    echo "[5/6] Computing CLIP embeddings..."
    python -m semantics.clip_embeddings \
        --masks "${MASKS_DIR}" \
        --frames "${COLMAP_DIR}/images" \
        --output "${EMBEDDINGS_FILE}"
    echo "  ✓ Embeddings saved to ${EMBEDDINGS_FILE}"
else
    echo "[5/6] ✓ Embeddings already computed at ${EMBEDDINGS_FILE}"
fi
echo ""

# --- Step 6: Launch Viewer ---
echo "[6/6] Launching interactive viewer..."
echo "  Open http://localhost:8080 in your browser"
echo ""
python -m viewer.app --scene "${SCENE_NAME}"
