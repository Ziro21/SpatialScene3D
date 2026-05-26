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
# See notebooks/colab_pipeline.ipynb for the cloud steps.
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
echo "[1/5] Preprocessing video → frames..."
python -m preprocess.extract_frames \
    --video "${VIDEO_PATH}" \
    --output_dir "data/${SCENE_NAME}/frames" \
    --fps 3 \
    --height 518
echo "  ✓ Frames saved to data/${SCENE_NAME}/frames/"
echo ""

# --- Step 2: Geometry (MASt3R-SLAM) ---
# This step runs on Colab. Check if COLMAP output already exists.
COLMAP_DIR="data/${SCENE_NAME}/colmap"
if [ ! -d "${COLMAP_DIR}" ]; then
    echo "[2/5] ⚠ COLMAP workspace not found at ${COLMAP_DIR}/"
    echo "  → Run MASt3R-SLAM on Colab first (see notebooks/colab_pipeline.ipynb)"
    echo "  → Download the COLMAP workspace to ${COLMAP_DIR}/"
    echo "  → Then re-run this script."
    exit 1
fi
echo "[2/5] ✓ COLMAP workspace found at ${COLMAP_DIR}/"
echo ""

# --- Step 3: 3DGS (gsplat) ---
# This step also runs on Colab. Check if .ply exists.
PLY_FILE="outputs/${SCENE_NAME}/splat.ply"
if [ ! -f "${PLY_FILE}" ]; then
    echo "[3/5] ⚠ Gaussian splat not found at ${PLY_FILE}"
    echo "  → Train gsplat on Colab first (see notebooks/colab_pipeline.ipynb)"
    echo "  → Download splat.ply to ${PLY_FILE}"
    echo "  → Then re-run this script."
    exit 1
fi
echo "[3/5] ✓ Gaussian splat found at ${PLY_FILE}"
echo ""

# --- Step 4: Semantics (Grounded-SAM-2 + 3D lift + CLIP) ---
SEMANTIC_DIR="data/${SCENE_NAME}/semantics"
if [ ! -d "${SEMANTIC_DIR}" ]; then
    echo "[4/5] Running semantic pipeline..."
    
    # Check if masks exist (from Colab) or need to run locally
    MASKS_DIR="data/${SCENE_NAME}/masks"
    if [ ! -d "${MASKS_DIR}" ]; then
        echo "  ⚠ Masks not found at ${MASKS_DIR}/"
        echo "  → Run Grounded-SAM-2 on Colab first (see notebooks/colab_pipeline.ipynb)"
        echo "  → Download masks to ${MASKS_DIR}/"
        echo "  → Then re-run this script."
        exit 1
    fi
    
    echo "  Lifting 2D masks → 3D Gaussian labels..."
    python -m semantics.lift_to_3d \
        --scene_dir "data/${SCENE_NAME}" \
        --ply_path "${PLY_FILE}" \
        --output_dir "${SEMANTIC_DIR}"
    
    echo "  Computing CLIP embeddings..."
    python -m semantics.clip_embeddings \
        --scene_dir "data/${SCENE_NAME}" \
        --output_dir "${SEMANTIC_DIR}"
    
    echo "  ✓ Semantics saved to ${SEMANTIC_DIR}/"
else
    echo "[4/5] ✓ Semantics already computed at ${SEMANTIC_DIR}/"
fi
echo ""

# --- Step 5: Launch Viewer ---
echo "[5/5] Launching interactive viewer..."
echo "  Open http://localhost:8080 in your browser"
echo ""
python -m viewer.app \
    --scene_dir "data/${SCENE_NAME}" \
    --ply_path "${PLY_FILE}" \
    --semantic_dir "${SEMANTIC_DIR}"
