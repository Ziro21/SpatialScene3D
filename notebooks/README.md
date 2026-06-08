# Colab Notebooks
This directory contains earlier/helper Google Colab notebooks for the
GPU-dependent stages.

> **The final end-to-end pipeline is [`../notebook_v10_5.ipynb`](../notebook_v10_5.ipynb)
> in the repo root.** The notebooks here are supporting/early entry points.

## Notebooks
- `colab_pipeline.ipynb` — Cloud pipeline: COLMAP → gsplat → Grounded-SAM-2

> Note: MASt3R-SLAM was the original geometry approach; the shipped pipeline uses
> COLMAP Structure-from-Motion. See [`../DESIGN.md`](../DESIGN.md) for the rationale.

## Workflow
1. Upload your video frames to Google Drive
2. Open the notebook in Colab
3. Follow the cells step by step
4. Download the outputs (COLMAP workspace, .ply, masks) to your Mac
