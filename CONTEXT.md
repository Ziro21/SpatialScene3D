# Project Context & Conversation History

This document provides context on the development of the `scene3d` project, capturing the history of our conversations, design decisions, and the current state of the codebase. It is intended to provide immediate context for Claude or any other AI coding assistant.

## 1. Project Goal
The goal of this project is to build a 3D scene reconstruction system from monocular video for the **Humanoid Perception and Spatial AI Internship** challenge. The system takes a short video of an indoor area and reconstructs a 3D scene with optional semantic labels.

## 2. Architecture & Design Choices
We discussed and agreed upon the following architecture to handle the computationally intensive parts of 3D reconstruction while making the pipeline accessible:

- **Preprocessing (Local Mac):** Frame extraction from video with blur filtering and deduplication.
- **Geometry (Colab GPU):** MASt3R-SLAM is used for real-time dense SLAM to get camera poses and dense point clouds (chosen over VGGT for stability and to avoid frame limits).
- **Splatting (Colab GPU):** `gsplat` (3D Gaussian Splatting) is used to create a photorealistic 3D representation from the MASt3R-SLAM outputs.
- **Segmentation (Colab GPU):** Grounded-SAM-2 extracts instance masks for objects in the scene.
- **Semantic Lifting (Local Mac):** 2D masks are projected into 3D Gaussians. We use CLIP ViT-L/14 to generate embeddings for each semantic instance.
- **Viewer (Local Mac):** A `viser`-based web viewer is implemented to interact with the 3D scene, supporting RGB, Depth, Normals, Semantic, and Text-based CLIP queries.

*(See `DESIGN.md` for a full breakdown of trade-offs, like why we didn't use NeRF or COLMAP).*

## 3. Workflow & How to Run
A critical piece of context: **The project is intentionally split between local execution (Mac) and cloud execution (Google Colab).** 

If you try to run the heavy processing steps locally (`colab_pipeline.py` or large weight downloads), it will hang or fail due to the lack of an NVIDIA GPU.

**Correct Workflow:**
1. **Local:** Run `extract_frames.py` to prepare the dataset.
2. **Cloud:** Upload frames to Colab/Google Drive and run `fresh-collab.ipynb` on a Colab T4/A100 GPU.
3. **Local:** Download the outputs (`splat.ply`, `masks/`, `colmap/`) back to the Mac.
4. **Local:** Run `lift_to_3d.py`, `clip_embeddings.py`, and `viewer/app.py`.

## 4. Current State of the Codebase
- **Fully Implemented:** The entire system is coded, including preprocessing, Colab pipelines, semantic lifting, the interactive viewer, and 53 unit tests.
- **Recent Progress:** We spent the last few hours heavily debugging and fixing the Colab notebook (`fresh-collab.ipynb`) for processing `scene2`.
  - **Fixed MASt3R-SLAM Installation:** Bypassed the failing `curope` and `imgui` extensions using `--no-deps`.
  - **Fixed Missing Dependencies:** Manually added `faiss-cpu`, `trimesh`, and `roma` to the pip install cell. We deliberately avoided adding `moderngl-window` as it forces a bad `numpy` downgrade.
  - **Fixed Fallback Bug:** Changed the `.ply` search to strictly look in `SLAM_LOGS` so it stops accidentally training on the sample `mustard_bottle.ply`.
  - **Fixed Glob Search:** Added `recursive=True` to the gsplat checkpoint search.
  - **Fixed Hardcoded Paths:** Changed the final zip cell to dynamically zip `{DRIVE_OUTPUT}`.
- **Current Status:** As of right now, the Colab notebook is successfully executing! MASt3R-SLAM successfully generated the room's point cloud, and `gsplat` is currently actively training in the background.

## 5. Next Steps for Claude
- The user is currently waiting for their Colab notebook to finish running `gsplat` and Grounded-SAM-2.
- Once finished, the user will download `scene2_outputs.zip` to their Mac.
- **Help the user run the local semantic lifting pipeline and launch the 3D viewer.**
- Finalize the project by populating the `README.md` with evaluation metrics (`metrics.py`) and an animated demo GIF.
