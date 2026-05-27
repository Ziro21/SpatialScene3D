# scene3d

> **3D Scene Reconstruction from Monocular Video** — From a phone video to an interactive, semantically-labelled 3D Gaussian Splat with open-vocabulary text queries.

<!-- TODO: Insert hero GIF here after recording demo -->
<!-- ![Hero Demo](assets/outputs/hero_demo.gif) -->

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Ziro21/scene3d.git
cd scene3d
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Preprocess a video (runs on Mac, CPU only)
python -m preprocess.extract_frames \
    --input assets/videos/my_scene.mp4 \
    --output data/scene1/frames \
    --fps 3 --height 518

# 3. Upload frames to Google Drive, run Colab notebook
# → See notebooks/colab_pipeline.py
# → Downloads: splat.ply, masks/, colmap/ → data/scene1/

# 4. Semantic lifting (runs on Mac)
python -m semantics.lift_to_3d \
    --splat outputs/scene1/splat.ply \
    --masks data/scene1/masks \
    --colmap data/scene1/colmap \
    --output outputs/scene1/splat_semantic.ply

# 5. CLIP embeddings (runs on Mac, MPS)
python -m semantics.clip_embeddings \
    --masks data/scene1/masks \
    --frames data/scene1/colmap/images \
    --output outputs/scene1/embeddings.npz

# 6. Launch interactive viewer
python -m viewer.app --scene scene1
# → Open http://localhost:8080 in your browser

# 7. Evaluate
python -m eval.metrics --scene scene1
```

Or use the one-command script:
```bash
bash run.sh assets/videos/my_scene.mp4 scene1
```

---

## System Architecture

```
Phone Video (.mp4)
       │
       ▼
┌─────────────────────────┐
│  1. PREPROCESS          │  Mac (CPU)
│  ffmpeg → frames        │  Blur filter + dedup + exposure norm
│  extract_frames.py      │
└──────────┬──────────────┘
           │ Upload to Google Drive
           ▼
┌─────────────────────────┐
│  2. GEOMETRY            │  Colab GPU (T4/A100)
│  MASt3R-SLAM            │  Camera poses + dense pointcloud
│  → COLMAP format export │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  3. GAUSSIAN SPLATTING  │  Colab GPU
│  gsplat 1.3.0           │  Photorealistic 3D Gaussians (.ply)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  4. SEGMENTATION        │  Colab GPU
│  Grounded-SAM-2         │  Per-frame instance masks
│  (DINO + SAM 2)         │
└──────────┬──────────────┘
           │ Download to Mac
           ▼
┌─────────────────────────┐
│  5. SEMANTIC LIFTING    │  Mac (CPU/MPS)
│  lift_to_3d.py          │  Project masks → per-Gaussian labels
│  clip_embeddings.py     │  CLIP ViT-L/14 per instance
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  6. INTERACTIVE VIEWER  │  Mac (browser)
│  viser (localhost:8080) │  RGB / Depth / Normals / Semantic
│  + Text Query (CLIP)    │  + Top-down occupancy map
└─────────────────────────┘
```

---

## Viewer Render Modes

| Mode | Description |
|------|-------------|
| **RGB** | Original Gaussian splat colours — photorealistic novel views |
| **Depth** | Viridis colourmap — distance from scene centroid |
| **Normals** | PCA-estimated surface normals as RGB |
| **Semantic** | Gaussians coloured by class label (chair, table, wall...) |
| **Text Query** | Type any text → CLIP heatmap highlights matching objects |

---

## Results

<!-- Populated by: python -m eval.metrics --scene scene1 -->
| Scene | PSNR ↑ | SSIM ↑ | Chamfer ↓ (m) | Precision@10 |
|-------|--------|--------|---------------|--------------|
| Scene 1 | — | — | — | — |
| Scene 2 | — | — | — | — |

> Results will be populated after running the full pipeline with `python -m eval.metrics`.

---

## Project Structure

```
scene3d/
├── preprocess/
│   └── extract_frames.py      # Video → frames (ffmpeg + blur + dedup)
├── geometry/
│   ├── colmap_utils.py        # COLMAP binary format writers
│   └── export_colmap.py       # MASt3R-SLAM → COLMAP conversion
├── semantics/
│   ├── lift_to_3d.py          # 2D masks → per-Gaussian 3D labels
│   └── clip_embeddings.py     # CLIP ViT-L/14 per instance
├── viewer/
│   └── app.py                 # viser interactive viewer (5 modes)
├── eval/
│   ├── metrics.py             # PSNR, SSIM, Chamfer, Precision@K
│   └── results.json           # Auto-populated evaluation results
├── notebooks/
│   └── colab_pipeline.py      # GPU pipeline (MASt3R-SLAM + gsplat + SAM)
├── tests/
│   ├── test_preprocess.py     # 5 tests
│   ├── test_geometry.py       # 7 tests
│   ├── test_semantics.py      # 11 tests
│   ├── test_viewer.py         # 15 tests
│   └── test_eval.py           # 15 tests  ← 53 total
├── run.sh                     # One-command pipeline
├── Makefile                   # make test | lint | demo | eval
├── requirements.txt           # Pinned dependencies
├── README.md                  # This file
└── DESIGN.md                  # Technical design memo (3 pages)
```

---

## Design Choices

See [DESIGN.md](DESIGN.md) for a detailed discussion covering:

1. **Problem Framing** — Why indoor monocular 3D is hard
2. **Alternatives Considered** — COLMAP, VGGT, NeRF vs. MASt3R-SLAM
3. **Decisions & Trade-offs** — Geometry, representation, and semantics choices
4. **Robotics Relevance** — How each component maps to humanoid robot perception
5. **Limitations & Future Work** — Honest failure cases and next steps

---

## Technology Stack

| Component | Library | Runs Where |
|---|---|---|
| Geometry backbone | MASt3R-SLAM (CVPR 2025) | ☁️ Colab GPU |
| Scene representation | gsplat 1.3.0 (3DGS) | ☁️ Colab GPU |
| Segmentation | Grounded-SAM-2 (DINO + SAM 2) | ☁️ Colab GPU |
| Text embeddings | CLIP ViT-L/14 | 💻 Mac (MPS) |
| Semantic lifting | Custom projection + voting | 💻 Mac (CPU) |
| Interactive viewer | viser | 💻 Mac (browser) |
| Preprocessing | ffmpeg + OpenCV | 💻 Mac (CPU) |
| Testing | pytest (53 tests) | 💻 Mac |

---

## Testing

```bash
# Run all 53 tests
make test
# or
python -m pytest tests/ -v
```

| Module | Tests | What's Covered |
|--------|-------|----------------|
| Preprocessing | 5 | Frame extraction, blur detection, pipeline |
| Geometry | 7 | Quaternion math, COLMAP binary writers |
| Semantics | 11 | Projection, mask lookup, PLY I/O, cropping |
| Viewer | 15 | Colour modes, PLY loading, occupancy maps |
| Evaluation | 15 | PSNR, SSIM, Chamfer, results serialisation |

---

## References

1. Murai et al., *MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors*, CVPR 2025 [[arXiv:2412.12392](https://arxiv.org/abs/2412.12392)]
2. Wang et al., *VGGT: Visual Geometry Grounded Transformer*, CVPR 2025 Best Paper [[arXiv:2503.11651](https://arxiv.org/abs/2503.11651)]
3. Leroy et al., *DUSt3R / MASt3R: Dense Unconstrained 3D Reconstruction*, CVPR 2024
4. Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023
5. Ren et al., *Grounded SAM 2: Ground and Track Anything in Videos*, IDEA-Research 2024
6. Ravi et al., *SAM 2: Segment Anything in Images and Videos*, Meta AI 2024
7. Radford et al., *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*, ICML 2021
8. **Yoo et al., *OpenMonoGS-SLAM*, arXiv:2512.08625, December 2025** ← closest prior art
9. Wang et al., *CUT3R*, CVPR 2025 Oral [[arXiv:2501.12387](https://arxiv.org/abs/2501.12387)]
10. Yang et al., *Depth Anything V2*, NeurIPS 2024
11. Qin et al., *LangSplat*, CVPR 2024
12. Schönberger & Frahm, *COLMAP: Structure-from-Motion Revisited*, CVPR 2016

---

## License

- **Code**: Apache 2.0
- **MASt3R-SLAM weights**: Apache 2.0 ✅
- **Grounded-SAM-2 weights**: Apache 2.0 ✅
- **CLIP weights**: MIT ✅

---

## Acknowledgements

Built for the Humanoid Internship Challenge — Perception & Spatial AI, London 2025.
