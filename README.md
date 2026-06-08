# SpatialScene3D

> **Self-Evaluating Spatial AI Pipeline** — from a monocular indoor phone video to
> an interactive, semantically-labelled 3D Gaussian Splat with open-vocabulary text
> queries, plus an agentic QA layer that assesses its own release-readiness.

---

## What this is

A complete, working pipeline that takes a short indoor room video and reconstructs
a dense 3D scene with semantic understanding, then **evaluates its own output**:

```
indoor video → COLMAP SfM → 3D Gaussian Splatting → Grounding DINO + SAM2 masks
            → 2D-to-3D semantic lifting → CLIP embeddings → agentic QA & release gate
```

The end-to-end pipeline runs in a single self-contained Colab notebook
([`notebook_v10_5.ipynb`](notebook_v10_5.ipynb)). A local Python package
(`preprocess/`, `semantics/`, `viewer/`, `eval/`) provides the supporting modules
and the interactive viewer.

> **Implementation note (honest):** the shipped geometry front-end is **COLMAP
> Structure-from-Motion**, not MASt3R-SLAM. MASt3R-SLAM was the original design
> (see [DESIGN.md](DESIGN.md)) but proved unreliable to build on Colab; COLMAP was
> chosen to deliver a robust, reproducible end-to-end pipeline. The MASt3R-SLAM
> swap remains scoped as future work. See [DEVELOPMENT.md](DEVELOPMENT.md) for the
> full engineering story.

---

## Results (held-out evaluation — final run, v10.5)

Metrics are computed on **15 held-out frames never seen during training** (every
10th registered frame), with a gsplat-consistent camera-pose alignment. These are
honest generalisation numbers, not training-set scores.

| Metric | Value |
|---|---:|
| Held-out test PSNR ↑ | **28.93 dB** |
| Held-out test SSIM ↑ | **0.903** |
| Held-out test LPIPS ↓ | **0.058** |
| Frames used (train / test) | 150 (135 / 15) |
| Final Gaussian count | 484,707 |
| Semantically labelled Gaussians | 181,516 / 484,707 = **37.4%** |
| Unique 3D semantic classes | 22 |
| 2D masks generated | 3,222 across 150 frames |

The held-out test PSNR (28.93 dB) essentially matches the training PSNR, confirming
no train/test leakage. Full evidence — metrics JSON, per-class tables, and
ground-truth-vs-render comparison images — is in
[`final_report_package v10.5/`](final_report_package%20v10.5). The evaluation is
written up in two files there: the auto-generated
[`final_evaluation_summary.md`](final_report_package%20v10.5/report/final_evaluation_summary.md)
(numeric metrics) and
[`EVALUATION_ADDENDUM.md`](final_report_package%20v10.5/report/EVALUATION_ADDENDUM.md)
(open-vocabulary CLIP search, packaging/accessibility, and the agentic QA gate).

---

## Agentic Spatial QA & Release Gate

A bounded **agentic layer** ([`qa_supervisor.py`](qa_supervisor.py), Section 10 of
the notebook) sits on top of the deterministic reconstruction pipeline and decides
whether a run is release-ready. Two tiers:

1. **Deterministic stage-gate** — auditable rules over the saved metrics produce a
   `PASS / WARNING / FAIL` verdict per stage and an overall release decision.
   Reproducible; never wrong about its own numbers.
2. **LLM reasoning layer** — a single bounded, provider-agnostic LLM call (Groq /
   OpenAI / Gemini / local Ollama) reasons over the metrics + per-class tables to
   diagnose the *dominant* weakness, name the *specific* under-covered classes, and
   recommend concrete next actions. This is the genuinely agentic part: open-ended
   reasoning over heterogeneous evidence a fixed threshold cannot do.

On the final run it returns **RELEASE WITH LIMITATIONS** (visual reconstruction,
mask quality, efficiency PASS; semantic coverage and raw-PLY opacity WARNING). The
report is saved in each evidence package under `qa/`.

```bash
# Run the QA layer over a run's metrics (deterministic gate always works):
python qa_supervisor.py \
  --metrics_dir "final_full_scene_package v10.5/metrics" \
  --output_dir  "final_full_scene_package v10.5/qa"

# Enable the live LLM reasoning layer (free Groq key from console.groq.com):
export LLM_API_KEY="your_key"      # or put it in a local .env (gitignored)
# Provider/model are configurable: LLM_BASE_URL, LLM_MODEL
```

Why this matters for spatial AI: a perception system must know *when its own output
is trustworthy*. This layer is a small, honest demonstration of that self-assessment
step.

---

## Large artefacts

The reconstruction produces PLYs that exceed GitHub's 100 MB/file limit, so they are
distributed as follows:

| Artefact | Size | Where |
|---|---:|---|
| Pruned **viewer** PLY | 90 MB | In-repo via **Git LFS** (`final_full_scene_package v10.5/scene_outputs/`) |
| **Semantic** PLY (labelled) | 117 MB | GitHub **Release** (see below) |
| Raw full PLY | 115 MB | GitHub **Release** |
| CLIP embeddings | 68 KB | In-repo (`video1-final/outputs/embeddings.npz`) |
| 2D masks archive | 2.5 MB | In-repo (`video1-final/outputs/masks.zip`) |

> The semantic + raw PLYs are attached to the latest **GitHub Release**:
> https://github.com/Ziro21/SpatialScene3D/releases/latest

---

## Pipeline (how to reproduce)

**End-to-end (recommended):** open [`notebook_v10_5.ipynb`](notebook_v10_5.ipynb)
in Google Colab (GPU runtime), set the scene paths in the config cell, and run all.
It performs frame extraction → COLMAP → gsplat → Grounding DINO/SAM2 → semantic
lifting → CLIP → packaging → agentic QA.

**Local supporting tools (Mac):**

```bash
git clone https://github.com/Ziro21/SpatialScene3D.git
cd SpatialScene3D
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Interactive viewer over a downloaded semantic PLY:
python -m viewer.app --scene scene1
# → http://localhost:8080   (RGB / Depth / Normals / Semantic / CLIP text query)
```

---

## System Architecture

```
Phone Video (.mp4)
       │  Mac (CPU): ffmpeg → frames, blur filter + dedup
       ▼
┌──────────────────────────────┐
│  GEOMETRY — COLMAP SfM        │  Colab GPU: camera poses + sparse points
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  GAUSSIAN SPLATTING — gsplat  │  Colab GPU: photorealistic 3D Gaussians (.ply)
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  SEGMENTATION — Grounded-SAM2 │  Colab GPU: per-frame open-vocab masks
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  SEMANTIC LIFTING + CLIP      │  2D masks → per-Gaussian labels; CLIP per class
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  AGENTIC QA & RELEASE GATE    │  self-assess metrics → release decision
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  INTERACTIVE VIEWER (viser)   │  Mac: RGB / Depth / Normals / Semantic / Text
└──────────────────────────────┘
```

---

## Viewer Render Modes

| Mode | Description |
|------|-------------|
| **RGB** | Original Gaussian splat colours — photorealistic novel views |
| **Depth** | Viridis colourmap — distance from scene centroid |
| **Normals** | PCA-estimated surface normals as RGB |
| **Semantic** | Gaussians coloured by class label (sofa, door, ceiling, …) |
| **Text Query** | Type any text → CLIP heatmap highlights matching objects ([demo](final_report_package%20v10.5/report/CLIP_QUERY_DEMO.md)) |

---

## Repository layout

```
SpatialScene3D/
├── notebook_v10_5.ipynb            # Final end-to-end Colab pipeline (THE deliverable)
├── qa_supervisor.py                # Agentic QA & release gate
├── convert.py                      # PLY → .splat helper (drag-drop web viewer)
├── preprocess/  geometry/  semantics/  viewer/  eval/   # Local Python package
├── notebooks/                      # Original cloud-pipeline entry point + helpers
├── tests/                          # pytest suite for the local modules (53 tests)
├── final_report_package v10.5/     # Lightweight evidence (metrics, tables, images, qa)
├── final_full_scene_package v10.5/ # Full evidence + viewer PLY (LFS)
├── video1-final/outputs/           # CLIP embeddings + masks archive
├── assets/videos/                  # Example input videos (LFS)
├── archive/notebooks/              # Earlier notebook versions (dev history + index)
├── DESIGN.md                       # Technical design memo (incl. MASt3R rationale)
├── DEVELOPMENT.md                  # Engineering timeline & honest decisions
├── README.md  requirements.txt  Makefile  run.sh  download_weights.sh
```

> Development history (v3 → v10.5) is preserved in
> [`archive/notebooks/`](archive/notebooks/) with a progression index.

---

## Design & development docs

- **[DESIGN.md](DESIGN.md)** — problem framing, alternatives (COLMAP / VGGT / NeRF /
  MASt3R-SLAM), trade-offs, robotics relevance, limitations.
- **[DEVELOPMENT.md](DEVELOPMENT.md)** — the engineering journey: local prototype →
  Colab migration to COLMAP → the three substantive bug fixes → final run →
  agentic QA layer. Includes an honest account of what was shipped vs. planned.

---

## Technology Stack

| Component | Library | Runs Where |
|---|---|---|
| Geometry backbone | **COLMAP** SfM | ☁️ Colab GPU |
| Scene representation | gsplat 1.3.0 (3DGS) | ☁️ Colab GPU |
| Segmentation | Grounded-SAM-2 (DINO + SAM 2) | ☁️ Colab GPU |
| Text embeddings | CLIP ViT-L/14 | ☁️ Colab GPU |
| Semantic lifting | Custom projection + multi-view voting | ☁️ Colab / 💻 Mac |
| Agentic QA | Deterministic gate + LLM (Groq/OpenAI/Gemini/Ollama) | 💻 anywhere |
| Interactive viewer | viser | 💻 Mac (browser) |
| Preprocessing | ffmpeg + OpenCV | 💻 Mac (CPU) |

---

## Testing

```bash
make test            # or: python -m pytest tests/ -v
```

> Note: the pytest suite covers the **local Python modules** (`preprocess/`,
> `geometry/`, `semantics/`, `viewer/`, `eval/`). The final results were produced
> by the Colab notebook, which is validated by the held-out evaluation in Section
> 5c and the agentic QA gate in Section 10 rather than by these unit tests.

---

## References

1. Schönberger & Frahm, *COLMAP: Structure-from-Motion Revisited*, CVPR 2016
2. Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023
3. Ren et al., *Grounded SAM 2: Ground and Track Anything in Videos*, IDEA-Research 2024
4. Ravi et al., *SAM 2: Segment Anything in Images and Videos*, Meta AI 2024
5. Radford et al., *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*, ICML 2021
6. Qin et al., *LangSplat: 3D Language Gaussian Splatting*, CVPR 2024
7. Murai et al., *MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors*, CVPR 2025 [[arXiv:2412.12392](https://arxiv.org/abs/2412.12392)] — *original design target, future work*
8. Wang et al., *VGGT: Visual Geometry Grounded Transformer*, CVPR 2025 [[arXiv:2503.11651](https://arxiv.org/abs/2503.11651)]

---

## License

Code under Apache 2.0. Model weights under their respective licenses (COLMAP: BSD;
Grounded-SAM-2: Apache 2.0; CLIP: MIT).

Author: Zeyad Khalil.
