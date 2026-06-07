# Development History — Notebook Progression

These are the earlier iterations of the pipeline, kept to show how the system
evolved. **The final, current deliverable is [`../../notebook_v10_5.ipynb`](../../notebook_v10_5.ipynb)
in the repo root** — these are for context only.

| Version | Focus / what changed |
|---|---|
| `notebook_v3.ipynb` | First all-in-one Colab pass: video → frames → geometry → gsplat → SAM2 → semantics. Originally targeted MASt3R-SLAM for geometry. |
| `notebook_v5.ipynb` | Same structure, hardening the end-to-end run and fixing early pipeline breakages. |
| `notebook-v7.ipynb` | Reframed as "Room-Scale 3D Scene Reconstruction." Geometry pivoted to **COLMAP SfM** (MASt3R-SLAM proved unreliable to build on Colab) and the semantic-lifting stage was rebuilt. |
| `notebook_v8.ipynb` | Improved semantic lifting (2D→3D projection + majority vote) and CLIP open-vocabulary query path. |
| `notebook_v10.ipynb` | Added a proper **held-out evaluation** (train/test split, PSNR/SSIM/LPIPS on unseen frames) and richer artefact packaging. |
| **`notebook_v10_5.ipynb`** (root) | **Final.** Adds the **agentic QA & release gate** (deterministic stage-gate + bounded LLM reasoning), the canonicalised semantic labels, and the full evidence packages. |

See [`../../DEVELOPMENT.md`](../../DEVELOPMENT.md) for the narrative engineering
timeline behind these steps.
