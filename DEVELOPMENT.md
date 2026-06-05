# Development Timeline

This document narrates how `scene3d` evolved from an initial multi-module local
prototype into the final, self-contained Colab notebook (`notebook_v10.ipynb`)
that produced the submitted results. It is written after the fact to give a
reviewer the engineering story that a tidy commit log alone cannot convey.

## Phase 1 — Local module prototype

The project started as a conventional Python package split across
`preprocess/`, `geometry/`, `semantics/`, `viewer/`, and `eval/`, with a
`Makefile`, pinned `requirements.txt`, and a pytest suite. The intended
architecture was a hybrid: lightweight preprocessing and the interactive viewer
on a local Mac (M1, 8 GB), with the GPU-heavy geometry, splatting, and
segmentation on Google Colab.

The original geometry plan was **MASt3R-SLAM** (CVPR 2025) for dense SLAM, as
argued in `DESIGN.md`. This is still the strongest forward-looking direction and
is documented there as the intended next step.

## Phase 2 — Colab migration and the move to COLMAP

Running MASt3R-SLAM end-to-end on Colab proved fragile: the `curope` and `imgui`
native extensions failed to build, dependency pins fought each other
(`moderngl-window` forced a bad NumPy downgrade), and the install was not
reproducible across Colab's Python 3.12 image. To get a **reliable, runnable
end-to-end pipeline**, the geometry front-end was switched to **COLMAP
Structure-from-Motion**, which is robust, well-documented, and integrates
directly with gsplat's expected data format.

The pipeline was consolidated into a single self-contained notebook so the whole
flow (frames → COLMAP → gsplat → Grounding DINO/SAM2 → semantic lifting → CLIP)
runs top-to-bottom on one Colab GPU. This is the `notebook_vN` progression
(v3 → v5 → v7 → v8 → v10).

## Phase 3 — The bugs that mattered

Three substantive correctness bugs were found and fixed during this phase. Each
materially changed the results.

1. **COLMAP `SIMPLE_RADIAL` camera intrinsics parsing.** The semantic-lifting
   code assumed a `PINHOLE` camera (`fx, fy, cx, cy`), but COLMAP had estimated a
   `SIMPLE_RADIAL` model (`f, cx, cy, k1`) — a completely different byte layout.
   Every projection was therefore using `fy = cx`, `cx = cy`, `cy = k1`, so 2D
   masks landed on the wrong Gaussians. Fixing this with a model-aware parser
   took semantic labelling from a broken state to correct projections.

2. **Coordinate-space mismatch (Gaussians vs cameras).** gsplat normalises the
   scene (centre + scale) during training, so the trained Gaussians live in a
   normalised space, while COLMAP camera poses are in the original space.
   Projecting one through the other produced garbage. The fix un-normalises the
   Gaussians back into COLMAP space for lifting, and, for evaluation, aligns the
   held-out camera poses to gsplat's space using a similarity transform
   (rotation + scale + translation) estimated from the matched training cameras.

3. **Honest held-out evaluation.** Before the coordinate fix, test-set PSNR was
   ~7 dB (meaningless). After it, held-out test PSNR matched training PSNR to
   within 0.14 dB (28.83 vs 28.69), confirming the metric is real and there is no
   train/test leakage.

Two semantic-quality fixes accompanied these:

- **Label canonicalisation (P3):** Grounding DINO emits overlapping compound
  detections (e.g. `shelf cabinet bookshelf`, `monitor television`). A
  canonicalisation map merges these into clean single-word classes (237 compound
  labels merged in the final run).
- **CLIP embedding ↔ semantic PLY ID alignment (P5):** the CLIP per-label
  embeddings are written with the *same* integer label IDs as the semantic PLY,
  so an open-vocabulary text query maps correctly onto 3D Gaussians.

## Phase 4 — Final run, evaluation, and packaging

The final run (`notebook_v10.ipynb`) processed 150 frames, trained on 135 and
held out 15 (every 10th registered frame) for evaluation. It produced:

- **Held-out test PSNR 28.83 dB, SSIM 0.901, LPIPS 0.059** — honest
  generalisation metrics on unseen frames.
- 484,460 Gaussians; 186,794 (38.6%) semantically labelled across 22 clean
  classes.
- An evaluation evidence bundle (`final_report_package/`): metrics JSON, label
  distribution and 2D→3D comparison tables, and ground-truth-vs-render
  comparison images.

## Phase 5 — Agentic workflow design (self-evaluating pipeline)

The task brief intentionally left the approach open and invited "any tools,
models, frameworks, or agentic workflows you find effective." Rather than bolt a
heavy multi-agent framework onto a finished pipeline (which tends to read as
decoration), the project takes a **hybrid stance**: the reconstruction core stays
deterministic and auditable, and a **bounded agentic layer** is added on top to
self-assess release-readiness.

This lives in `qa_supervisor.py` and runs as Section 10 of the notebook. It has
two tiers:

- **Deterministic stage-gate.** Auditable rules over the saved metrics produce a
  `PASS / WARNING / FAIL` verdict per stage (visual reconstruction, semantic
  lifting, mask quality, PLY quality, efficiency) and an overall release decision.
  Reproducible and never wrong about its own numbers.
- **LLM reasoning layer (the genuinely agentic part).** A single bounded,
  provider-agnostic LLM call (OpenAI-compatible; used here with Groq's Llama 3.3
  70B, swappable to OpenAI / Gemini / local Ollama) reasons over the same metrics
  plus the gate verdicts to diagnose the *dominant* weakness, identify the likely
  *root cause*, and recommend *one concrete next action*. It is constrained to
  cite the real numbers, and it degrades gracefully — with no API key the gate
  still runs and a saved diagnosis from a prior live run is shown.

On the final run the supervisor returns **RELEASE WITH LIMITATIONS**: visual
reconstruction, mask quality, and efficiency PASS; semantic lifting and raw-PLY
opacity are WARNINGs. The LLM layer correctly identifies incomplete semantic
coverage (flat-surface under-segmentation) as the dominant weakness.

Why this is the relevant creative angle for a humanoid/perception role: a real
perception system must know *when its own output is trustworthy* and when to
re-acquire data. This layer is a small, honest demonstration of exactly that —
the pipeline reasons about its own quality and gates its own release.

## Known limitations (carried into the submission honestly)

- ~61% of Gaussians remain unlabelled; large flat surfaces (walls) under-segment.
- The shipped geometry is COLMAP, not the MASt3R-SLAM stack argued for in
  `DESIGN.md`; that swap is scoped as future work.
- Per-stage runtime was not instrumented for the final run.
