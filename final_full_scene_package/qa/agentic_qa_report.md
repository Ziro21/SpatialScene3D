# Agentic Spatial QA — Release Report

**Final release decision: RELEASE WITH LIMITATIONS**  (overall verdict: WARNING)

## Deterministic stage gate

| Stage | Verdict | Reason |
|---|---|---|
| Visual reconstruction | **PASS** | Held-out PSNR 28.83 dB, SSIM 0.901, LPIPS 0.059 on 15 unseen frames. |
| Semantic 3D lifting | **WARNING** | 38.6% of Gaussians labelled across 22 classes. Meaningful but incomplete coverage (large flat surfaces under-segment). |
| 2D mask quality | **PASS** | 150/150 frames produced masks (21.5/frame), mean confidence 0.47 — consistent coverage, moderate open-vocab confidence. |
| Gaussian PLY quality | **WARNING** | Raw PLY has 20.6% low-opacity floaters; the pruned viewer PLY removes them (now 0.0%). Viewer output is clean. |
| Efficiency / packaging | **PASS** | 150 frames -> 484460 Gaussians; all evaluation artefacts and packages produced. Runtime not instrumented for this run. |

## Agentic diagnosis (LLM reasoning layer)

*Source: genuine live LLM call (Groq Llama 3.3 70B) on 2026-06-06.*

**Release Report**

1. **Overall Release Decision**: RELEASE WITH LIMITATIONS
2. **Per-Stage Verdicts**:
   - Visual reconstruction: PASS, with held-out PSNR 28.83 dB, SSIM 0.901, LPIPS 0.059 on 15 unseen frames.
   - Semantic 3D lifting: WARNING, due to 38.6% of Gaussians labelled across 22 classes, indicating incomplete coverage.
   - 2D mask quality: PASS, with 150/150 frames producing masks and a mean confidence of 0.47.
   - Gaussian PLY quality: WARNING, due to 20.6% low-opacity floaters in the raw PLY, although the pruned viewer PLY removes them.
   - Efficiency / packaging: PASS, with all evaluation artifacts and packages produced.
3. **Dominant Weakness and Root Cause**: The dominant weakness is the incomplete coverage of semantic 3D lifting, particularly for large flat surfaces. The root cause is likely the under-segmentation of classes such as "ceiling" (11.597% of total) and "wall" (low confidence, 0.361 mean confidence). Specific object/structure classes responsible include "curtain" (0.001% of total), "sink" (0.004% of total), and "fireplace" (0.015% of total).
4. **Recommended Next Actions**:
   - Add detection prompts for under-segmented classes like "curtain", "sink", and "fireplace" to improve their coverage.
   - Adjust the class thresholds for "wall" and "ceiling" to increase their confidence and coverage.
   - Implement a post-processing step to merge small, low-confidence segments into larger, more coherent regions, particularly for classes like "table" (0.079% of total) and "lamp" (0.089% of total).
