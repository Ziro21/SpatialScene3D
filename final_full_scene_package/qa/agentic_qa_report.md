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

*Source: live LLM call.*

**Release Report**

1. **Overall Release Decision**: RELEASE WITH LIMITATIONS
2. **Per-Stage Verdicts**:
   - Visual reconstruction: PASS, with high PSNR (28.83 dB) and SSIM (0.901) values.
   - Semantic 3D lifting: WARNING, due to incomplete coverage (38.6% of Gaussians labelled).
   - 2D mask quality: PASS, with consistent coverage (21.5 masks/frame) and moderate open-vocab confidence (0.47).
   - Gaussian PLY quality: WARNING, due to 20.6% low-opacity floaters in the raw PLY.
   - Efficiency / packaging: PASS, with all evaluation artifacts and packages produced.
3. **Dominant Weakness and Root Cause**: The dominant weakness is the incomplete coverage of semantic 3D lifting, with a labelled fraction of only 38.57%. The root cause is likely the under-detection of large flat surfaces, such as ceilings (11.60% of total) and walls, which have low confidence detections (e.g., wall: 0.361 mean confidence). Specifically, classes like "curtain" (0.001% of total), "sink" (0.004% of total), and "fireplace" (0.015% of total) have very low coverage.
4. **Recommended Next Actions**:
   - Add detection prompts for under-detected classes like "curtain", "sink", and "fireplace" to improve their coverage.
   - Adjust the class thresholds for large flat surfaces like "ceiling" and "wall" to increase their detection confidence.
   - Implement a post-processing step to merge small, isolated Gaussians into larger, more coherent objects, which may help improve the coverage of classes like "table" (0.079% of total) and "lamp" (0.089% of total).
