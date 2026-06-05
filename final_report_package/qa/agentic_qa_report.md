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
   - Visual reconstruction: PASS (Held-out PSNR 28.83 dB, SSIM 0.901, LPIPS 0.059 on 15 unseen frames)
   - Semantic 3D lifting: WARNING (38.6% of Gaussians labelled across 22 classes)
   - 2D mask quality: PASS (150/150 frames produced masks, mean confidence 0.47)
   - Gaussian PLY quality: WARNING (20.6% low-opacity floaters in raw PLY, removed in pruned viewer PLY)
   - Efficiency / packaging: PASS (150 frames -> 484460 Gaussians, all evaluation artefacts and packages produced)
3. **Dominant Weakness and Root Cause**: The dominant weakness is the incomplete semantic 3D lifting, with only 38.57% of Gaussians labelled. The root cause is likely the under-detection of certain classes, particularly "curtain" (0.001% of total), "sink" (0.004% of total), and "fireplace" (0.015% of total). These classes have low coverage, indicating that the model may not be effectively detecting these objects.
4. **Recommended Next Actions**:
   - Add detection prompts for "curtain", "sink", and "fireplace" to improve their detection rates.
   - Adjust the class thresholds for "wall" (mean confidence 0.361) and "mirror" (mean confidence 0.362) to improve their detection reliability.
   - Increase the number of training samples for "table" (0.079% of total) and "lamp" (0.089% of total) to improve their 3D coverage.
