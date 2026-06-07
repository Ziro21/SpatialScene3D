# Agentic Spatial QA — Release Report

**Final release decision: RELEASE WITH LIMITATIONS**  (overall verdict: WARNING)

## Deterministic stage gate

| Stage | Verdict | Reason |
|---|---|---|
| Visual reconstruction | **PASS** | Held-out PSNR 28.93 dB, SSIM 0.903, LPIPS 0.058 on 15 unseen frames. |
| Semantic 3D lifting | **WARNING** | 37.4% of Gaussians labelled across 22 classes. Meaningful but incomplete coverage (large flat surfaces under-segment). |
| 2D mask quality | **PASS** | 150/150 frames produced masks (21.5/frame), mean confidence 0.47 — consistent coverage, moderate open-vocab confidence. |
| Gaussian PLY quality | **WARNING** | Raw PLY has 20.8% low-opacity floaters; the pruned viewer PLY removes them (now 0.0%). Viewer output is clean. |
| Efficiency / packaging | **PASS** | 150 frames -> 484707 Gaussians; all evaluation artefacts and packages produced. Runtime not instrumented for this run. |

## Agentic diagnosis (LLM reasoning layer)

*Source: genuine live LLM call (Groq Llama 3.3 70B) on 2026-06-07.*

**Release Decision:** RELEASE WITH LIMITATIONS

**Stage Verdicts:**
1. Visual reconstruction: PASS - Held-out PSNR 28.93 dB, SSIM 0.903, LPIPS 0.058 on 15 unseen frames.
2. Semantic 3D lifting: WARNING - 37.4% of Gaussians labelled across 22 classes, with incomplete coverage.
3. 2D mask quality: PASS - 150/150 frames produced masks, mean confidence 0.47.
4. Gaussian PLY quality: WARNING - Raw PLY has 20.8% low-opacity floaters, pruned viewer PLY removes them.
5. Efficiency / packaging: PASS - 150 frames -> 484707 Gaussians, all evaluation artefacts and packages produced.

**Dominant Weakness and Root Cause:**
The dominant weakness is the incomplete semantic 3D lifting coverage, particularly for large flat surfaces. The root cause is likely the under-detection of certain classes, such as "curtain" (0.001% of total), "sink" (0.004% of total), and "fireplace" (0.027% of total). These classes have low coverage, indicating that the model may not be effectively detecting and labelling these objects.

**Recommended Next Actions:**
1. Add detection prompts for under-detected classes, such as "curtain", "sink", and "fireplace", to improve their coverage.
2. Adjust the class thresholds for "wall" and "ceiling" to reduce over-detection and improve the overall semantic 3D lifting quality.
3. Increase the mean confidence threshold for "sink" and "mirror" to 0.4 to improve the reliability of their detections, as they currently have low mean confidence (0.345 and 0.361, respectively).
