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
2. **Stage Verdicts**:
   - Visual reconstruction: PASS, with a held-out PSNR of 28.83 dB, SSIM of 0.901, and LPIPS of 0.059.
   - Semantic 3D lifting: WARNING, due to 38.6% of Gaussians being labelled, indicating incomplete coverage.
   - 2D mask quality: PASS, with consistent coverage and moderate open-vocab confidence.
   - Gaussian PLY quality: WARNING, due to 20.6% low-opacity floaters in the raw PLY, although the pruned viewer PLY removes them.
   - Efficiency / packaging: PASS, with all evaluation artifacts and packages produced.
3. **Dominant Weakness and Root Cause**: The dominant weakness is the incomplete coverage in semantic 3D lifting, with only 38.6% of Gaussians labelled. The root cause is likely the under-segmentation of large flat surfaces.
4. **Recommended Next Action**: To improve the semantic 3D lifting stage, re-run the semantic lifting process with adjusted parameters to target the under-segmentation of large flat surfaces, aiming to increase the labelled fraction beyond 38.6%.
