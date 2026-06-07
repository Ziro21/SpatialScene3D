
# Final Evaluation Summary

> **Note:** this file is auto-generated from the numeric metrics and covers visual reconstruction, Gaussian counts, 3D semantic coverage, 2D mask quality, and storage. For the parts not captured here — open-vocabulary CLIP search, artefact packaging/accessibility, and the agentic QA & release gate — see [`EVALUATION_ADDENDUM.md`](EVALUATION_ADDENDUM.md).

## Overall result

The final pipeline successfully converted a monocular indoor video into a dense 3D Gaussian Splatting reconstruction with semantic labels and quantitative evaluation evidence.

## Visual reconstruction

The held-out visual evaluation used 15 test frames. The model achieved:

- PSNR: 28.926003027454193 dB
- SSIM: 0.90260024865468347
- LPIPS: 0.058430341879526775

These results show good reconstruction quality on unseen viewpoints.

## Gaussian reconstruction

The raw Gaussian reconstruction contained 484707 Gaussians. The semantic PLY preserved 484707 Gaussians, confirming that semantic labelling did not reduce or corrupt the reconstructed cloud.

The pruned viewer PLY contained 379997 Gaussians, removing 104710 Gaussians, equal to 21.602741449989374%. This reduced viewer complexity and removed low-opacity Gaussians from the viewer-ready output.

## 3D semantic output

The semantic lifting stage labelled 181516 out of 484707 Gaussians, giving 37.448602970454317% labelled 3D coverage. The remaining unlabelled proportion was 62.551397029545683%.

This indicates that the pipeline produced a meaningful semantic 3D representation, although coverage remains incomplete.

## 2D mask quality

The 2D segmentation stage generated 3222 masks across 150 frames, with 150 frames containing masks. The average number of masks per frame was 21.48.

The mean mask confidence was 0.46757833766944656, and the median confidence was 0.43014325201511383. This indicates moderate but usable open-vocabulary segmentation confidence.

## Storage and output size

The raw PLY size was 114.64010906219482 MB. The semantic PLY size was 116.95147609710693 MB. The pruned viewer PLY size was 89.875018119812012 MB.

## Main strengths

- Strong held-out reconstruction quality.
- Dense Gaussian scene representation.
- Consistent 2D mask generation across all frames.
- Meaningful 3D semantic labelling of room structures and objects.
- Pruned viewer PLY provides a cleaner, lighter visualisation output.

## Main limitations

- Semantic coverage is moderate rather than complete.
- Wall labelling remains weak.
- Some small-object masks may be noisy.
- Raw PLY contains low-opacity floaters, although the pruned viewer PLY addresses this.
- Exact runtime per stage was not recorded for this run.
