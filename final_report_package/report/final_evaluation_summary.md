
# Final Evaluation Summary

## Overall result

The final pipeline successfully converted a monocular indoor video into a dense 3D Gaussian Splatting reconstruction with semantic labels and quantitative evaluation evidence.

## Visual reconstruction

The held-out visual evaluation used 15 test frames. The model achieved:

- PSNR: 28.832300788747677 dB
- SSIM: 0.90113477309544876
- LPIPS: 0.058924285819133122

These results show good reconstruction quality on unseen viewpoints.

## Gaussian reconstruction

The raw Gaussian reconstruction contained 484460 Gaussians. The semantic PLY preserved 484460 Gaussians, confirming that semantic labelling did not reduce or corrupt the reconstructed cloud.

The pruned viewer PLY contained 380995 Gaussians, removing 103465 Gaussians, equal to 21.356768360648971%. This reduced viewer complexity and removed low-opacity Gaussians from the viewer-ready output.

## 3D semantic output

The semantic lifting stage labelled 186794 out of 484460 Gaussians, giving 38.557156421582796% labelled 3D coverage. The remaining unlabelled proportion was 61.442843578417204%.

This indicates that the pipeline produced a meaningful semantic 3D representation, although coverage remains incomplete.

## 2D mask quality

The 2D segmentation stage generated 3223 masks across 150 frames, with 150 frames containing masks. The average number of masks per frame was 21.486666666666668.

The mean mask confidence was 0.46726112995612262, and the median confidence was 0.42827460169792175. This indicates moderate but usable open-vocabulary segmentation confidence.

## Storage and output size

The raw PLY size was 114.58169078826904 MB. The semantic PLY size was 116.89188003540039 MB. The pruned viewer PLY size was 90.111056327819824 MB.

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
