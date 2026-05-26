# scene3d

> 3D Scene Reconstruction from Monocular Video — From a phone video to an interactive, semantically-labelled 3D Gaussian Splat.

<!-- TODO: Insert hero GIF here -->
<!-- ![Hero Demo](assets/outputs/hero_demo.gif) -->

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/scene3d.git
cd scene3d
pip install -r requirements.txt

# 2. Preprocess a video
bash run.sh assets/videos/example.mp4 my_scene

# 3. Cloud steps (Colab)
# → See notebooks/colab_pipeline.ipynb for MASt3R-SLAM + gsplat

# 4. Run full pipeline (after Colab outputs are downloaded)
bash run.sh assets/videos/example.mp4 my_scene
```

## Pipeline

```
Phone Video → Frame Extraction → MASt3R-SLAM (camera poses + depth)
           → gsplat (3D Gaussian Splat) → Grounded-SAM-2 (semantics)
           → Interactive viser Viewer (RGB / Depth / Semantic / Text Query)
```

## Results

<!-- TODO: Fill in after evaluation -->
| Scene | PSNR ↑ | SSIM ↑ | Chamfer ↓ | Precision@10 |
|-------|--------|--------|-----------|--------------|
| Scene 1 | — | — | — | — |
| Scene 2 | — | — | — | — |

## Design Choices

See [DESIGN.md](DESIGN.md) for a detailed technical discussion of architecture decisions, trade-offs, and robotics relevance.

## References

- Murai et al., *MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors*, CVPR 2025
- Wang et al., *VGGT: Visual Geometry Grounded Transformer*, CVPR 2025 Best Paper
- Leroy et al., *DUSt3R / MASt3R*, CVPR 2024
- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023
- Ren et al., *Grounded SAM 2: Ground and Track Anything in Videos*, IDEA-Research 2024
- Yoo et al., *OpenMonoGS-SLAM*, arXiv:2512.08625, December 2025
- Piekenbrinck et al., *OpenSplat3D*, CVPR 2025 Workshop on OpenSUN3D

## License

- **Code**: Apache 2.0
- **MASt3R-SLAM weights**: Apache 2.0
- **Grounded-SAM-2 weights**: Apache 2.0

## Acknowledgements

Built for the Humanoid Internship Challenge — Perception & Spatial AI, London 2025.
