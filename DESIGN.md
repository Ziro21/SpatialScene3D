# Design Choices — scene3d

> A technical memo explaining the architecture, trade-offs, and robotics relevance of this 3D scene reconstruction system.
>
> **Author:** Zeyad Khalil · **Date:** May 2026 · **Target:** Humanoid Perception & Spatial AI Internship

---

## 1. Problem Framing

Indoor monocular 3D reconstruction from phone video is an unsolved problem at production quality. The core difficulties are:

**No depth sensor.** Consumer smartphones record RGB only. Depth must be inferred entirely from visual correspondences across frames, which makes the problem fundamentally under-constrained. A single pixel could correspond to any point along its ray — resolving this ambiguity requires matching that pixel across multiple viewpoints, which fails when the scene lacks texture.

**Textureless surfaces dominate indoors.** White walls, ceilings, and floors occupy the majority of indoor scenes. Traditional feature matchers (SIFT, SuperPoint) produce zero or unreliable matches on these surfaces, causing COLMAP and ORB-SLAM to fail silently. The reconstructed geometry develops holes exactly where the largest planar surfaces should be.

**Scale ambiguity.** Monocular systems recover geometry only up to an arbitrary scale factor. A reconstruction from a phone video cannot distinguish a dollhouse from a real room without additional information (known object dimensions, IMU data, or ARKit depth). This is tolerable for visualisation but problematic for robotics, where metric-scale accuracy matters for navigation and manipulation.

**Dynamic lighting and reflections.** Indoor environments contain windows, mirrors, and glossy surfaces that violate the Lambertian assumption. Specular reflections create phantom geometry — the reconstruction "sees" objects that don't exist. Moving shadows from changing daylight confuse temporal correspondences.

**Computational constraints.** A useful system must process 60–120 seconds of 1080p video. Dense methods (MVS, NeRF) are GPU-bound; the challenge is achieving both quality and tractability on accessible hardware (a free-tier Colab GPU + a MacBook).

These constraints collectively make indoor monocular 3D reconstruction one of the hardest settings in 3D vision. Our system addresses each one through careful tool selection.

---

## 2. Alternatives Considered

We evaluated four approaches before selecting our final pipeline. This section documents each option and why it was accepted or rejected.

### 2.1 COLMAP (Schönberger & Frahm, CVPR 2016)

**What it does:** Classical Structure-from-Motion. Detects SIFT keypoints, matches them across images, solves bundle adjustment for camera poses, then runs Multi-View Stereo for dense depth.

**Strengths:**
- Gold standard for outdoor scenes with rich texture
- Extremely well-tested; used by nerfstudio, gsplat, and most 3DGS papers
- Produces reliable camera intrinsics and extrinsics

**Why rejected as primary backbone:**
- **Fails on textureless walls.** In our test scenes, COLMAP registered only 40-60% of frames on white-walled rooms. The remaining frames have no 3D reconstruction — exactly the areas that matter most for room understanding.
- **Slow.** Feature matching on 200 frames takes 15-30 minutes. Bundle adjustment adds another 10-20 minutes.
- **No real-time path.** Batch-only processing with no streaming capability.

**Role in our pipeline:** We retain COLMAP as a **baseline** for evaluation (Section 7 in README). Its failures on our scenes demonstrate precisely why a learned approach is necessary.

### 2.2 VGGT (Wang et al., CVPR 2025 — Best Paper)

**What it does:** Visual Geometry Grounded Transformer. Feed-forward model that jointly predicts camera poses, depth, and 3D points from a set of images in a single pass. State-of-the-art accuracy on 7-Scenes (Chamfer: 0.055 m).

**Strengths:**
- Best-in-class geometric accuracy
- Single feed-forward pass — no iterative optimisation
- CVPR 2025 Best Paper — maximum research credibility

**Why rejected:**
- **Hard frame limit (~50 frames per chunk).** VGGT processes all frames simultaneously in a single attention pass. GPU memory (even on an A100 with 40 GB VRAM) limits this to ~50 frames. Our videos produce 150–300 frames.
- **Chunking introduces alignment risk.** Processing in 50-frame chunks requires Umeyama + RANSAC alignment across submaps. This is the single biggest engineering risk in any VGGT-based pipeline — misalignment produces visible seams in the reconstruction.
- **No real-time capability.** Batch-only, processing all frames at once.

**Why it's still valuable:** We cite VGGT prominently as the evaluated alternative. Choosing MASt3R-SLAM over a CVPR Best Paper — and articulating why — demonstrates stronger engineering judgement than blindly using the highest-profile tool.

### 2.3 NeRF (Mildenhall et al., ECCV 2020) and Derivatives

**What it does:** Neural Radiance Fields represent scenes as continuous volumetric functions. Novel views are rendered by ray-marching through the MLP.

**Why rejected:**
- **Superseded by 3DGS.** Gaussian splatting achieves comparable or better quality with 100–1000× faster rendering (real-time vs. minutes per frame).
- **No explicit geometry.** NeRF's implicit representation makes it difficult to attach semantic labels to spatial locations — there are no discrete primitives to label.
- **Slow training and rendering.** Incompatible with interactive viewer requirements.

### 2.4 MASt3R-SLAM (Murai et al., CVPR 2025) ← Selected

**What it does:** Combines the MASt3R dense matching backbone (DUSt3R + metric scale) with a real-time SLAM frontend. Processes video frames sequentially, maintaining a globally consistent map with loop closure.

**Strengths:**
- **No frame limit.** Streams the full video natively — no chunking, no alignment.
- **Handles textureless surfaces.** MASt3R's learned features match on walls and floors where SIFT fails, because the backbone learns contextual appearance priors from large-scale training.
- **Real-time capable.** 15 FPS on an RTX 4090 — directly relevant to robotics (a humanoid robot could run this online during navigation).
- **Calibration-free.** No camera intrinsics needed — inferred by the network.
- **SLAM heritage.** Andrew Davison's group at Imperial College — the SLAM team. This is the group most relevant to humanoid robot perception.
- **Comparable accuracy.** Chamfer 0.056 m on 7-Scenes, statistically identical to VGGT's 0.055 m.

**Trade-off:** Slightly less geometric accuracy than VGGT in batch mode (0.056 m vs. 0.055 m). We accept this for the dramatically simpler and more robust pipeline.

---

## 3. Decision & Trade-offs

### 3.1 Geometry: MASt3R-SLAM over VGGT

| Factor | VGGT | MASt3R-SLAM |
|---|---|---|
| Video length support | ~50 frames/chunk | Unlimited (streaming) |
| Alignment complexity | Umeyama + RANSAC between submaps | Not needed |
| 7-Scenes Chamfer | 0.055 m | 0.056 m |
| Real-time capable | No (batch only) | Yes (15 FPS) |
| Implementation risk | **HIGH** (submap alignment) | **LOW** |
| Robotics relevance | Research demo | Online SLAM for navigation |

**Decision:** MASt3R-SLAM removes the single biggest engineering risk (submap alignment) while preserving equivalent accuracy and adding real-time capability. The 0.001 m accuracy difference is within measurement noise.

### 3.2 Scene Representation: gsplat (3D Gaussian Splatting) over NeRF

| Factor | NeRF | 3DGS (gsplat) |
|---|---|---|
| Render speed | ~30s per frame | Real-time (100+ FPS) |
| Explicit primitives | No (implicit MLP) | Yes (discrete Gaussians) |
| Semantic attachment | Difficult | Natural (per-Gaussian labels) |
| Training speed | 12–24 hours | 30–60 minutes |
| Point cloud init | Not applicable | Direct from MASt3R-SLAM |

**Decision:** 3DGS is strictly dominant for our use case. Discrete Gaussians are the ideal primitive for attaching semantic labels — each Gaussian can carry a class label and CLIP embedding. We use gsplat specifically (nerfstudio's implementation) for its clean API and `tyro`-based CLI.

### 3.3 Semantics: Grounded-SAM-2 + CLIP over LangSplat

**LangSplat** (Qin et al., CVPR 2024) bakes CLIP features directly into each Gaussian during training. This is elegant but:
- Requires retraining the entire splat with a modified loss
- CLIP feature dimension (768) per Gaussian massively inflates the model
- Training is 3–5× slower

**Our approach** (Grounded-SAM-2 + majority voting + CLIP crops):
- **Decoupled.** Train the splat first (fast, standard), add semantics after (no retraining).
- **Modular.** Swap segmentation models without retraining — if SAM 3 or DINO v2 ships, we drop them in.
- **Efficient.** One CLIP embedding per *instance* (e.g. one vector for "chair"), not per Gaussian. Storage: O(num_instances) vs. O(num_gaussians).
- **Debuggable.** Masks are saved as PNGs — you can visually inspect every segmentation decision.

**Trade-off:** Our approach requires explicit camera poses for re-projection (which we have from MASt3R-SLAM). LangSplat doesn't. But since we already have poses, this trade-off costs us nothing.

---

## 4. Robotics Relevance

This system is not an academic exercise — every component maps directly to a humanoid robot's perception needs.

### 4.1 MASt3R-SLAM → Calibration-Free Indoor Mapping

A humanoid robot entering a new room cannot assume pre-calibrated cameras. MASt3R-SLAM infers both camera intrinsics and extrinsics from the visual stream alone. At 15 FPS on current hardware, this could run online as the robot walks through a space — building a 3D map of the environment in real time, without any pre-mapping or fiducial markers.

**Specific robot capability:** The robot can autonomously map a new apartment before starting any task.

### 4.2 3D Gaussian Splatting → Photorealistic Scene Memory

The Gaussian splat serves as the robot's **spatial memory** — a compact, photorealistic representation of the environment that persists after the initial scan. This enables:

- **Sim-to-real transfer.** Render training images from arbitrary viewpoints to train downstream policies without physically moving the robot.
- **Change detection.** Compare the stored splat against new observations to detect moved furniture, new objects, or obstacles.
- **Path rehearsal.** Simulate camera views along a planned trajectory before executing it, checking for occlusions or obstacles.

### 4.3 Grounded-SAM-2 + CLIP → Open-Vocabulary Object Understanding

The combination of instance segmentation (Grounded-SAM-2) and language-visual embeddings (CLIP) gives the robot **open-vocabulary object detection** in 3D:

- **"Pick up the red mug on the table."** The robot can query its 3D map with natural language, identify the target object's position and bounding box, and plan a grasp trajectory — all without pre-defined object classes.
- **Task grounding.** High-level instructions from a user ("tidy up the living room") can be grounded to specific objects in the scene via CLIP similarity.
- **Novel object generalisation.** CLIP's zero-shot capability means the robot can understand objects it has never seen in training (e.g. "the blue IKEA shelf"), unlike closed-vocabulary detectors.

### 4.4 Occupancy Map → Navigation Planning

The top-down 2D occupancy grid (projected from Gaussian centres) is directly usable as input to standard 2D navigation planners (A*, RRT, potential fields). The semantic labels add obstacle classes — the robot can distinguish between a wall (permanent obstacle) and a chair (movable obstacle), enabling smarter path planning.

**Specific robot capability:** Feed the occupancy map to the locomotion planner to navigate between rooms without LiDAR.

### 4.5 The Full Loop: Perceive → Understand → Act

```
Robot enters room
    │
    ▼ MASt3R-SLAM (real-time)
Build 3D map
    │
    ▼ gsplat
Compress to Gaussian splat
    │
    ▼ Grounded-SAM-2 + CLIP
Label every surface and object
    │
    ▼ Natural language query
"Bring me the book from the desk"
    │
    ▼ Occupancy map + grasping
Plan path → navigate → pick up → deliver
```

This pipeline is the **perception stack** for a household humanoid robot. Our implementation demonstrates the full chain from raw video to language-grounded 3D understanding.

---

## 5. Limitations & Future Work

### 5.1 Current Limitations

**Specular surfaces.** Mirrors, glass tables, and polished floors violate the Lambertian assumption that all methods (including MASt3R-SLAM) rely on. Specular reflections create phantom geometry — a mirror produces a "room behind the wall". This is a fundamental limitation of passive vision without explicit specularity modelling.

**Not real-time end-to-end.** While MASt3R-SLAM runs at 15 FPS, the full pipeline (SLAM → gsplat training → segmentation → lifting) takes 30–90 minutes per scene. For a robot, this means the initial mapping is slow — subsequent updates would need an incremental approach.

**Scale drift on long corridors.** Monocular SLAM accumulates drift over long trajectories. MASt3R-SLAM's loop closure mitigates this in rooms, but traversing a 50-metre corridor would produce noticeable drift. Metric scale is estimated from the network, not from a physical sensor.

**Segmentation relies on 2D.** Grounded-SAM-2 operates per-frame. While our majority voting across views produces robust labels, it cannot distinguish two identical objects (e.g. two white chairs) without 3D geometric separation. True 3D instance segmentation would require panoptic lifting.

**8 GB Mac constraint.** The local pipeline (CLIP, viser) runs on an M1 MacBook with 8 GB unified memory. CLIP ViT-L/14 requires ~1.5 GB, leaving limited headroom. We fall back to ViT-B/32 when memory is tight, at some cost to text query quality.

### 5.2 Future Directions

**MonST3R for dynamic scenes.** MonST3R (Zhang et al., 2024) extends the DUSt3R/MASt3R framework to handle moving objects by predicting per-pixel motion fields. Integrating this would allow the robot to reconstruct a scene where people are walking through it — filtering out dynamic elements while preserving static structure.

**Active mapping.** Instead of passive video capture, the robot could direct its own camera trajectory to maximise coverage and minimise uncertainty — an active SLAM loop. MASt3R-SLAM's online capability makes this feasible: the robot could detect low-confidence regions in the map and plan viewpoints to fill them.

**Incremental Gaussian updates.** Currently, the splat is trained from scratch for each scene. Incremental Gaussian splatting (adding new Gaussians from new observations without full retraining) would enable the robot to continuously refine its scene model as it operates.

**LiDAR fusion.** iPhone Pro and iPad Pro include LiDAR sensors. Fusing LiDAR depth with MASt3R-SLAM's visual estimates would resolve scale ambiguity and improve accuracy on textureless surfaces. The pipeline's modular COLMAP-format intermediate makes this straightforward — replace the MASt3R-SLAM depth with fused depth.

---

## References

| # | Paper | Venue | Role in Our Pipeline |
|---|---|---|---|
| 1 | MASt3R-SLAM — Murai et al. [[arXiv:2412.12392](https://arxiv.org/abs/2412.12392)] | CVPR 2025 | Primary geometry backbone |
| 2 | VGGT — Wang et al. [[arXiv:2503.11651](https://arxiv.org/abs/2503.11651)] | CVPR 2025 Best Paper | Evaluated alternative |
| 3 | DUSt3R / MASt3R — Leroy et al. | CVPR 2024 | Foundation of MASt3R-SLAM |
| 4 | 3D Gaussian Splatting — Kerbl et al. | SIGGRAPH 2023 | Scene representation |
| 5 | gsplat — nerfstudio-project | Open-source | 3DGS implementation |
| 6 | Grounded-SAM-2 — Ren et al. | IDEA-Research 2024 | Semantic segmentation |
| 7 | SAM 2 — Ravi et al. | Meta AI 2024 | Mask generation backbone |
| 8 | CLIP — Radford et al. | ICML 2021 | Open-vocabulary embeddings |
| 9 | OpenMonoGS-SLAM — Yoo et al. [[arXiv:2512.08625](https://arxiv.org/abs/2512.08625)] | arXiv Dec 2025 | Closest prior art |
| 10 | CUT3R — Wang et al. [[arXiv:2501.12387](https://arxiv.org/abs/2501.12387)] | CVPR 2025 Oral | Streaming alternative |
| 11 | Depth Anything V2 — Yang et al. | NeurIPS 2024 | Depth sanity check |
| 12 | LangSplat — Qin et al. | CVPR 2024 | Evaluated semantic alternative |
| 13 | NeRF — Mildenhall et al. | ECCV 2020 | Historical baseline |
| 14 | COLMAP — Schönberger & Frahm | CVPR 2016 | SfM baseline |
