"""
app.py — Interactive 3D Semantic Viewer (viser)
================================================

Multi-modal viewer for semantically-labelled 3D Gaussian Splats.

Render modes (switchable via sidebar tabs):
  1. RGB        — Original Gaussian splat colours
  2. Depth      — Distance from camera, viridis colourmap
  3. Semantic   — Gaussians coloured by semantic class label
  4. Text Query — CLIP cosine-similarity heatmap (type any text)

Usage:
  python -m viewer.app \\
      --scene scene1 \\
      --data_dir data/ \\
      --output_dir outputs/

  Then open http://localhost:8080 in your browser.

Requirements:
  pip install viser plyfile numpy
  (CLIP needed only for text queries: pip install git+https://github.com/openai/CLIP.git)
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import viser


# ============================================================
# Data Loading
# ============================================================
def load_semantic_ply(ply_path: str) -> dict:
    """
    Load a Gaussian splat PLY, including semantic properties if present.

    Returns a dict with keys:
      - xyz: (N, 3) positions
      - rgb: (N, 3) uint8 colours
      - opacity: (N,) float or None
      - scales: (N, 3) log-scales or None
      - rotations: (N, 4) quaternions or None
      - semantic_label: (N,) uint16 or None
      - semantic_rgb: (N, 3) uint8 or None
      - covariances: (N, 3, 3) or None
    """
    from plyfile import PlyData

    print(f"  Loading PLY: {ply_path}")
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    names = set(v.data.dtype.names)
    n = len(v["x"])

    data = {}

    # Positions
    data["xyz"] = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)

    # Colours
    if "red" in names:
        data["rgb"] = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.uint8)
    else:
        data["rgb"] = np.full((n, 3), 180, dtype=np.uint8)

    # Opacity
    if "opacity" in names:
        data["opacity"] = np.array(v["opacity"], dtype=np.float32)
    else:
        data["opacity"] = np.ones(n, dtype=np.float32)

    # Gaussian scales and rotations (from gsplat output)
    if "scale_0" in names:
        data["scales"] = np.column_stack([v["scale_0"], v["scale_1"], v["scale_2"]]).astype(
            np.float32
        )
    else:
        data["scales"] = None

    if "rot_0" in names:
        data["rotations"] = np.column_stack(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]]
        ).astype(np.float32)
    else:
        data["rotations"] = None

    # Semantic properties (added by lift_to_3d.py)
    if "semantic_label" in names:
        data["semantic_label"] = np.array(v["semantic_label"], dtype=np.uint16)
        data["semantic_rgb"] = np.column_stack(
            [v["semantic_r"], v["semantic_g"], v["semantic_b"]]
        ).astype(np.uint8)
    else:
        data["semantic_label"] = None
        data["semantic_rgb"] = None

    # Compute covariances from scales + rotations if available
    data["covariances"] = _compute_covariances(data["scales"], data["rotations"], n)

    print(f"  {n} Gaussians loaded")
    if data["semantic_label"] is not None:
        n_labelled = np.sum(data["semantic_label"] > 0)
        print(f"  {n_labelled}/{n} have semantic labels")

    return data


def _compute_covariances(
    scales: Optional[np.ndarray],
    rotations: Optional[np.ndarray],
    n: int,
) -> np.ndarray:
    """
    Compute 3×3 covariance matrices from Gaussian scales and rotations.

    If scales/rotations are not available, creates isotropic Gaussians
    with a small default scale.

    Covariance = R @ S @ S^T @ R^T  where S = diag(exp(scale))
    """
    if scales is not None and rotations is not None:
        covariances = np.zeros((n, 3, 3), dtype=np.float32)
        for i in range(n):
            # Scale matrix
            s = np.exp(scales[i])
            S = np.diag(s)

            # Rotation matrix from quaternion (w, x, y, z)
            q = rotations[i]
            qw, qx, qy, qz = q[0], q[1], q[2], q[3]
            R = np.array(
                [
                    [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                    [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
                    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
                ],
                dtype=np.float32,
            )

            RS = R @ S
            covariances[i] = RS @ RS.T
        return covariances
    else:
        # Default: small isotropic Gaussians
        default_scale = 0.005
        cov = np.eye(3, dtype=np.float32) * (default_scale**2)
        return np.tile(cov, (n, 1, 1))


def load_label_mapping(ply_path: str) -> Dict[int, str]:
    """Load the sidecar label mapping JSON."""
    json_path = ply_path.replace(".ply", "_labels.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        return {int(k): v for k, v in data.get("labels", {}).items()}
    return {0: "unlabelled"}


def load_clip_embeddings(embeddings_path: str) -> Optional[dict]:
    """Load precomputed CLIP embeddings if available."""
    if not os.path.exists(embeddings_path):
        return None

    data = np.load(embeddings_path, allow_pickle=True)
    embeddings = {}
    label_names = list(data.get("label_names", []))
    label_ids = list(data.get("label_ids", []))

    for i, name in enumerate(label_names):
        safe_key = f"emb_{str(name).replace(' ', '_').replace('.', '')}"
        if safe_key in data:
            embeddings[str(name)] = {
                "vector": data[safe_key].astype(np.float32),
                "label_id": int(label_ids[i]) if i < len(label_ids) else i + 1,
            }

    if embeddings:
        dim = len(next(iter(embeddings.values()))["vector"])
        print(f"  Loaded {len(embeddings)} CLIP embeddings (dim={dim})")
    return embeddings if embeddings else None


# ============================================================
# Colour Modes
# ============================================================
def compute_depth_colours(xyz: np.ndarray, camera_pos: np.ndarray = None) -> np.ndarray:
    """
    Colour Gaussians by distance from a reference point (viridis colourmap).

    Args:
        xyz: (N, 3) Gaussian positions
        camera_pos: (3,) reference point (defaults to centroid)

    Returns:
        (N, 3) uint8 viridis colours
    """
    if camera_pos is None:
        camera_pos = np.mean(xyz, axis=0)

    depths = np.linalg.norm(xyz - camera_pos, axis=1)

    # Normalise to [0, 1] using percentiles (robust to outliers)
    d_min = np.percentile(depths, 2)
    d_max = np.percentile(depths, 98)
    d_norm = np.clip((depths - d_min) / (d_max - d_min + 1e-8), 0, 1)

    # Viridis colourmap (approximation)
    colours = _viridis_colormap(d_norm)
    return colours


def _viridis_colormap(values: np.ndarray) -> np.ndarray:
    """Apply viridis colourmap to normalised [0, 1] values → (N, 3) uint8."""
    # Simplified viridis: 5 anchor points
    anchors = np.array(
        [
            [68, 1, 84],  # 0.0 — dark purple
            [59, 82, 139],  # 0.25 — blue
            [33, 145, 140],  # 0.5 — teal
            [94, 201, 98],  # 0.75 — green
            [253, 231, 37],  # 1.0 — yellow
        ],
        dtype=np.float32,
    )

    idx = values * (len(anchors) - 1)
    idx_floor = np.clip(np.floor(idx).astype(int), 0, len(anchors) - 2)
    frac = (idx - idx_floor).reshape(-1, 1)

    colours = anchors[idx_floor] * (1 - frac) + anchors[idx_floor + 1] * frac
    return np.clip(colours, 0, 255).astype(np.uint8)


def compute_normal_colours(xyz: np.ndarray, k: int = 8) -> np.ndarray:
    """
    Estimate surface normals from local point neighborhoods → RGB.

    Uses PCA on k-nearest neighbours for each Gaussian.
    Normal (nx, ny, nz) mapped to colour: (|nx|, |ny|, |nz|) × 255.

    For large point clouds, we subsample neighbours for speed.
    """
    n = len(xyz)
    normals = np.zeros((n, 3), dtype=np.float32)

    # Build a simple spatial index by sorting along x
    # For large clouds, use a KD-tree subset approach
    from scipy.spatial import cKDTree

    tree = cKDTree(xyz)

    # Process in chunks for memory efficiency
    chunk_size = 10000
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        batch = xyz[start:end]

        _, idx = tree.query(batch, k=min(k, n))

        for i in range(end - start):
            neighbors = xyz[idx[i]]
            centered = neighbors - neighbors.mean(axis=0)
            if len(centered) < 3:
                normals[start + i] = [0, 0, 1]
                continue

            # PCA: smallest eigenvector = normal
            cov = centered.T @ centered
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                normals[start + i] = eigenvectors[:, 0]  # smallest eigenvalue
            except np.linalg.LinAlgError:
                normals[start + i] = [0, 0, 1]

    # Map normals to RGB: absolute value × 255
    normal_colours = (np.abs(normals) * 255).clip(0, 255).astype(np.uint8)
    return normal_colours


def compute_query_colours(
    query_text: str,
    semantic_labels: np.ndarray,
    label_names: Dict[int, str],
    clip_embeddings: Optional[dict],
    device: str = "cpu",
) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
    """
    Colour Gaussians by CLIP cosine similarity to a text query.

    Steps:
    1. Encode the text query with CLIP
    2. Compute cosine similarity against each instance embedding
    3. Map similarity → red (high) / blue (low) colourmap
    4. Colour each Gaussian by its instance's similarity score

    Returns:
        (N, 3) uint8 heatmap colours, and list of (label, score) matches
    """
    n = len(semantic_labels)
    colours = np.full((n, 3), 50, dtype=np.uint8)  # dark grey default

    if clip_embeddings is None:
        return colours, []

    try:
        import clip as clip_module
        import torch

        # Load CLIP and encode the query text
        try:
            model, _, _ = clip_module.load("ViT-L/14", device=device)
        except RuntimeError:
            model, _, _ = clip_module.load("ViT-B/32", device=device)

        with torch.no_grad():
            text_tokens = clip_module.tokenize([query_text]).to(device)
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_vec = text_features.cpu().numpy().flatten()

    except (ImportError, Exception) as e:
        print(f"  CLIP not available: {e}")
        return colours, []

    # Compute similarity for each label
    label_similarities = {}
    for label_str, emb_data in clip_embeddings.items():
        emb_vec = emb_data["vector"]
        sim = float(np.dot(text_vec, emb_vec))
        label_id = emb_data["label_id"]
        label_similarities[label_id] = sim

    # Build ranked results
    results = [
        (label_str, label_similarities.get(emb_data["label_id"], 0.0))
        for label_str, emb_data in clip_embeddings.items()
    ]
    results.sort(key=lambda x: x[1], reverse=True)

    if not label_similarities:
        return colours, results

    # Normalise similarities to [0, 1]
    sims = np.array(list(label_similarities.values()))
    s_min, s_max = sims.min(), sims.max()
    if s_max - s_min < 1e-6:
        s_max = s_min + 1

    # Colour each Gaussian by its label's similarity
    for label_id, sim in label_similarities.items():
        mask = semantic_labels == label_id
        norm_sim = (sim - s_min) / (s_max - s_min)

        # Heatmap: blue (cold, low sim) → red (hot, high sim)
        r = int(np.clip(norm_sim * 255, 0, 255))
        b = int(np.clip((1 - norm_sim) * 255, 0, 255))
        g = int(np.clip(norm_sim * 100, 0, 100))

        colours[mask] = [r, g, b]

    return colours, results


# ============================================================
# Viewer Application
# ============================================================
class SceneViewer:
    """
    Interactive 3D semantic viewer using viser.

    Provides multiple render modes switchable via sidebar controls,
    and a text input for CLIP-based open-vocabulary queries.
    """

    def __init__(
        self,
        splat_data: dict,
        label_names: Dict[int, str],
        clip_embeddings: Optional[dict] = None,
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        self.data = splat_data
        self.label_names = label_names
        self.clip_embeddings = clip_embeddings
        self.current_mode = "rgb"

        # Subsample for performance if needed
        n = len(splat_data["xyz"])
        if n > 500_000:
            print(f"  Subsampling {n} → 500,000 Gaussians for viewer performance")
            indices = np.random.choice(n, 500_000, replace=False)
            for key in ["xyz", "rgb", "opacity", "semantic_label", "semantic_rgb"]:
                if splat_data.get(key) is not None:
                    splat_data[key] = splat_data[key][indices]
            if splat_data.get("covariances") is not None:
                splat_data["covariances"] = splat_data["covariances"][indices]

        # Precompute colour modes
        print("  Precomputing depth colours...")
        self.depth_colours = compute_depth_colours(splat_data["xyz"])

        print("  Precomputing normal colours (this may take a moment)...")
        try:
            self.normal_colours = compute_normal_colours(splat_data["xyz"])
        except ImportError:
            print("  scipy not available — normals disabled")
            self.normal_colours = splat_data["rgb"].copy()

        # Start server
        self.server = viser.ViserServer(host=host, port=port)
        self._scene_handle = None
        self._setup_gui()
        self._update_scene()

        print(f"\n  ✓ Viewer running at http://localhost:{port}")
        print(f"    Open this URL in your browser!")

    def _setup_gui(self) -> None:
        """Create the sidebar GUI controls."""
        gui = self.server

        gui.scene.set_up_direction("+y")

        # Title
        gui.gui.add_markdown("## 🔬 scene3d Viewer")
        gui.gui.add_markdown("---")

        # Render mode dropdown
        self.mode_dropdown = gui.gui.add_dropdown(
            label="Render Mode",
            options=["RGB", "Depth", "Normals", "Semantic", "Text Query"],
            initial_value="RGB",
        )

        @self.mode_dropdown.on_update
        def _on_mode_change(event: viser.GuiEvent) -> None:
            mode = self.mode_dropdown.value.lower().replace(" ", "_")
            self.current_mode = mode
            self._update_scene()

        gui.gui.add_markdown("---")

        # Point size control
        self.point_size = gui.gui.add_slider(
            label="Point Size",
            min=0.001,
            max=0.05,
            step=0.001,
            initial_value=0.008,
        )

        @self.point_size.on_update
        def _on_size_change(event: viser.GuiEvent) -> None:
            self._update_scene()

        gui.gui.add_markdown("---")

        # Text query input (for CLIP mode)
        gui.gui.add_markdown("### 🔍 Text Query")
        self.query_input = gui.gui.add_text(
            label="Query",
            initial_value="chair",
        )
        self.query_button = gui.gui.add_button("Search")
        self.query_results = gui.gui.add_markdown("*Enter a query and click Search*")

        @self.query_button.on_click
        def _on_query(event: viser.GuiEvent) -> None:
            query = self.query_input.value.strip()
            if query:
                self.current_mode = "text_query"
                self.mode_dropdown.value = "Text Query"
                self._update_scene(query_text=query)

        gui.gui.add_markdown("---")

        # Scene info
        n = len(self.data["xyz"])
        info_lines = [f"**Gaussians:** {n:,}"]
        if self.data.get("semantic_label") is not None:
            n_labelled = int(np.sum(self.data["semantic_label"] > 0))
            info_lines.append(f"**Labelled:** {n_labelled:,} ({100*n_labelled/n:.0f}%)")

            # List labels
            unique_labels = np.unique(self.data["semantic_label"])
            for lid in sorted(unique_labels):
                if lid == 0:
                    continue
                name = self.label_names.get(int(lid), f"class_{lid}")
                count = int(np.sum(self.data["semantic_label"] == lid))
                info_lines.append(f"  • {name}: {count:,}")

        if self.clip_embeddings:
            info_lines.append(f"**CLIP instances:** {len(self.clip_embeddings)}")

        gui.gui.add_markdown("\n".join(info_lines))

    def _update_scene(self, query_text: str = "") -> None:
        """Redraw the point cloud with the current colour mode."""
        xyz = self.data["xyz"]
        mode = self.current_mode

        # Choose colours based on mode
        if mode == "rgb":
            colours = self.data["rgb"]
        elif mode == "depth":
            colours = self.depth_colours
        elif mode == "normals":
            colours = self.normal_colours
        elif mode == "semantic":
            if self.data.get("semantic_rgb") is not None:
                colours = self.data["semantic_rgb"]
            else:
                colours = self.data["rgb"]
        elif mode == "text_query":
            if query_text:
                device = "mps" if _has_mps() else "cpu"
                colours, results = compute_query_colours(
                    query_text,
                    self.data.get("semantic_label", np.zeros(len(xyz), dtype=np.uint16)),
                    self.label_names,
                    self.clip_embeddings,
                    device=device,
                )
                # Update results display
                if results:
                    lines = [f'**Query: "{query_text}"**\n']
                    for label, score in results[:5]:
                        bar = "🟥" if score > 0.25 else "🟧" if score > 0.2 else "🟦"
                        lines.append(f"{bar} {label}: {score:.3f}")
                    self.query_results.content = "\n".join(lines)
                else:
                    self.query_results.content = "*No CLIP embeddings available*"
            else:
                colours = self.data["rgb"]
        else:
            colours = self.data["rgb"]

        # Remove existing scene node
        if self._scene_handle is not None:
            self._scene_handle.remove()

        # Render as point cloud
        self._scene_handle = self.server.scene.add_point_cloud(
            name="/splat",
            points=xyz.astype(np.float32),
            colors=colours.astype(np.uint8),
            point_size=self.point_size.value,
            point_shape="circle",
        )

    def run(self) -> None:
        """Block forever, serving the viewer."""
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n  Shutting down viewer...")
            self.server.stop()


def _has_mps() -> bool:
    """Check if MPS (Apple Silicon GPU) is available."""
    try:
        import torch

        return torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        return False


# ============================================================
# Occupancy Map (Bonus)
# ============================================================
def generate_occupancy_map(
    xyz: np.ndarray,
    semantic_labels: np.ndarray,
    label_names: Dict[int, str],
    resolution: float = 0.05,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Generate a top-down 2D occupancy map from Gaussian centres.

    Projects all Gaussians onto the XZ plane (assuming Y is up),
    creating a 2D grid coloured by the dominant semantic class in each cell.

    Args:
        xyz: (N, 3) Gaussian positions
        semantic_labels: (N,) label per Gaussian
        label_names: mapping label_id → name
        resolution: metres per grid cell
        output_path: optional path to save as PNG

    Returns:
        (H, W, 3) uint8 occupancy map image
    """
    import cv2

    # Use X and Z axes (assuming Y is up)
    x, z = xyz[:, 0], xyz[:, 2]

    x_min, x_max = np.percentile(x, 1), np.percentile(x, 99)
    z_min, z_max = np.percentile(z, 1), np.percentile(z, 99)

    grid_w = int((x_max - x_min) / resolution) + 1
    grid_h = int((z_max - z_min) / resolution) + 1
    grid_w = min(grid_w, 2000)
    grid_h = min(grid_h, 2000)

    # Initialise grid
    occupancy = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    # Generate colours per label
    from semantics.lift_to_3d import _generate_label_colours

    label_colours = _generate_label_colours(label_names)

    # Rasterise Gaussians into grid cells
    cell_votes = {}
    for i in range(len(xyz)):
        cx = int((x[i] - x_min) / resolution)
        cy = int((z[i] - z_min) / resolution)
        if 0 <= cx < grid_w and 0 <= cy < grid_h:
            if (cy, cx) not in cell_votes:
                cell_votes[(cy, cx)] = []
            cell_votes[(cy, cx)].append(int(semantic_labels[i]))

    # Assign colour by majority label in each cell
    from collections import Counter

    for (cy, cx), votes in cell_votes.items():
        if votes:
            label = Counter(votes).most_common(1)[0][0]
            colour = label_colours.get(label, (128, 128, 128))
            occupancy[cy, cx] = colour

    # Scale up for visibility
    scale = max(1, 800 // max(grid_w, grid_h))
    if scale > 1:
        occupancy = cv2.resize(
            occupancy, (grid_w * scale, grid_h * scale), interpolation=cv2.INTER_NEAREST
        )

    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(occupancy, cv2.COLOR_RGB2BGR))
        print(f"  ✓ Occupancy map saved: {output_path}")

    return occupancy


# ============================================================
# CLI
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive 3D semantic viewer for Gaussian splats"
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="scene1",
        help="Scene name (looks for data/{scene}/ and outputs/{scene}/)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Base outputs directory",
    )
    parser.add_argument(
        "--splat",
        type=str,
        default=None,
        help="Direct path to .ply file (overrides --scene)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the viewer server",
    )
    parser.add_argument(
        "--occupancy",
        action="store_true",
        help="Generate top-down occupancy map",
    )

    args = parser.parse_args()

    # Resolve file paths
    if args.splat:
        ply_path = args.splat
    else:
        # Try semantic PLY first, fall back to raw PLY
        semantic_ply = os.path.join(args.output_dir, args.scene, "splat_semantic.ply")
        raw_ply = os.path.join(args.output_dir, args.scene, "splat.ply")
        ply_path = semantic_ply if os.path.exists(semantic_ply) else raw_ply

    masks_dir = os.path.join(args.data_dir, args.scene, "masks")
    embeddings_path = os.path.join(args.output_dir, args.scene, "embeddings.npz")

    print("\n═══ scene3d Interactive Viewer ═══\n")
    print(f"  Scene: {args.scene}")
    print(f"  PLY:   {ply_path}")
    print(f"  Masks: {masks_dir}")
    print(f"  CLIP:  {embeddings_path}")

    # Load data
    if not os.path.exists(ply_path):
        print(f"\n  ⚠ PLY file not found: {ply_path}")
        print(f"  Run the Colab pipeline first, then:")
        print(
            f"    python -m semantics.lift_to_3d --splat ... --masks ... --colmap ... --output ..."
        )
        return

    splat_data = load_semantic_ply(ply_path)
    label_names = load_label_mapping(ply_path)

    clip_embs = None
    if os.path.exists(embeddings_path):
        clip_embs = load_clip_embeddings(embeddings_path)
    else:
        print(f"  ℹ No CLIP embeddings found — text queries will be disabled")

    # Generate occupancy map if requested
    if args.occupancy and splat_data.get("semantic_label") is not None:
        occ_path = os.path.join(args.output_dir, args.scene, "occupancy_map.png")
        generate_occupancy_map(
            splat_data["xyz"],
            splat_data["semantic_label"],
            label_names,
            output_path=occ_path,
        )

    # Launch viewer
    viewer = SceneViewer(
        splat_data=splat_data,
        label_names=label_names,
        clip_embeddings=clip_embs,
        port=args.port,
    )
    viewer.run()


if __name__ == "__main__":
    main()
