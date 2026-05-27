"""
lift_to_3d.py — Project 2D semantic masks onto 3D Gaussians
=============================================================

Pipeline:
  1. Load the 3D Gaussian splat (.ply) — each Gaussian has (x, y, z, ...)
  2. Load camera poses from the COLMAP workspace (world-to-camera transforms)
  3. For each frame, project all Gaussians into that camera's image plane
  4. Look up which mask (if any) covers each projected Gaussian
  5. Majority-vote across all frames → each Gaussian gets one label

This runs on your Mac (CPU/MPS). No GPU required — it's just
projection math and nearest-neighbour lookups.

Usage:
  python -m semantics.lift_to_3d \\
      --splat outputs/scene1/splat.ply \\
      --masks data/scene1/masks/ \\
      --colmap data/scene1/colmap/ \\
      --output outputs/scene1/splat_semantic.ply
"""

import argparse
import json
import os
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# PLY I/O — Read and write Gaussian splat files
# ============================================================
def load_splat_ply(ply_path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load a 3D Gaussian Splat .ply file.

    Returns the xyz positions and a dict of all other per-Gaussian
    properties (opacity, scale, rotation, SH coefficients, etc.)
    so we can write them back with the added semantic label.

    Args:
        ply_path: path to the .ply file

    Returns:
        Tuple of:
          - xyz: (N, 3) Gaussian centre positions
          - properties: dict mapping property_name → (N,) or (N, K) arrays
    """
    try:
        from plyfile import PlyData

        ply = PlyData.read(ply_path)
        vertices = ply["vertex"]

        xyz = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float64)

        # Store all other properties for round-tripping
        properties = {}
        for prop in vertices.data.dtype.names:
            if prop not in ("x", "y", "z"):
                properties[prop] = np.array(vertices[prop])

        print(f"  Loaded splat: {len(xyz)} Gaussians")
        print(f"  Properties: {list(properties.keys())[:10]}...")
        return xyz, properties

    except ImportError:
        raise ImportError("plyfile is required: pip install plyfile")


def save_semantic_ply(
    output_path: str,
    xyz: np.ndarray,
    properties: Dict[str, np.ndarray],
    labels: np.ndarray,
    label_names: Dict[int, str],
) -> None:
    """
    Save the Gaussian splat with added semantic label as custom PLY properties.

    Adds two new properties per vertex:
      - semantic_label: integer label ID (uint16)
      - semantic_r/g/b: colour for visualisation (uint8 × 3)

    Args:
        output_path: path to write the output .ply
        xyz: (N, 3) Gaussian positions
        properties: dict of original per-Gaussian properties
        labels: (N,) integer semantic label per Gaussian
        label_names: mapping label_id → class name string
    """
    from plyfile import PlyData, PlyElement

    n = len(xyz)

    # Generate consistent colours per label
    label_colours = _generate_label_colours(label_names)

    # Build the dtype for the structured array
    dtypes = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    for name, arr in properties.items():
        if arr.dtype in (np.float32, np.float64):
            dtypes.append((name, "f4"))
        elif arr.dtype in (np.uint8,):
            dtypes.append((name, "u1"))
        elif arr.dtype in (np.int32, np.int64):
            dtypes.append((name, "i4"))
        else:
            dtypes.append((name, "f4"))

    # Add semantic properties
    dtypes.extend(
        [
            ("semantic_label", "u2"),
            ("semantic_r", "u1"),
            ("semantic_g", "u1"),
            ("semantic_b", "u1"),
        ]
    )

    vertex_data = np.empty(n, dtype=dtypes)
    vertex_data["x"] = xyz[:, 0].astype(np.float32)
    vertex_data["y"] = xyz[:, 1].astype(np.float32)
    vertex_data["z"] = xyz[:, 2].astype(np.float32)

    for name, arr in properties.items():
        vertex_data[name] = arr.astype(vertex_data[name].dtype)

    vertex_data["semantic_label"] = labels.astype(np.uint16)
    for i in range(n):
        r, g, b = label_colours.get(int(labels[i]), (128, 128, 128))
        vertex_data["semantic_r"][i] = r
        vertex_data["semantic_g"][i] = g
        vertex_data["semantic_b"][i] = b

    el = PlyElement.describe(vertex_data, "vertex")
    PlyData([el], text=False).write(output_path)

    # Save the label mapping as a sidecar JSON
    mapping_path = output_path.replace(".ply", "_labels.json")
    mapping = {
        "labels": {str(k): v for k, v in label_names.items()},
        "colours": {str(k): list(v) for k, v in label_colours.items()},
        "num_gaussians": int(n),
        "label_distribution": dict(Counter(labels.tolist())),
    }
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"  ✓ Saved semantic PLY: {output_path}")
    print(f"  ✓ Saved label mapping: {mapping_path}")


def _generate_label_colours(label_names: Dict[int, str]) -> Dict[int, Tuple[int, int, int]]:
    """Generate visually distinct colours for each label using HSV spacing."""
    colours = {}
    # Label 0 = unlabelled → grey
    colours[0] = (128, 128, 128)

    n_labels = max(1, len(label_names) - (1 if 0 in label_names else 0))
    for i, label_id in enumerate(sorted(label_names.keys())):
        if label_id == 0:
            continue
        hue = int(180 * i / n_labels)  # OpenCV hue range is 0-179
        hsv = np.array([[[hue, 220, 230]]], dtype=np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
        colours[label_id] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    return colours


# ============================================================
# Camera Loading — Read poses from COLMAP binary
# ============================================================
def load_colmap_cameras(colmap_dir: str) -> Tuple[np.ndarray, List[Dict]]:
    """
    Load camera intrinsics and extrinsics from a COLMAP workspace.

    Reads cameras.bin and images.bin to get:
      - Intrinsic matrix K (3×3) for each camera
      - World-to-camera transforms (R, t) for each image

    Args:
        colmap_dir: path to COLMAP workspace (containing sparse/0/)

    Returns:
        Tuple of:
          - K: (3, 3) camera intrinsic matrix
          - images: list of dicts with keys 'name', 'R', 't', 'image_id'
    """
    sparse_dir = os.path.join(colmap_dir, "sparse", "0")

    # --- Read cameras.bin ---
    cameras_path = os.path.join(sparse_dir, "cameras.bin")
    with open(cameras_path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        camera_id = struct.unpack("<i", f.read(4))[0]
        model_id = struct.unpack("<i", f.read(4))[0]
        width = struct.unpack("<Q", f.read(8))[0]
        height = struct.unpack("<Q", f.read(8))[0]

        # PINHOLE model: fx, fy, cx, cy
        if model_id == 1:
            fx = struct.unpack("<d", f.read(8))[0]
            fy = struct.unpack("<d", f.read(8))[0]
            cx = struct.unpack("<d", f.read(8))[0]
            cy = struct.unpack("<d", f.read(8))[0]
        else:
            raise ValueError(f"Unsupported camera model: {model_id}")

    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    # --- Read images.bin ---
    images_path = os.path.join(sparse_dir, "images.bin")
    images = []

    with open(images_path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_images):
            image_id = struct.unpack("<i", f.read(4))[0]
            qw = struct.unpack("<d", f.read(8))[0]
            qx = struct.unpack("<d", f.read(8))[0]
            qy = struct.unpack("<d", f.read(8))[0]
            qz = struct.unpack("<d", f.read(8))[0]
            tx = struct.unpack("<d", f.read(8))[0]
            ty = struct.unpack("<d", f.read(8))[0]
            tz = struct.unpack("<d", f.read(8))[0]
            camera_id_ref = struct.unpack("<i", f.read(4))[0]

            # Read null-terminated image name
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8")

            # Skip 2D keypoints
            num_points = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_points):
                f.read(24)  # x, y (2 doubles) + point3d_id (1 long)

            # Quaternion to rotation matrix
            R = _quat_to_rot(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])

            images.append(
                {
                    "image_id": image_id,
                    "name": name,
                    "R": R,  # world-to-camera rotation
                    "t": t,  # world-to-camera translation
                }
            )

    # Sort by name for consistent ordering
    images.sort(key=lambda x: x["name"])
    print(f"  Loaded {len(images)} camera poses, image size: {width}×{height}")
    return K, images


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Quaternion (w,x,y,z) → 3×3 rotation matrix."""
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )


# ============================================================
# Mask Loading
# ============================================================
def load_masks(masks_dir: str) -> Tuple[List[Dict], Dict[int, str]]:
    """
    Load Grounded-SAM-2 mask manifest and images.

    The masks/ directory contains:
      - masks.json: manifest with per-frame mask info
      - *.png: binary mask images (255 = inside, 0 = outside)

    Args:
        masks_dir: path to masks directory

    Returns:
        Tuple of:
          - frames_masks: list of dicts, one per frame, with mask data
          - label_names: mapping instance_id → class name
    """
    manifest_path = os.path.join(masks_dir, "masks.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Build label name mapping
    # Label 0 = unlabelled (no mask)
    label_names = {0: "unlabelled"}
    label_counter = 1

    # Map (label_string) → label_id for consistent labelling across frames
    label_string_to_id = {}

    for frame_info in manifest:
        for mask_info in frame_info.get("masks", []):
            label_str = mask_info["label"].lower().strip()
            if label_str not in label_string_to_id:
                label_string_to_id[label_str] = label_counter
                label_names[label_counter] = label_str
                label_counter += 1
            mask_info["_label_id"] = label_string_to_id[label_str]

    print(f"  Loaded mask manifest: {len(manifest)} frames")
    print(f"  Label classes: {label_names}")
    return manifest, label_names


# ============================================================
# Core Algorithm: Project + Vote
# ============================================================
def project_gaussians_to_frame(
    xyz: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    img_w: int,
    img_h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D Gaussian centres into a camera's image plane.

    Uses the standard pinhole projection:
      p_camera = R @ p_world + t
      p_pixel  = K @ p_camera (then divide by z)

    Args:
        xyz: (N, 3) 3D Gaussian positions in world frame
        K: (3, 3) camera intrinsic matrix
        R: (3, 3) world-to-camera rotation
        t: (3,) world-to-camera translation
        img_w: image width in pixels
        img_h: image height in pixels

    Returns:
        Tuple of:
          - pixels: (N, 2) projected pixel coordinates (u, v)
          - valid: (N,) boolean mask — True if Gaussian is visible
    """
    # Transform to camera coordinates
    p_cam = (R @ xyz.T).T + t  # (N, 3)

    # Filter: must be in front of camera (z > 0)
    valid = p_cam[:, 2] > 0.01

    # Project to pixel coordinates
    p_proj = (K @ p_cam.T).T  # (N, 3)
    z = p_proj[:, 2:3]
    z = np.where(z > 0.01, z, 0.01)  # avoid division by zero
    pixels = p_proj[:, :2] / z  # (N, 2)

    # Filter: must be within image bounds
    valid &= (pixels[:, 0] >= 0) & (pixels[:, 0] < img_w)
    valid &= (pixels[:, 1] >= 0) & (pixels[:, 1] < img_h)

    return pixels, valid


def assign_labels_for_frame(
    pixels: np.ndarray,
    valid: np.ndarray,
    masks_info: List[Dict],
    masks_dir: str,
) -> np.ndarray:
    """
    For each visible Gaussian, look up which mask (if any) covers it.

    Args:
        pixels: (N, 2) projected pixel coordinates
        valid: (N,) boolean visibility mask
        masks_info: list of mask dicts for this frame (from masks.json)
        masks_dir: directory containing mask PNG files

    Returns:
        (N,) array of label IDs (0 = unlabelled/not visible)
    """
    n = len(pixels)
    labels = np.zeros(n, dtype=np.int32)

    if not masks_info:
        return labels

    # Load all masks for this frame and check each Gaussian
    loaded_masks = []
    for mask_info in masks_info:
        mask_path = os.path.join(masks_dir, mask_info["mask_file"])
        if os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            loaded_masks.append((mask_img, mask_info["_label_id"]))

    if not loaded_masks:
        return labels

    # For each visible Gaussian, check which mask it falls into
    valid_indices = np.where(valid)[0]
    for idx in valid_indices:
        u, v = int(pixels[idx, 0]), int(pixels[idx, 1])
        for mask_img, label_id in loaded_masks:
            if v < mask_img.shape[0] and u < mask_img.shape[1]:
                if mask_img[v, u] > 127:  # inside mask
                    labels[idx] = label_id
                    break  # first matching mask wins (highest confidence)

    return labels


def lift_semantics_to_3d(
    splat_path: str,
    masks_dir: str,
    colmap_dir: str,
    output_path: str,
) -> str:
    """
    Full semantic lifting pipeline.

    For each Gaussian:
    1. Project it into every camera where it's visible
    2. Look up the semantic mask at that pixel
    3. Majority-vote across all frames → final label

    Args:
        splat_path: path to input .ply Gaussian splat
        masks_dir: path to masks/ directory with masks.json
        colmap_dir: path to COLMAP workspace
        output_path: where to write the semantic .ply

    Returns:
        Path to the output semantic .ply file
    """
    print("\n═══ Semantic Lifting: 2D masks → 3D Gaussians ═══\n")

    # 1. Load Gaussian splat
    print("  Loading Gaussian splat...")
    xyz, properties = load_splat_ply(splat_path)
    n_gaussians = len(xyz)

    # 2. Load camera poses
    print("\n  Loading camera poses...")
    K, images = load_colmap_cameras(colmap_dir)

    # Get image dimensions from first frame
    frames_dir = os.path.join(colmap_dir, "images")
    sample_img = cv2.imread(os.path.join(frames_dir, images[0]["name"]))
    img_h, img_w = sample_img.shape[:2]
    print(f"  Image dimensions: {img_w}×{img_h}")

    # 3. Load masks
    print("\n  Loading semantic masks...")
    manifest, label_names = load_masks(masks_dir)

    # Build frame_name → mask_info lookup
    frame_to_masks = {}
    for frame_info in manifest:
        frame_to_masks[frame_info["frame"]] = frame_info.get("masks", [])

    # 4. Project and vote
    print(f"\n  Projecting {n_gaussians} Gaussians through {len(images)} cameras...")

    # vote_counts[i] = Counter of label votes for Gaussian i
    vote_counts = [Counter() for _ in range(n_gaussians)]
    frames_processed = 0

    for img_info in images:
        frame_name = img_info["name"]
        masks_info = frame_to_masks.get(frame_name, [])

        # Skip frames with no masks
        if not masks_info:
            continue

        # Project all Gaussians into this camera
        pixels, valid = project_gaussians_to_frame(
            xyz, K, img_info["R"], img_info["t"], img_w, img_h
        )

        # Look up labels
        frame_labels = assign_labels_for_frame(pixels, valid, masks_info, masks_dir)

        # Accumulate votes (only for non-zero labels)
        labelled = frame_labels > 0
        for idx in np.where(labelled)[0]:
            vote_counts[idx][frame_labels[idx]] += 1

        n_labelled = np.sum(labelled)
        frames_processed += 1

        if frames_processed % 10 == 0 or frames_processed == 1:
            print(
                f"    Frame {frames_processed}/{len(images)}: "
                f"{frame_name}, {n_labelled} Gaussians labelled"
            )

    # 5. Majority vote
    print(f"\n  Running majority vote across {frames_processed} frames...")
    final_labels = np.zeros(n_gaussians, dtype=np.int32)

    for i in range(n_gaussians):
        if vote_counts[i]:
            # Most common label wins
            final_labels[i] = vote_counts[i].most_common(1)[0][0]

    # Statistics
    labelled_count = np.sum(final_labels > 0)
    pct = 100 * labelled_count / n_gaussians
    print(f"  Labelled: {labelled_count}/{n_gaussians} Gaussians ({pct:.1f}%)")

    label_dist = Counter(final_labels.tolist())
    for label_id, count in sorted(label_dist.items()):
        name = label_names.get(label_id, "unknown")
        print(f"    {name}: {count} ({100*count/n_gaussians:.1f}%)")

    # 6. Save
    print(f"\n  Saving semantic splat...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_semantic_ply(output_path, xyz, properties, final_labels, label_names)

    print(f"\n═══ Semantic lifting complete ═══")
    return output_path


# ============================================================
# CLI
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project 2D semantic masks onto 3D Gaussians via majority voting"
    )
    parser.add_argument(
        "--splat",
        type=str,
        required=True,
        help="Path to input Gaussian splat .ply file",
    )
    parser.add_argument(
        "--masks",
        type=str,
        required=True,
        help="Path to masks/ directory (must contain masks.json)",
    )
    parser.add_argument(
        "--colmap",
        type=str,
        required=True,
        help="Path to COLMAP workspace (containing sparse/0/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for output semantic .ply file",
    )

    args = parser.parse_args()
    lift_semantics_to_3d(
        splat_path=args.splat,
        masks_dir=args.masks,
        colmap_dir=args.colmap,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
