"""
export_colmap.py — Convert MASt3R-SLAM output to COLMAP workspace
=================================================================

MASt3R-SLAM outputs to a logs/ directory containing:
  - .ply point cloud files (dense 3D reconstruction)
  - Trajectory / pose data (format varies by version)

gsplat needs:
  - COLMAP binary format: cameras.bin, images.bin, points3D.bin
  - Images in a specific directory layout

This script bridges the two formats. It supports multiple input types:
  1. .ply point cloud file (always available from MASt3R-SLAM)
  2. .npy pose arrays (if MASt3R-SLAM saved them)
  3. Trajectory text files (TUM/KITTI format if available)

Usage:
  python -m geometry.export_colmap \\
      --ply /path/to/slam_output.ply \\
      --frames_dir /path/to/frames/ \\
      --colmap_dir /path/to/colmap/

  python -m geometry.export_colmap \\
      --slam_output /path/to/slam_logs/ \\
      --frames_dir /path/to/frames/ \\
      --colmap_dir /path/to/colmap/
"""

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from geometry.colmap_utils import (
    Camera,
    Image,
    Point3D,
    rotation_matrix_to_quaternion,
    write_colmap_workspace,
)


# ============================================================
# PLY Loading (primary path for MASt3R-SLAM)
# ============================================================
def load_ply(ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a .ply point cloud file.

    MASt3R-SLAM saves its reconstruction as a .ply file in the logs/
    directory. This function reads the vertex positions and colours.

    Supports both ASCII and binary .ply format (via plyfile library
    if available, else a simple ASCII parser).

    Args:
        ply_path: path to the .ply file

    Returns:
        Tuple of (xyz array [N, 3], rgb array [N, 3] as uint8)
    """
    try:
        from plyfile import PlyData

        ply = PlyData.read(ply_path)
        vertices = ply["vertex"]
        xyz = np.column_stack([vertices["x"], vertices["y"], vertices["z"]])

        # Get colours if available
        if "red" in vertices.data.dtype.names:
            rgb = np.column_stack([vertices["red"], vertices["green"], vertices["blue"]]).astype(
                np.uint8
            )
        else:
            rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)

        print(f"    Loaded .ply: {len(xyz)} points")
        return xyz, rgb

    except ImportError:
        # Fallback: simple ASCII .ply parser
        return _load_ply_ascii(ply_path)


def _load_ply_ascii(ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Simple ASCII .ply parser (no external dependencies)."""
    with open(ply_path, "r") as f:
        lines = f.readlines()

    # Parse header
    num_vertices = 0
    header_end = 0
    has_colour = False
    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            num_vertices = int(line.split()[-1])
        if "red" in line or "diffuse_red" in line:
            has_colour = True
        if line.strip() == "end_header":
            header_end = i + 1
            break

    # Parse vertices
    xyz = np.zeros((num_vertices, 3), dtype=np.float64)
    rgb = np.full((num_vertices, 3), 128, dtype=np.uint8)

    for i in range(num_vertices):
        parts = lines[header_end + i].split()
        xyz[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
        if has_colour and len(parts) >= 6:
            rgb[i] = [int(parts[3]), int(parts[4]), int(parts[5])]

    print(f"    Loaded .ply (ASCII): {num_vertices} points")
    return xyz, rgb


# ============================================================
# Pose Loading (multiple format support)
# ============================================================
def load_poses_npy(slam_output_dir: str) -> Optional[np.ndarray]:
    """
    Try to load camera poses from .npy files.

    Args:
        slam_output_dir: directory containing MASt3R-SLAM output

    Returns:
        (N, 4, 4) array of camera-to-world transforms, or None
    """
    poses_path = os.path.join(slam_output_dir, "poses.npy")
    if os.path.exists(poses_path):
        poses = np.load(poses_path)
        print(f"    Loaded poses from .npy: {poses.shape}")
        return poses

    # Search for any .npy files that might contain poses
    npy_files = glob.glob(os.path.join(slam_output_dir, "**/*.npy"), recursive=True)
    for f in npy_files:
        data = np.load(f)
        if data.ndim == 3 and data.shape[1:] == (4, 4):
            print(f"    Found poses in {f}: {data.shape}")
            return data

    return None


def load_poses_txt(slam_output_dir: str) -> Optional[np.ndarray]:
    """
    Try to load camera poses from trajectory text files.

    Supports TUM format (timestamp tx ty tz qx qy qz qw) and
    KITTI format (3x4 flattened matrices, one per line).

    Args:
        slam_output_dir: directory containing trajectory files

    Returns:
        (N, 4, 4) array of camera-to-world transforms, or None
    """
    # Search for trajectory files
    traj_candidates = (
        glob.glob(os.path.join(slam_output_dir, "**/*traj*.txt"), recursive=True)
        + glob.glob(os.path.join(slam_output_dir, "**/*pose*.txt"), recursive=True)
        + glob.glob(os.path.join(slam_output_dir, "**/CameraTrajectory.txt"), recursive=True)
    )

    for traj_path in traj_candidates:
        try:
            with open(traj_path, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("#")]

            if not lines:
                continue

            parts_per_line = len(lines[0].split())

            if parts_per_line == 8:
                # TUM format: timestamp tx ty tz qx qy qz qw
                poses = _parse_tum_trajectory(lines)
                print(f"    Loaded TUM trajectory from {traj_path}: {len(poses)} poses")
                return poses

            elif parts_per_line == 12:
                # KITTI format: 3x4 matrix flattened
                poses = _parse_kitti_trajectory(lines)
                print(f"    Loaded KITTI trajectory from {traj_path}: {len(poses)} poses")
                return poses

        except Exception as e:
            print(f"    Could not parse {traj_path}: {e}")
            continue

    return None


def _parse_tum_trajectory(lines: List[str]) -> np.ndarray:
    """Parse TUM format: timestamp tx ty tz qx qy qz qw → (N, 4, 4)."""
    poses = []
    for line in lines:
        parts = line.split()
        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
        qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

        # Quaternion to rotation matrix
        R = _quat_to_rot(qw, qx, qy, qz)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        poses.append(T)

    return np.array(poses)


def _parse_kitti_trajectory(lines: List[str]) -> np.ndarray:
    """Parse KITTI format: 12 floats per line (3x4 matrix) → (N, 4, 4)."""
    poses = []
    for line in lines:
        parts = [float(x) for x in line.split()]
        T = np.eye(4)
        T[:3, :] = np.array(parts).reshape(3, 4)
        poses.append(T)

    return np.array(poses)


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Quaternion (w, x, y, z) to 3x3 rotation matrix."""
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )


# ============================================================
# Point Cloud to COLMAP Points3D
# ============================================================
def ply_to_colmap_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    max_points: int = 500_000,
) -> Dict[int, Point3D]:
    """
    Convert a .ply point cloud to COLMAP Point3D entries.

    Subsamples if the point cloud is larger than max_points.
    Filters out invalid points (NaN, very far, or at origin).

    Args:
        xyz: (N, 3) point positions
        rgb: (N, 3) point colours as uint8
        max_points: maximum number of points to output

    Returns:
        Dict mapping point3d_id → Point3D
    """
    # Filter invalid points
    valid = np.all(np.isfinite(xyz), axis=1)
    norms = np.linalg.norm(xyz, axis=1)
    valid &= norms > 1e-6  # not at origin
    valid &= norms < 100.0  # not absurdly far
    xyz = xyz[valid]
    rgb = rgb[valid]

    print(f"    Valid points after filtering: {len(xyz)}")

    # Subsample if needed
    if len(xyz) > max_points:
        indices = np.random.choice(len(xyz), max_points, replace=False)
        xyz = xyz[indices]
        rgb = rgb[indices]
        print(f"    Subsampled to {max_points} points")

    # Build Point3D dict
    points3d = {}
    for j in range(len(xyz)):
        point_id = j + 1
        points3d[point_id] = Point3D(
            point3d_id=point_id,
            xyz=np.array(xyz[j], dtype=np.float64),
            rgb=np.array(rgb[j], dtype=np.uint8),
            error=0.0,
            track=[],
        )

    print(f"    Final point cloud: {len(points3d)} points")
    return points3d


# ============================================================
# Pose Conversion
# ============================================================
def poses_to_colmap_images(
    poses: np.ndarray,
    frame_names: List[str],
    camera_id: int = 1,
) -> Dict[int, Image]:
    """
    Convert camera-to-world poses to COLMAP Image entries.

    IMPORTANT CONVENTION DIFFERENCE:
      - MASt3R-SLAM outputs camera-to-world: P_world = T @ P_camera
      - COLMAP expects world-to-camera: P_camera = R @ P_world + t

    So we invert each pose matrix before extracting R and t.

    Args:
        poses: (N, 4, 4) camera-to-world transformation matrices
        frame_names: list of frame filenames (e.g. ["000001.jpg", ...])
        camera_id: which Camera ID these images belong to

    Returns:
        Dict mapping image_id → Image
    """
    images = {}

    for i in range(min(len(poses), len(frame_names))):
        # Invert: camera-to-world → world-to-camera
        T_c2w = poses[i]  # 4x4 camera-to-world
        T_w2c = np.linalg.inv(T_c2w)  # 4x4 world-to-camera

        # Extract rotation matrix (3x3) and translation (3,)
        R = T_w2c[:3, :3]
        t = T_w2c[:3, 3]

        # Convert rotation matrix to quaternion
        qw, qx, qy, qz = rotation_matrix_to_quaternion(R)

        image_id = i + 1  # COLMAP IDs start from 1
        images[image_id] = Image(
            image_id=image_id,
            qw=qw,
            qx=qx,
            qy=qy,
            qz=qz,
            tx=float(t[0]),
            ty=float(t[1]),
            tz=float(t[2]),
            camera_id=camera_id,
            name=frame_names[i],
        )

    return images


def make_identity_images(
    frame_names: List[str],
    camera_id: int = 1,
) -> Dict[int, Image]:
    """
    Create placeholder COLMAP Image entries with identity poses.

    Used as fallback when MASt3R-SLAM trajectory data is not available.
    gsplat can optimise camera poses during training if initialised
    with reasonable starting positions.

    Args:
        frame_names: list of frame filenames
        camera_id: camera ID to assign

    Returns:
        Dict mapping image_id → Image (identity rotation, spaced translations)
    """
    images = {}
    for i, name in enumerate(frame_names):
        image_id = i + 1
        images[image_id] = Image(
            image_id=image_id,
            qw=1.0,
            qx=0.0,
            qy=0.0,
            qz=0.0,
            tx=i * 0.1,
            ty=0.0,
            tz=0.0,  # evenly spaced along X
            camera_id=camera_id,
            name=name,
        )
    return images


# ============================================================
# Full Export Pipeline
# ============================================================
def export_to_colmap(
    frames_dir: str,
    colmap_output_dir: str,
    ply_path: Optional[str] = None,
    slam_output_dir: Optional[str] = None,
    max_points: int = 500_000,
) -> str:
    """
    Full export pipeline: MASt3R-SLAM output → COLMAP workspace.

    Tries multiple strategies to find pose and point data:
    1. If ply_path is given, load the point cloud from it
    2. If slam_output_dir is given, search for .ply, .npy, and .txt files
    3. Falls back to identity poses if no trajectory data is found

    Creates the directory structure that gsplat expects:
      colmap_output_dir/
        images/           ← copied frame images
        sparse/
          0/
            cameras.bin
            images.bin
            points3D.bin

    Args:
        frames_dir: directory containing preprocessed frame JPEGs
        colmap_output_dir: where to write the COLMAP workspace
        ply_path: path to .ply point cloud file (optional)
        slam_output_dir: directory containing MASt3R-SLAM logs (optional)
        max_points: maximum number of 3D points

    Returns:
        Path to the COLMAP workspace directory
    """
    import cv2

    # Get frame filenames (sorted)
    frame_names = sorted([f for f in os.listdir(frames_dir) if f.endswith((".jpg", ".png"))])
    num_frames = len(frame_names)
    print(f"\n  Found {num_frames} frames in {frames_dir}")

    # --- Load point cloud ---
    print("\n  Loading 3D point cloud...")
    xyz, rgb = None, None

    if ply_path and os.path.exists(ply_path):
        xyz, rgb = load_ply(ply_path)
    elif slam_output_dir:
        # Search for .ply files in the SLAM output
        ply_files = glob.glob(os.path.join(slam_output_dir, "**/*.ply"), recursive=True)
        if ply_files:
            largest_ply = max(ply_files, key=os.path.getsize)
            print(f"    Found .ply: {largest_ply}")
            xyz, rgb = load_ply(largest_ply)

    # --- Load poses ---
    print("\n  Loading camera poses...")
    poses = None

    if slam_output_dir:
        # Try .npy first, then .txt
        poses = load_poses_npy(slam_output_dir)
        if poses is None:
            poses = load_poses_txt(slam_output_dir)

    # --- Camera intrinsics ---
    print("\n  Building camera intrinsics...")
    first_frame = cv2.imread(os.path.join(frames_dir, frame_names[0]))
    img_h, img_w = first_frame.shape[:2]

    # Estimate intrinsics (standard phone camera approximation)
    focal = float(max(img_w, img_h))
    cx = float(img_w) / 2.0
    cy = float(img_h) / 2.0

    cameras = {
        1: Camera(
            camera_id=1,
            model_id=1,  # PINHOLE
            width=img_w,
            height=img_h,
            params=[focal, focal, cx, cy],
        )
    }
    print(f"    Camera: {img_w}x{img_h}, focal={focal:.0f}")

    # --- Camera extrinsics ---
    print("\n  Building camera extrinsics...")
    if poses is not None:
        images = poses_to_colmap_images(poses, frame_names, camera_id=1)
        print(f"    Using {len(images)} recovered poses")
    else:
        print("    ⚠ No trajectory data found — using identity poses (gsplat will optimise)")
        images = make_identity_images(frame_names, camera_id=1)

    # --- 3D Points ---
    print("\n  Building sparse point cloud...")
    if xyz is not None:
        points3d = ply_to_colmap_points(xyz, rgb, max_points)
    else:
        print("    ⚠ No point cloud found — writing empty points3D.bin")
        points3d = {}

    # --- Write COLMAP workspace ---
    print("\n  Writing COLMAP workspace...")
    os.makedirs(colmap_output_dir, exist_ok=True)

    # Copy images into the workspace
    images_target = os.path.join(colmap_output_dir, "images")
    if not os.path.exists(images_target):
        try:
            os.symlink(os.path.abspath(frames_dir), images_target)
        except OSError:
            shutil.copytree(frames_dir, images_target)

    sparse_dir = write_colmap_workspace(
        cameras=cameras,
        images=images,
        points3d=points3d,
        output_dir=colmap_output_dir,
    )

    print(f"\n  ✓ COLMAP workspace ready at {colmap_output_dir}/")
    print(f"    Images:  {images_target}/")
    print(f"    Sparse:  {sparse_dir}/")

    return colmap_output_dir


# ============================================================
# CLI Entry Point
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MASt3R-SLAM output to COLMAP workspace for gsplat"
    )
    parser.add_argument(
        "--ply",
        type=str,
        default=None,
        help="Path to .ply point cloud from MASt3R-SLAM",
    )
    parser.add_argument(
        "--slam_output",
        type=str,
        default=None,
        help="Directory containing MASt3R-SLAM logs (searched for .ply, .npy, .txt)",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        required=True,
        help="Directory containing preprocessed frame JPEGs",
    )
    parser.add_argument(
        "--colmap_dir",
        type=str,
        required=True,
        help="Output directory for COLMAP workspace",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=500_000,
        help="Maximum number of 3D points (default: 500000)",
    )

    args = parser.parse_args()

    if not args.ply and not args.slam_output:
        parser.error("At least one of --ply or --slam_output must be provided")

    export_to_colmap(
        frames_dir=args.frames_dir,
        colmap_output_dir=args.colmap_dir,
        ply_path=args.ply,
        slam_output_dir=args.slam_output,
        max_points=args.max_points,
    )


if __name__ == "__main__":
    main()
