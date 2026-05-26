"""
colmap_utils.py — Read/Write COLMAP binary format
===================================================

COLMAP is the standard format that gsplat (and most 3DGS tools) consume.
It consists of three binary files in a directory called "sparse/0/":

  cameras.bin  — camera intrinsics (focal length, image size)
  images.bin   — camera extrinsics (position + orientation per frame)
  points3D.bin — 3D point positions + colours

This module provides writers for all three. The binary format is documented at:
https://colmap.github.io/format.html#binary-file-format

We only support the PINHOLE camera model (model_id=1), which has 4 params:
  fx, fy, cx, cy  (focal lengths and principal point)

This is sufficient for phone cameras after undistortion.
"""

import os
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


# ============================================================
# Data Classes
# ============================================================
@dataclass
class Camera:
    """
    A camera intrinsics definition.

    In COLMAP, a "camera" defines the lens properties (focal length, image size).
    Multiple images can share the same camera (they were taken by the same phone).

    Fields:
        camera_id: unique integer ID (starts from 1)
        model_id: COLMAP camera model (1 = PINHOLE)
        width: image width in pixels
        height: image height in pixels
        params: model parameters — for PINHOLE: [fx, fy, cx, cy]
    """

    camera_id: int
    model_id: int  # 1 = PINHOLE
    width: int
    height: int
    params: List[float]  # [fx, fy, cx, cy] for PINHOLE


@dataclass
class Image:
    """
    A camera extrinsics definition (one per frame).

    In COLMAP, an "image" defines WHERE the camera was when a particular
    photo was taken: its rotation (quaternion) and translation.

    The convention is world-to-camera:
      point_camera = R @ point_world + t

    Fields:
        image_id: unique integer ID (starts from 1)
        qw, qx, qy, qz: rotation quaternion (world-to-camera)
        tx, ty, tz: translation vector (world-to-camera)
        camera_id: which Camera this image uses
        name: filename (e.g. "000001.jpg")
        point2d_ids: list of 2D keypoint observations (can be empty)
    """

    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str
    xys: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    point3d_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


@dataclass
class Point3D:
    """
    A 3D point in the reconstruction.

    Each point has a position (xyz), colour (rgb), reprojection error,
    and a "track" — the list of images that observe this point.

    Fields:
        point3d_id: unique integer ID (starts from 1)
        xyz: 3D position [x, y, z]
        rgb: colour [r, g, b] as uint8
        error: mean reprojection error (set to 0.0 if unknown)
        track: list of (image_id, point2d_idx) pairs
    """

    point3d_id: int
    xyz: np.ndarray  # shape (3,)
    rgb: np.ndarray  # shape (3,), uint8
    error: float = 0.0
    track: List[Tuple[int, int]] = field(default_factory=list)


# ============================================================
# Writers
# ============================================================
def write_cameras_binary(cameras: Dict[int, Camera], path: str) -> None:
    """
    Write cameras.bin in COLMAP binary format.

    Format:
      - 8 bytes: number of cameras (uint64)
      - For each camera:
        - 4 bytes: camera_id (int32)
        - 4 bytes: model_id (int32)
        - 8 bytes: width (uint64)
        - 8 bytes: height (uint64)
        - N * 8 bytes: params (float64 each) — N=4 for PINHOLE
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        # Number of cameras
        f.write(struct.pack("<Q", len(cameras)))

        for cam_id, cam in cameras.items():
            f.write(struct.pack("<i", cam.camera_id))
            f.write(struct.pack("<i", cam.model_id))
            f.write(struct.pack("<Q", cam.width))
            f.write(struct.pack("<Q", cam.height))
            for p in cam.params:
                f.write(struct.pack("<d", p))


def write_images_binary(images: Dict[int, Image], path: str) -> None:
    """
    Write images.bin in COLMAP binary format.

    Format:
      - 8 bytes: number of images (uint64)
      - For each image:
        - 4 bytes: image_id (int32)
        - 8*4 bytes: qw, qx, qy, qz (float64 each) — rotation quaternion
        - 8*3 bytes: tx, ty, tz (float64 each) — translation
        - 4 bytes: camera_id (int32)
        - N+1 bytes: image name as null-terminated string
        - 8 bytes: number of 2D points (uint64)
        - For each 2D point:
          - 8 bytes: x (float64)
          - 8 bytes: y (float64)
          - 8 bytes: point3d_id (int64) — -1 if not associated
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))

        for img_id, img in images.items():
            f.write(struct.pack("<i", img.image_id))
            f.write(struct.pack("<d", img.qw))
            f.write(struct.pack("<d", img.qx))
            f.write(struct.pack("<d", img.qy))
            f.write(struct.pack("<d", img.qz))
            f.write(struct.pack("<d", img.tx))
            f.write(struct.pack("<d", img.ty))
            f.write(struct.pack("<d", img.tz))
            f.write(struct.pack("<i", img.camera_id))

            # Image name as null-terminated string
            name_bytes = img.name.encode("utf-8") + b"\x00"
            f.write(name_bytes)

            # 2D points
            num_points2d = len(img.xys)
            f.write(struct.pack("<Q", num_points2d))
            for j in range(num_points2d):
                f.write(struct.pack("<d", img.xys[j, 0]))
                f.write(struct.pack("<d", img.xys[j, 1]))
                f.write(struct.pack("<q", img.point3d_ids[j]))


def write_points3d_binary(points3d: Dict[int, Point3D], path: str) -> None:
    """
    Write points3D.bin in COLMAP binary format.

    Format:
      - 8 bytes: number of points (uint64)
      - For each point:
        - 8 bytes: point3d_id (uint64)
        - 8*3 bytes: x, y, z (float64 each)
        - 3 bytes: r, g, b (uint8 each)
        - 8 bytes: error (float64)
        - 8 bytes: track_length (uint64)
        - For each track element:
          - 4 bytes: image_id (int32)
          - 4 bytes: point2d_idx (int32)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points3d)))

        for pt_id, pt in points3d.items():
            f.write(struct.pack("<Q", pt.point3d_id))
            f.write(struct.pack("<d", pt.xyz[0]))
            f.write(struct.pack("<d", pt.xyz[1]))
            f.write(struct.pack("<d", pt.xyz[2]))
            f.write(struct.pack("<B", int(pt.rgb[0])))
            f.write(struct.pack("<B", int(pt.rgb[1])))
            f.write(struct.pack("<B", int(pt.rgb[2])))
            f.write(struct.pack("<d", pt.error))
            f.write(struct.pack("<Q", len(pt.track)))
            for image_id, point2d_idx in pt.track:
                f.write(struct.pack("<i", image_id))
                f.write(struct.pack("<i", point2d_idx))


# ============================================================
# Convenience: write a complete COLMAP workspace
# ============================================================
def write_colmap_workspace(
    cameras: Dict[int, Camera],
    images: Dict[int, Image],
    points3d: Dict[int, Point3D],
    output_dir: str,
) -> str:
    """
    Write a complete COLMAP sparse reconstruction workspace.

    Creates:
      output_dir/
        sparse/
          0/
            cameras.bin
            images.bin
            points3D.bin

    This is the standard directory layout that gsplat and nerfstudio expect.

    Returns:
        Path to the sparse/0/ directory
    """
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)

    write_cameras_binary(cameras, os.path.join(sparse_dir, "cameras.bin"))
    write_images_binary(images, os.path.join(sparse_dir, "images.bin"))
    write_points3d_binary(points3d, os.path.join(sparse_dir, "points3D.bin"))

    print(f"    COLMAP workspace written to {sparse_dir}/")
    print(f"      cameras:  {len(cameras)}")
    print(f"      images:   {len(images)}")
    print(f"      points3D: {len(points3d)}")

    return sparse_dir


# ============================================================
# Helper: rotation matrix → quaternion
# ============================================================
def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Convert a 3x3 rotation matrix to a quaternion (qw, qx, qy, qz).

    Uses the Shepperd method which is numerically stable for all rotations.

    COLMAP convention: quaternion is (qw, qx, qy, qz) with qw first,
    representing a world-to-camera rotation.

    Args:
        R: 3x3 rotation matrix (numpy array)

    Returns:
        Tuple (qw, qx, qy, qz)
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    # Normalize
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    return float(qw / norm), float(qx / norm), float(qy / norm), float(qz / norm)
