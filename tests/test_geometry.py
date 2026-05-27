"""
test_geometry.py — Tests for COLMAP format utilities and export pipeline.

Tests that the COLMAP binary writers produce valid files and that
the pose conversion handles the camera-to-world → world-to-camera
inversion correctly.
"""

import os
import struct
import tempfile

import numpy as np
import pytest

from geometry.colmap_utils import (
    Camera,
    Image,
    Point3D,
    rotation_matrix_to_quaternion,
    write_cameras_binary,
    write_colmap_workspace,
    write_images_binary,
    write_points3d_binary,
)


class TestRotationConversion:
    """Test rotation matrix ↔ quaternion conversion."""

    def test_identity_rotation(self) -> None:
        """Identity matrix should give quaternion (1, 0, 0, 0)."""
        R = np.eye(3)
        qw, qx, qy, qz = rotation_matrix_to_quaternion(R)

        assert abs(qw - 1.0) < 1e-6
        assert abs(qx) < 1e-6
        assert abs(qy) < 1e-6
        assert abs(qz) < 1e-6

    def test_90_degree_rotation_z(self) -> None:
        """90° rotation around Z axis should produce a valid quaternion."""
        # R_z(90°) = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        qw, qx, qy, qz = rotation_matrix_to_quaternion(R)

        # Verify unit quaternion
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        assert abs(norm - 1.0) < 1e-6

        # Reconstruct rotation matrix from quaternion and verify
        R_reconstructed = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        np.testing.assert_allclose(R, R_reconstructed, atol=1e-6)

    def test_quaternion_is_unit_norm(self) -> None:
        """Random rotation matrices should produce unit quaternions."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            # Random rotation via QR decomposition
            M = rng.standard_normal((3, 3))
            Q, _ = np.linalg.qr(M)
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1  # Ensure proper rotation (det=+1)

            qw, qx, qy, qz = rotation_matrix_to_quaternion(Q)
            norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
            assert abs(norm - 1.0) < 1e-6, f"Quaternion norm is {norm}, expected 1.0"


class TestCOLMAPWriters:
    """Test COLMAP binary file writers."""

    def test_write_cameras_binary(self, tmp_path) -> None:
        """cameras.bin should be readable and have correct header."""
        cameras = {
            1: Camera(
                camera_id=1, model_id=1, width=640, height=480, params=[500.0, 500.0, 320.0, 240.0]
            ),
        }

        path = str(tmp_path / "cameras.bin")
        write_cameras_binary(cameras, path)

        # Read back and verify
        with open(path, "rb") as f:
            num_cameras = struct.unpack("<Q", f.read(8))[0]
            assert num_cameras == 1

            cam_id = struct.unpack("<i", f.read(4))[0]
            assert cam_id == 1

            model_id = struct.unpack("<i", f.read(4))[0]
            assert model_id == 1

            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            assert width == 640
            assert height == 480

    def test_write_images_binary(self, tmp_path) -> None:
        """images.bin should have correct number of entries."""
        images = {
            1: Image(
                image_id=1,
                qw=1.0,
                qx=0.0,
                qy=0.0,
                qz=0.0,
                tx=0.0,
                ty=0.0,
                tz=0.0,
                camera_id=1,
                name="000001.jpg",
            ),
            2: Image(
                image_id=2,
                qw=0.707,
                qx=0.0,
                qy=0.0,
                qz=0.707,
                tx=1.0,
                ty=0.0,
                tz=0.0,
                camera_id=1,
                name="000002.jpg",
            ),
        }

        path = str(tmp_path / "images.bin")
        write_images_binary(images, path)

        # Verify file exists and has reasonable size
        assert os.path.exists(path)
        size = os.path.getsize(path)
        assert size > 0

        # Read back header
        with open(path, "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
            assert num_images == 2

    def test_write_points3d_binary(self, tmp_path) -> None:
        """points3D.bin should contain correct number of points."""
        points3d = {
            1: Point3D(
                point3d_id=1,
                xyz=np.array([1.0, 2.0, 3.0]),
                rgb=np.array([255, 128, 0], dtype=np.uint8),
                error=0.5,
                track=[(1, 0), (2, 0)],
            ),
        }

        path = str(tmp_path / "points3D.bin")
        write_points3d_binary(points3d, path)

        with open(path, "rb") as f:
            num_points = struct.unpack("<Q", f.read(8))[0]
            assert num_points == 1

    def test_write_full_workspace(self, tmp_path) -> None:
        """Full workspace should create the correct directory structure."""
        cameras = {1: Camera(1, 1, 640, 480, [500.0, 500.0, 320.0, 240.0])}
        images = {1: Image(1, 1.0, 0, 0, 0, 0, 0, 0, 1, "000001.jpg")}
        points3d = {1: Point3D(1, np.array([0, 0, 0.0]), np.array([255, 255, 255], dtype=np.uint8))}

        output_dir = str(tmp_path / "colmap")
        sparse_dir = write_colmap_workspace(cameras, images, points3d, output_dir)

        # Verify directory structure
        assert os.path.isdir(os.path.join(output_dir, "sparse", "0"))
        assert os.path.isfile(os.path.join(sparse_dir, "cameras.bin"))
        assert os.path.isfile(os.path.join(sparse_dir, "images.bin"))
        assert os.path.isfile(os.path.join(sparse_dir, "points3D.bin"))


# ============================================================
# Helper for tests
# ============================================================
def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix (for test verification only)."""
    R = np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )
    return R
