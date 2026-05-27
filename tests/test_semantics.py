"""
test_semantics.py — Tests for semantic lifting and CLIP embeddings
"""

import json
import os
import struct
import tempfile
import shutil

import cv2
import numpy as np
import pytest


# ============================================================
# Helper: Create mock data for tests
# ============================================================
def create_mock_colmap_workspace(tmp_dir, num_frames=3, img_w=100, img_h=100):
    """Create a minimal COLMAP workspace with identity cameras."""
    sparse_dir = os.path.join(tmp_dir, "colmap", "sparse", "0")
    images_dir = os.path.join(tmp_dir, "colmap", "images")
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Write cameras.bin — PINHOLE model
    focal = float(max(img_w, img_h))
    cx, cy = img_w / 2.0, img_h / 2.0
    with open(os.path.join(sparse_dir, "cameras.bin"), "wb") as f:
        f.write(struct.pack("<Q", 1))  # num cameras
        f.write(struct.pack("<i", 1))  # camera_id
        f.write(struct.pack("<i", 1))  # PINHOLE
        f.write(struct.pack("<Q", img_w))
        f.write(struct.pack("<Q", img_h))
        for p in [focal, focal, cx, cy]:
            f.write(struct.pack("<d", p))

    # Write images.bin — identity poses, spaced along Z
    frame_names = [f"frame_{i:04d}.png" for i in range(num_frames)]
    with open(os.path.join(sparse_dir, "images.bin"), "wb") as f:
        f.write(struct.pack("<Q", num_frames))
        for i, fname in enumerate(frame_names):
            f.write(struct.pack("<i", i + 1))
            f.write(struct.pack("<d", 1.0))  # qw
            f.write(struct.pack("<d", 0.0))  # qx
            f.write(struct.pack("<d", 0.0))  # qy
            f.write(struct.pack("<d", 0.0))  # qz
            f.write(struct.pack("<d", 0.0))  # tx
            f.write(struct.pack("<d", 0.0))  # ty
            f.write(struct.pack("<d", -5.0 + i * 0.5))  # tz — step back
            f.write(struct.pack("<i", 1))    # camera_id
            f.write(fname.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))    # 0 keypoints

    # Create dummy frame images
    for fname in frame_names:
        img = np.random.randint(0, 255, (img_h, img_w, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(images_dir, fname), img)

    return os.path.join(tmp_dir, "colmap"), frame_names


def create_mock_splat_ply(tmp_dir, num_gaussians=50):
    """Create a minimal PLY file with Gaussian centres."""
    from plyfile import PlyData, PlyElement

    # Random Gaussians in a cube centered at origin
    xyz = np.random.uniform(-1, 1, (num_gaussians, 3)).astype(np.float32)

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("opacity", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    data = np.empty(num_gaussians, dtype=dtype)
    data["x"] = xyz[:, 0]
    data["y"] = xyz[:, 1]
    data["z"] = xyz[:, 2]
    data["opacity"] = np.ones(num_gaussians, dtype=np.float32)
    data["red"] = np.random.randint(0, 255, num_gaussians, dtype=np.uint8)
    data["green"] = np.random.randint(0, 255, num_gaussians, dtype=np.uint8)
    data["blue"] = np.random.randint(0, 255, num_gaussians, dtype=np.uint8)

    el = PlyElement.describe(data, "vertex")
    ply_path = os.path.join(tmp_dir, "splat.ply")
    PlyData([el], text=False).write(ply_path)
    return ply_path, xyz


def create_mock_masks(tmp_dir, frame_names, img_w=100, img_h=100):
    """
    Create mock masks — left half labelled 'chair', right half 'table'.
    """
    masks_dir = os.path.join(tmp_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    manifest = []
    for i, fname in enumerate(frame_names):
        base = os.path.splitext(fname)[0]
        masks_info = []

        # Mask 1: left half = 'chair'
        chair_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        chair_mask[:, :img_w // 2] = 255
        chair_file = f"{base}_mask_000.png"
        cv2.imwrite(os.path.join(masks_dir, chair_file), chair_mask)
        masks_info.append({
            "mask_file": chair_file,
            "label": "chair",
            "confidence": 0.9,
            "instance_id": 0,
            "_label_id": 1,
        })

        # Mask 2: right half = 'table'
        table_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        table_mask[:, img_w // 2:] = 255
        table_file = f"{base}_mask_001.png"
        cv2.imwrite(os.path.join(masks_dir, table_file), table_mask)
        masks_info.append({
            "mask_file": table_file,
            "label": "table",
            "confidence": 0.85,
            "instance_id": 1,
            "_label_id": 2,
        })

        manifest.append({
            "frame": fname,
            "frame_index": i,
            "masks": masks_info,
        })

    with open(os.path.join(masks_dir, "masks.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return masks_dir


# ============================================================
# Tests: Semantic Lifting
# ============================================================
class TestProjection:
    """Test 3D → 2D projection math."""

    def test_project_point_at_origin(self):
        """A Gaussian at the origin should project to the image centre."""
        from semantics.lift_to_3d import project_gaussians_to_frame

        xyz = np.array([[0.0, 0.0, 5.0]])  # 5m in front of camera
        K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=np.float64)
        R = np.eye(3)
        t = np.zeros(3)

        pixels, valid = project_gaussians_to_frame(xyz, K, R, t, 100, 100)

        assert valid[0], "Point should be visible"
        # Should project to (50, 50) — the image centre
        np.testing.assert_allclose(pixels[0], [50, 50], atol=0.1)

    def test_point_behind_camera_is_invalid(self):
        """A Gaussian behind the camera should be marked invalid."""
        from semantics.lift_to_3d import project_gaussians_to_frame

        xyz = np.array([[0.0, 0.0, -5.0]])  # behind camera
        K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=np.float64)
        R = np.eye(3)
        t = np.zeros(3)

        _, valid = project_gaussians_to_frame(xyz, K, R, t, 100, 100)

        assert not valid[0], "Point behind camera should be invalid"

    def test_point_outside_image_is_invalid(self):
        """A Gaussian that projects outside the image should be invalid."""
        from semantics.lift_to_3d import project_gaussians_to_frame

        # Far to the left — will project off-screen
        xyz = np.array([[-1000.0, 0.0, 1.0]])
        K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=np.float64)
        R = np.eye(3)
        t = np.zeros(3)

        _, valid = project_gaussians_to_frame(xyz, K, R, t, 100, 100)

        assert not valid[0], "Off-screen point should be invalid"


class TestMaskLookup:
    """Test mask-based label assignment."""

    def test_label_inside_mask(self):
        """A pixel inside a mask should get the mask's label."""
        from semantics.lift_to_3d import assign_labels_for_frame

        # Pixel at (25, 50) — inside the left-half mask
        pixels = np.array([[25.0, 50.0]])
        valid = np.array([True])
        masks_info = [{"mask_file": "test_mask.png", "_label_id": 1}]

        with tempfile.TemporaryDirectory() as tmp:
            # Create a left-half mask
            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[:, :50] = 255
            cv2.imwrite(os.path.join(tmp, "test_mask.png"), mask)

            labels = assign_labels_for_frame(pixels, valid, masks_info, tmp)

        assert labels[0] == 1

    def test_label_outside_mask(self):
        """A pixel outside all masks should get label 0 (unlabelled)."""
        from semantics.lift_to_3d import assign_labels_for_frame

        # Pixel at (75, 50) — outside the left-half mask
        pixels = np.array([[75.0, 50.0]])
        valid = np.array([True])
        masks_info = [{"mask_file": "test_mask.png", "_label_id": 1}]

        with tempfile.TemporaryDirectory() as tmp:
            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[:, :50] = 255
            cv2.imwrite(os.path.join(tmp, "test_mask.png"), mask)

            labels = assign_labels_for_frame(pixels, valid, masks_info, tmp)

        assert labels[0] == 0


class TestSemanticPLY:
    """Test the full semantic lifting pipeline output."""

    def test_output_ply_has_semantic_properties(self):
        """The output PLY should contain semantic_label and colour fields."""
        from semantics.lift_to_3d import load_splat_ply, save_semantic_ply

        with tempfile.TemporaryDirectory() as tmp:
            # Create a mock splat
            ply_path, _ = create_mock_splat_ply(tmp, num_gaussians=10)
            xyz, props = load_splat_ply(ply_path)

            # Save with labels
            labels = np.array([0, 1, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
            label_names = {0: "unlabelled", 1: "chair", 2: "table"}
            output_path = os.path.join(tmp, "semantic.ply")

            save_semantic_ply(output_path, xyz, props, labels, label_names)

            # Verify the output file exists and has the right properties
            assert os.path.exists(output_path)

            from plyfile import PlyData
            ply = PlyData.read(output_path)
            prop_names = ply["vertex"].data.dtype.names

            assert "semantic_label" in prop_names
            assert "semantic_r" in prop_names
            assert "semantic_g" in prop_names
            assert "semantic_b" in prop_names

    def test_label_mapping_json_saved(self):
        """A sidecar JSON with label mapping should be saved."""
        from semantics.lift_to_3d import load_splat_ply, save_semantic_ply

        with tempfile.TemporaryDirectory() as tmp:
            ply_path, _ = create_mock_splat_ply(tmp, num_gaussians=5)
            xyz, props = load_splat_ply(ply_path)

            labels = np.array([0, 1, 1, 2, 0], dtype=np.int32)
            label_names = {0: "unlabelled", 1: "chair", 2: "table"}
            output_path = os.path.join(tmp, "semantic.ply")

            save_semantic_ply(output_path, xyz, props, labels, label_names)

            json_path = output_path.replace(".ply", "_labels.json")
            assert os.path.exists(json_path)

            with open(json_path) as f:
                mapping = json.load(f)

            assert "labels" in mapping
            assert "1" in mapping["labels"]
            assert mapping["labels"]["1"] == "chair"


class TestMaskLoading:
    """Test mask manifest loading."""

    def test_load_masks_returns_label_mapping(self):
        """Loading masks should produce a label name mapping."""
        from semantics.lift_to_3d import load_masks

        with tempfile.TemporaryDirectory() as tmp:
            frame_names = ["frame_0000.png"]
            masks_dir = create_mock_masks(tmp, frame_names)
            manifest, label_names = load_masks(masks_dir)

            assert len(manifest) == 1
            assert 0 in label_names  # unlabelled
            assert "chair" in label_names.values()
            assert "table" in label_names.values()


class TestCropInstance:
    """Test instance cropping for CLIP."""

    def test_crop_returns_image(self):
        """Cropping an instance from a frame should return a valid image."""
        from semantics.clip_embeddings import crop_instance_from_frame

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255  # 60×60 region

        crop = crop_instance_from_frame(image, mask)

        assert crop is not None
        assert crop.shape[0] > 30  # at least 60 + some padding
        assert crop.shape[1] > 30
        assert crop.shape[2] == 3

    def test_crop_returns_none_for_tiny_mask(self):
        """Very small masks should return None."""
        from semantics.clip_embeddings import crop_instance_from_frame

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50:55, 50:55] = 255  # only 5×5 — too small

        crop = crop_instance_from_frame(image, mask, min_size=32)

        assert crop is None

    def test_crop_returns_none_for_empty_mask(self):
        """Empty masks should return None."""
        from semantics.clip_embeddings import crop_instance_from_frame

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)  # completely empty

        crop = crop_instance_from_frame(image, mask)

        assert crop is None
