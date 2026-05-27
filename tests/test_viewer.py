"""
test_viewer.py — Tests for the interactive 3D viewer
"""

import json
import os
import tempfile

import cv2
import numpy as np
import pytest


# ============================================================
# Colour Mode Tests
# ============================================================
class TestDepthColours:
    """Test depth-based colouring."""

    def test_returns_correct_shape(self):
        """Depth colours should be (N, 3) uint8."""
        from viewer.app import compute_depth_colours

        xyz = np.random.uniform(-1, 1, (100, 3)).astype(np.float32)
        colours = compute_depth_colours(xyz)

        assert colours.shape == (100, 3)
        assert colours.dtype == np.uint8

    def test_close_points_differ_from_far(self):
        """Close and far points should get different colours."""
        from viewer.app import compute_depth_colours

        # Line of points from near to far
        xyz = np.zeros((20, 3), dtype=np.float32)
        xyz[:, 2] = np.linspace(0, 10, 20)

        colours = compute_depth_colours(xyz, camera_pos=np.array([0, 0, 0]))

        # Near and far points should have different colours
        near_colour = colours[0]
        far_colour = colours[-1]
        assert not np.array_equal(near_colour, far_colour)


class TestViridisColormap:
    """Test the viridis colourmap approximation."""

    def test_output_range(self):
        """Colourmap output should be in [0, 255]."""
        from viewer.app import _viridis_colormap

        values = np.linspace(0, 1, 50)
        colours = _viridis_colormap(values)

        assert colours.shape == (50, 3)
        assert colours.dtype == np.uint8
        assert colours.min() >= 0
        assert colours.max() <= 255

    def test_zero_is_dark_purple(self):
        """Value 0 should produce dark purple (viridis start)."""
        from viewer.app import _viridis_colormap

        colours = _viridis_colormap(np.array([0.0]))
        # Should be approximately (68, 1, 84) — dark purple
        assert colours[0, 0] < 100  # R
        assert colours[0, 1] < 50   # G low
        assert colours[0, 2] > 50   # B high-ish

    def test_one_is_yellow(self):
        """Value 1 should produce yellow (viridis end)."""
        from viewer.app import _viridis_colormap

        colours = _viridis_colormap(np.array([1.0]))
        # Should be approximately (253, 231, 37) — yellow
        assert colours[0, 0] > 200  # R high
        assert colours[0, 1] > 200  # G high
        assert colours[0, 2] < 100  # B low


class TestNormalColours:
    """Test normal estimation colouring."""

    def test_returns_correct_shape(self):
        """Normal colours should be (N, 3) uint8."""
        from viewer.app import compute_normal_colours

        xyz = np.random.uniform(-1, 1, (50, 3)).astype(np.float32)
        colours = compute_normal_colours(xyz, k=5)

        assert colours.shape == (50, 3)
        assert colours.dtype == np.uint8

    def test_planar_points_have_consistent_normals(self):
        """Points on a plane should have similar normal colours."""
        from viewer.app import compute_normal_colours

        # Points on the XY plane (Z=0)
        n = 30
        xyz = np.zeros((n, 3), dtype=np.float32)
        xyz[:, 0] = np.random.uniform(-1, 1, n)
        xyz[:, 1] = np.random.uniform(-1, 1, n)
        xyz[:, 2] = 0.0  # all on Z=0 plane

        colours = compute_normal_colours(xyz, k=5)

        # The Z component of normals should dominate → all should have
        # similar colours (high blue channel = |nz|)
        blue_values = colours[:, 2]
        assert np.std(blue_values) < 50  # low variance in blue channel


# ============================================================
# Data Loading Tests
# ============================================================
class TestLoadSemanticPly:
    """Test PLY loading."""

    def test_loads_basic_ply(self):
        """Should load a minimal PLY with xyz + rgb."""
        from viewer.app import load_semantic_ply
        from plyfile import PlyData, PlyElement

        with tempfile.TemporaryDirectory() as tmp:
            n = 20
            data = np.empty(n, dtype=[
                ("x", "f4"), ("y", "f4"), ("z", "f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ])
            data["x"] = np.random.uniform(-1, 1, n)
            data["y"] = np.random.uniform(-1, 1, n)
            data["z"] = np.random.uniform(-1, 1, n)
            data["red"] = np.random.randint(0, 255, n)
            data["green"] = np.random.randint(0, 255, n)
            data["blue"] = np.random.randint(0, 255, n)

            el = PlyElement.describe(data, "vertex")
            ply_path = os.path.join(tmp, "test.ply")
            PlyData([el]).write(ply_path)

            result = load_semantic_ply(ply_path)

            assert result["xyz"].shape == (n, 3)
            assert result["rgb"].shape == (n, 3)
            assert result["covariances"].shape == (n, 3, 3)

    def test_loads_semantic_properties(self):
        """Should detect semantic_label and semantic_rgb properties."""
        from viewer.app import load_semantic_ply
        from plyfile import PlyData, PlyElement

        with tempfile.TemporaryDirectory() as tmp:
            n = 10
            data = np.empty(n, dtype=[
                ("x", "f4"), ("y", "f4"), ("z", "f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                ("semantic_label", "u2"),
                ("semantic_r", "u1"), ("semantic_g", "u1"), ("semantic_b", "u1"),
            ])
            data["x"] = np.random.uniform(-1, 1, n)
            data["y"] = np.random.uniform(-1, 1, n)
            data["z"] = np.random.uniform(-1, 1, n)
            data["red"] = data["green"] = data["blue"] = 128
            data["semantic_label"] = [0, 1, 1, 2, 0, 1, 2, 0, 1, 2]
            data["semantic_r"] = [128, 255, 255, 0, 128, 255, 0, 128, 255, 0]
            data["semantic_g"] = [128, 0, 0, 255, 128, 0, 255, 128, 0, 255]
            data["semantic_b"] = [128, 0, 0, 0, 128, 0, 0, 128, 0, 0]

            el = PlyElement.describe(data, "vertex")
            ply_path = os.path.join(tmp, "test.ply")
            PlyData([el]).write(ply_path)

            result = load_semantic_ply(ply_path)

            assert result["semantic_label"] is not None
            assert result["semantic_rgb"] is not None
            assert result["semantic_label"].shape == (n,)
            assert result["semantic_rgb"].shape == (n, 3)


class TestLabelMapping:
    """Test label mapping loading."""

    def test_loads_json_mapping(self):
        """Should load label mapping from sidecar JSON."""
        from viewer.app import load_label_mapping

        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "splat_semantic_labels.json")
            with open(json_path, "w") as f:
                json.dump({"labels": {"0": "unlabelled", "1": "chair", "2": "table"}}, f)

            ply_path = os.path.join(tmp, "splat_semantic.ply")
            result = load_label_mapping(ply_path)

            assert result[0] == "unlabelled"
            assert result[1] == "chair"
            assert result[2] == "table"

    def test_returns_default_when_missing(self):
        """Should return default mapping when JSON doesn't exist."""
        from viewer.app import load_label_mapping

        result = load_label_mapping("/nonexistent/path.ply")
        assert 0 in result
        assert result[0] == "unlabelled"


# ============================================================
# Occupancy Map Tests
# ============================================================
class TestOccupancyMap:
    """Test top-down occupancy map generation."""

    def test_returns_image(self):
        """Should return an (H, W, 3) uint8 image."""
        from viewer.app import generate_occupancy_map

        n = 100
        xyz = np.random.uniform(-2, 2, (n, 3)).astype(np.float32)
        labels = np.random.choice([0, 1, 2], n).astype(np.uint16)
        label_names = {0: "unlabelled", 1: "chair", 2: "table"}

        occ = generate_occupancy_map(xyz, labels, label_names, resolution=0.1)

        assert occ.ndim == 3
        assert occ.shape[2] == 3
        assert occ.dtype == np.uint8

    def test_saves_to_file(self):
        """Should save the occupancy map as PNG when path is given."""
        from viewer.app import generate_occupancy_map

        with tempfile.TemporaryDirectory() as tmp:
            n = 50
            xyz = np.random.uniform(-1, 1, (n, 3)).astype(np.float32)
            labels = np.ones(n, dtype=np.uint16)
            label_names = {0: "unlabelled", 1: "chair"}
            output = os.path.join(tmp, "occ.png")

            generate_occupancy_map(xyz, labels, label_names,
                                   resolution=0.1, output_path=output)

            assert os.path.exists(output)
            img = cv2.imread(output)
            assert img is not None


# ============================================================
# Covariance Computation Tests
# ============================================================
class TestCovariances:
    """Test covariance matrix computation."""

    def test_default_covariances_are_isotropic(self):
        """Without scales/rotations, covariances should be isotropic."""
        from viewer.app import _compute_covariances

        covs = _compute_covariances(None, None, n=5)

        assert covs.shape == (5, 3, 3)
        # Each should be scalar × I
        for i in range(5):
            np.testing.assert_allclose(covs[i, 0, 1], 0.0, atol=1e-6)
            np.testing.assert_allclose(covs[i, 0, 2], 0.0, atol=1e-6)
            assert covs[i, 0, 0] > 0  # positive diagonal

    def test_covariances_are_symmetric(self):
        """Computed covariances should be symmetric matrices."""
        from viewer.app import _compute_covariances

        scales = np.random.uniform(-2, 0, (10, 3)).astype(np.float32)
        # Random unit quaternions
        rots = np.random.randn(10, 4).astype(np.float32)
        rots = rots / np.linalg.norm(rots, axis=1, keepdims=True)

        covs = _compute_covariances(scales, rots, n=10)

        for i in range(10):
            np.testing.assert_allclose(covs[i], covs[i].T, atol=1e-5)
