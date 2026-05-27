"""
test_eval.py — Tests for evaluation metrics
"""

import json
import os
import tempfile

import cv2
import numpy as np
import pytest


# ============================================================
# PSNR Tests
# ============================================================
class TestPSNR:
    """Test Peak Signal-to-Noise Ratio computation."""

    def test_identical_images_return_inf(self):
        """Identical images should produce infinite PSNR."""
        from eval.metrics import compute_psnr

        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        assert compute_psnr(img, img) == float("inf")

    def test_different_images_return_finite(self):
        """Different images should produce a finite, positive PSNR."""
        from eval.metrics import compute_psnr

        img1 = np.full((50, 50, 3), 100, dtype=np.uint8)
        img2 = np.full((50, 50, 3), 150, dtype=np.uint8)

        psnr = compute_psnr(img1, img2)
        assert psnr > 0
        assert psnr < 100

    def test_more_similar_gives_higher_psnr(self):
        """More similar images should produce higher PSNR."""
        from eval.metrics import compute_psnr

        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        noisy_small = img.astype(np.int16) + np.random.randint(-5, 5, img.shape)
        noisy_large = img.astype(np.int16) + np.random.randint(-50, 50, img.shape)

        noisy_small = np.clip(noisy_small, 0, 255).astype(np.uint8)
        noisy_large = np.clip(noisy_large, 0, 255).astype(np.uint8)

        psnr_small_noise = compute_psnr(img, noisy_small)
        psnr_large_noise = compute_psnr(img, noisy_large)

        assert psnr_small_noise > psnr_large_noise

    def test_known_psnr_value(self):
        """Test against a known MSE → PSNR conversion."""
        from eval.metrics import compute_psnr

        # MSE = 100 → PSNR = 10*log10(255^2/100) ≈ 28.13 dB
        img1 = np.zeros((10, 10, 3), dtype=np.uint8)
        img2 = np.full((10, 10, 3), 10, dtype=np.uint8)  # MSE = 100

        psnr = compute_psnr(img1, img2)
        expected = 10 * np.log10(255**2 / 100)
        np.testing.assert_allclose(psnr, expected, atol=0.01)


# ============================================================
# SSIM Tests
# ============================================================
class TestSSIM:
    """Test Structural Similarity Index computation."""

    def test_identical_images_return_one(self):
        """Identical images should produce SSIM ≈ 1.0."""
        from eval.metrics import compute_ssim

        img = np.random.randint(50, 200, (60, 60, 3), dtype=np.uint8)
        ssim = compute_ssim(img, img)

        # Should be very close to 1 (may not be exactly 1 due to numerical precision)
        assert ssim > 0.99

    def test_different_images_return_less_than_one(self):
        """Different images should produce SSIM < 1.0."""
        from eval.metrics import compute_ssim

        img1 = np.full((60, 60, 3), 100, dtype=np.uint8)
        img2 = np.random.randint(0, 255, (60, 60, 3), dtype=np.uint8)

        ssim = compute_ssim(img1, img2)
        assert ssim < 1.0

    def test_ssim_range(self):
        """SSIM should be in [-1, 1]."""
        from eval.metrics import compute_ssim

        img1 = np.random.randint(0, 255, (60, 60, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (60, 60, 3), dtype=np.uint8)

        ssim = compute_ssim(img1, img2)
        assert -1.0 <= ssim <= 1.0


# ============================================================
# Chamfer Distance Tests
# ============================================================
class TestChamferDistance:
    """Test Chamfer distance computation."""

    def test_identical_clouds_return_zero(self):
        """Chamfer distance of a cloud to itself should be 0."""
        from eval.metrics import compute_chamfer_distance

        cloud = np.random.uniform(-1, 1, (100, 3))
        result = compute_chamfer_distance(cloud, cloud.copy())

        assert result["chamfer"] < 1e-10

    def test_shifted_cloud_returns_shift_magnitude(self):
        """A shifted cloud should have Chamfer ≈ shift distance."""
        from eval.metrics import compute_chamfer_distance

        # Use a tight cluster so nearest-neighbour distances ≈ shift
        cloud = np.random.uniform(-0.01, 0.01, (200, 3))
        shift = np.array([2.0, 0.0, 0.0])
        shifted = cloud + shift

        result = compute_chamfer_distance(cloud, shifted)

        # Chamfer should be approximately 2.0 (the shift distance)
        np.testing.assert_allclose(result["chamfer"], 2.0, atol=0.05)

    def test_returns_expected_keys(self):
        """Result dict should contain all expected keys."""
        from eval.metrics import compute_chamfer_distance

        cloud1 = np.random.uniform(-1, 1, (50, 3))
        cloud2 = np.random.uniform(-1, 1, (60, 3))

        result = compute_chamfer_distance(cloud1, cloud2)

        assert "chamfer" in result
        assert "source_to_target" in result
        assert "target_to_source" in result
        assert "source_points" in result
        assert "target_points" in result


# ============================================================
# Rendering Evaluation Tests
# ============================================================
class TestEvaluateRendering:
    """Test the full rendering evaluation pipeline."""

    def test_identical_frames_give_perfect_scores(self):
        """Comparing identical frames should produce PSNR=inf, SSIM≈1."""
        from eval.metrics import evaluate_rendering

        with tempfile.TemporaryDirectory() as tmp:
            gt_dir = os.path.join(tmp, "gt")
            rendered_dir = os.path.join(tmp, "rendered")
            os.makedirs(gt_dir)
            os.makedirs(rendered_dir)

            # Create identical frames
            for i in range(10):
                img = np.random.randint(50, 200, (60, 60, 3), dtype=np.uint8)
                name = f"frame_{i:04d}.png"
                cv2.imwrite(os.path.join(gt_dir, name), img)
                cv2.imwrite(os.path.join(rendered_dir, name), img)

            result = evaluate_rendering(gt_dir, rendered_dir, holdout_fraction=0.3)

            assert result["psnr_mean"] == float("inf")
            assert result["ssim_mean"] > 0.99
            assert result["num_test_frames"] > 0

    def test_empty_dirs_return_zero(self):
        """Empty directories should return zero metrics."""
        from eval.metrics import evaluate_rendering

        with tempfile.TemporaryDirectory() as tmp:
            gt_dir = os.path.join(tmp, "gt")
            rendered_dir = os.path.join(tmp, "rendered")
            os.makedirs(gt_dir)
            os.makedirs(rendered_dir)

            result = evaluate_rendering(gt_dir, rendered_dir)

            assert result["num_test_frames"] == 0
            assert result["psnr_mean"] == 0.0


# ============================================================
# Results Saving Tests
# ============================================================
class TestResultsSaving:
    """Test results JSON serialisation."""

    def test_saves_and_loads_results(self):
        """Should write and read back results correctly."""
        from eval.metrics import _save_results

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.json")

            results = {
                "scene": "test_scene",
                "rendering": {"psnr_mean": 25.5, "ssim_mean": 0.88},
            }

            _save_results(path, "test_scene", results)

            assert os.path.exists(path)
            with open(path, "r") as f:
                saved = json.load(f)

            assert "scenes" in saved
            assert "test_scene" in saved["scenes"]
            assert saved["scenes"]["test_scene"]["rendering"]["psnr_mean"] == 25.5

    def test_updates_existing_results(self):
        """Should update existing results without overwriting other scenes."""
        from eval.metrics import _save_results

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.json")

            _save_results(path, "scene1", {"psnr": 25.0})
            _save_results(path, "scene2", {"psnr": 28.0})

            with open(path, "r") as f:
                saved = json.load(f)

            assert "scene1" in saved["scenes"]
            assert "scene2" in saved["scenes"]


# ============================================================
# Precision@K Tests
# ============================================================
class TestPrecisionAtK:
    """Test semantic precision computation."""

    def test_returns_correct_structure(self):
        """Should return dict with expected keys even without embeddings."""
        from eval.metrics import compute_precision_at_k

        queries = [{"query": "chair", "expected": "chair"}]
        result = compute_precision_at_k(queries, "/nonexistent/path.npz")

        assert "precision_at_k" in result
        assert "num_correct" in result
        assert "num_queries" in result
        assert result["num_queries"] == 1
        assert result["precision_at_k"] == 0.0  # no embeddings → 0
