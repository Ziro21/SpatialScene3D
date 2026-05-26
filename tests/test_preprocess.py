"""
test_preprocess.py — Smoke tests for the preprocessing pipeline.

Tests each stage individually and the full pipeline end-to-end.
Uses a synthetically generated test video (no real phone needed).
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


# ============================================================
# Fixture: create a synthetic test video
# ============================================================
@pytest.fixture
def test_video(tmp_path: Path) -> str:
    """
    Generate a short synthetic video for testing.

    Creates a 3-second, 30fps video with:
    - Frames that slowly change colour (to test deduplication)
    - One intentionally blurry frame (to test blur detection)
    - Varying brightness (to test exposure normalisation)
    """
    video_path = str(tmp_path / "test_video.mp4")
    frame_dir = tmp_path / "raw_frames"
    frame_dir.mkdir()

    width, height = 640, 480
    num_frames = 90  # 3 seconds at 30fps

    for i in range(num_frames):
        # Create a frame with slowly changing colour
        hue = int((i / num_frames) * 180)  # 0-180 in OpenCV HSV
        frame = np.full((height, width, 3), (hue, 200, 200), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

        # Add some texture (a grid) so blur detection has edges to measure
        for x in range(0, width, 40):
            cv2.line(frame, (x, 0), (x, height), (255, 255, 255), 1)
        for y in range(0, height, 40):
            cv2.line(frame, (0, y), (width, y), (255, 255, 255), 1)

        # Make frame 45 intentionally blurry
        if i == 45:
            frame = cv2.GaussianBlur(frame, (31, 31), 15)

        # Vary brightness for exposure test
        brightness_shift = int(30 * np.sin(2 * np.pi * i / num_frames))
        frame = np.clip(frame.astype(np.int16) + brightness_shift, 0, 255).astype(np.uint8)

        cv2.imwrite(str(frame_dir / f"{i:06d}.jpg"), frame)

    # Use ffmpeg to create video from frames
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate", "30",
            "-i", str(frame_dir / "%06d.jpg"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            video_path,
        ],
        capture_output=True,
        check=True,
    )

    return video_path


# ============================================================
# Tests
# ============================================================
class TestFrameExtraction:
    """Test Stage 1: ffmpeg frame extraction."""

    def test_extracts_correct_number_of_frames(self, test_video: str, tmp_path: Path) -> None:
        """At 3 FPS from a 3-second video, we expect ~9 frames."""
        from preprocess.extract_frames import extract_frames_ffmpeg

        output_dir = str(tmp_path / "frames")
        frames = extract_frames_ffmpeg(test_video, output_dir, fps=3, height=480)

        # 3-second video at 3 FPS should give approximately 9 frames
        assert 7 <= len(frames) <= 11, f"Expected ~9 frames, got {len(frames)}"

    def test_frames_are_valid_images(self, test_video: str, tmp_path: Path) -> None:
        """All extracted frames should be readable by OpenCV."""
        from preprocess.extract_frames import extract_frames_ffmpeg

        output_dir = str(tmp_path / "frames")
        frames = extract_frames_ffmpeg(test_video, output_dir, fps=3, height=480)

        for frame_path in frames:
            img = cv2.imread(frame_path)
            assert img is not None, f"Failed to read {frame_path}"
            assert img.shape[0] > 0 and img.shape[1] > 0


class TestBlurDetection:
    """Test Stage 3: Laplacian blur detection."""

    def test_sharp_image_scores_higher_than_blurry(self, tmp_path: Path) -> None:
        """A sharp image should have a higher Laplacian variance than a blurry one."""
        from preprocess.extract_frames import compute_blur_score

        # Create a sharp image (grid pattern)
        sharp = np.zeros((200, 200), dtype=np.uint8)
        for x in range(0, 200, 20):
            cv2.line(sharp, (x, 0), (x, 200), 255, 1)
        sharp_path = str(tmp_path / "sharp.jpg")
        cv2.imwrite(sharp_path, sharp)

        # Create a blurry image (same grid, heavily blurred)
        blurry = cv2.GaussianBlur(sharp, (21, 21), 10)
        blurry_path = str(tmp_path / "blurry.jpg")
        cv2.imwrite(blurry_path, blurry)

        sharp_score = compute_blur_score(sharp_path)
        blurry_score = compute_blur_score(blurry_path)

        assert sharp_score > blurry_score * 2, (
            f"Sharp ({sharp_score:.1f}) should be much higher than blurry ({blurry_score:.1f})"
        )


class TestFullPipeline:
    """Test the complete 4-stage pipeline end-to-end."""

    def test_pipeline_produces_valid_output(self, test_video: str, tmp_path: Path) -> None:
        """Full pipeline should produce frames + manifest."""
        from preprocess.extract_frames import run_preprocessing

        output_dir = str(tmp_path / "pipeline_output")
        frames, manifest_path = run_preprocessing(
            video_path=test_video,
            output_dir=output_dir,
            fps=3,
            height=480,
        )

        # Should have some frames
        assert len(frames) > 0, "Pipeline produced no frames"

        # Frames should be in expected range for a 3-second video
        assert len(frames) <= 15, f"Too many frames: {len(frames)}"

        # Manifest should exist and be valid JSON
        assert os.path.isfile(manifest_path)
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["frame_count"] == len(frames)
        assert len(manifest["frames"]) == len(frames)

        # All listed frames should actually exist on disk
        for frame_info in manifest["frames"]:
            assert os.path.isfile(frame_info["path"]), f"Missing: {frame_info['path']}"

    def test_output_frames_are_sequential(self, test_video: str, tmp_path: Path) -> None:
        """Output frames should be numbered 000001.jpg, 000002.jpg, etc."""
        from preprocess.extract_frames import run_preprocessing

        output_dir = str(tmp_path / "seq_output")
        frames, _ = run_preprocessing(
            video_path=test_video,
            output_dir=output_dir,
            fps=3,
            height=480,
        )

        for i, frame_path in enumerate(frames):
            expected_name = f"{i + 1:06d}.jpg"
            actual_name = os.path.basename(frame_path)
            assert actual_name == expected_name, (
                f"Frame {i}: expected {expected_name}, got {actual_name}"
            )
