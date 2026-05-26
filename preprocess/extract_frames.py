"""
extract_frames.py — Video Preprocessing Pipeline
=================================================

Takes a phone video (.mp4) and produces clean, filtered JPEG frames
ready for MASt3R-SLAM geometry estimation.

Four sequential stages:
  1. Frame Extraction  — ffmpeg pulls frames at target FPS
  2. Exposure Normalisation — even out lighting variations
  3. Blur Detection    — remove motion-blurred frames
  4. Deduplication     — remove near-identical consecutive frames

Usage:
  python -m preprocess.extract_frames --video path/to/video.mp4
  python -m preprocess.extract_frames --video video.mp4 --output_dir data/scene1/frames --fps 3
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# ============================================================
# Stage 1: Frame Extraction via ffmpeg
# ============================================================
def extract_frames_ffmpeg(
    video_path: str,
    output_dir: str,
    fps: int = 3,
    height: int = 518,
) -> List[str]:
    """
    Extract frames from video at a target FPS and resolution.

    Why these defaults:
    - fps=3: a 45-second video gives ~135 frames. Enough overlap for
      MASt3R-SLAM to match features, but not so many it wastes VRAM.
    - height=518: this is the native input resolution for VGGT/MASt3R
      family models. Width is computed automatically to preserve aspect ratio.

    Args:
        video_path: path to the input .mp4 file
        output_dir: directory to save extracted JPEGs
        fps: frames per second to extract
        height: target height in pixels (width auto-scaled)

    Returns:
        List of paths to extracted frame images, sorted by name
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build the ffmpeg command
    # -vf fps=3           → extract 3 frames per second
    # -vf scale=-1:518    → scale height to 518px, width auto (preserves aspect ratio)
    # -q:v 2              → JPEG quality (2 = high quality, small file)
    # %06d.jpg            → output filenames: 000001.jpg, 000002.jpg, ...
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps},scale=-1:{height}",
        "-q:v", "2",
        "-y",  # overwrite existing files without asking
        os.path.join(output_dir, "%06d.jpg"),
    ]

    print(f"    Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    ffmpeg error:\n{result.stderr[-500:]}")
        sys.exit(1)

    # Collect all extracted frames, sorted
    frames = sorted(
        [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".jpg")]
    )
    print(f"    Extracted {len(frames)} frames at {fps} FPS, height={height}px")
    return frames


# ============================================================
# Stage 2: Exposure Normalisation
# ============================================================
def normalise_exposure(frames: List[str]) -> List[str]:
    """
    Normalise brightness across all frames using LAB colour space.

    Problem: as you walk around a room, the phone auto-adjusts exposure.
    One side of the room might be brighter than another. This confuses
    MASt3R-SLAM's feature matching because the same surface looks
    different in brightness across frames.

    Solution: convert each frame to LAB colour space (L = lightness,
    A and B = colour channels). Clip the L channel to the 5th-95th
    percentile range across the whole video, then stretch back to [0, 255].
    This evens out the lighting without destroying colour.

    Args:
        frames: list of frame file paths

    Returns:
        Same list (frames are modified in-place on disk)
    """
    # First pass: compute global L-channel statistics
    all_means = []
    for frame_path in frames:
        img = cv2.imread(frame_path)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        all_means.append(np.mean(lab[:, :, 0]))

    global_mean = np.mean(all_means)
    p5 = np.percentile(all_means, 5)
    p95 = np.percentile(all_means, 95)

    # Second pass: adjust each frame
    adjusted_count = 0
    for i, frame_path in enumerate(frames):
        frame_mean = all_means[i]

        # Only adjust frames outside the 5th-95th percentile range
        if frame_mean < p5 or frame_mean > p95:
            img = cv2.imread(frame_path)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

            # Shift L channel towards the global mean
            l_channel = lab[:, :, 0].astype(np.float32)
            shift = global_mean - frame_mean
            l_channel = np.clip(l_channel + shift, 0, 255).astype(np.uint8)
            lab[:, :, 0] = l_channel

            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            cv2.imwrite(frame_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            adjusted_count += 1

    print(f"    Exposure normalisation: adjusted {adjusted_count}/{len(frames)} frames")
    return frames


# ============================================================
# Stage 3: Blur Detection
# ============================================================
def compute_blur_score(image_path: str) -> float:
    """
    Compute a sharpness score for a single image using the Laplacian variance.

    The Laplacian operator detects edges. Sharp images have many strong edges
    (high variance of the Laplacian). Blurry images have weak, smeared edges
    (low variance).

    A typical sharp indoor frame scores 100-500.
    A motion-blurred frame scores 10-50.

    Args:
        image_path: path to the JPEG frame

    Returns:
        Laplacian variance (higher = sharper)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return float(laplacian.var())


def filter_blurry_frames(frames: List[str], drop_percentile: float = 5.0) -> List[str]:
    """
    Remove the blurriest frames (bottom N percentile by Laplacian variance).

    Why 5%: most phone videos have only a few genuinely blurry frames
    (at the start/end or during fast motion). Dropping 5% is conservative
    enough to remove the worst offenders without losing important coverage.

    Args:
        frames: list of frame file paths
        drop_percentile: bottom N% to remove (default: 5%)

    Returns:
        Filtered list of frame paths (blurry frames are deleted from disk)
    """
    # Compute blur scores for all frames
    scores = [(f, compute_blur_score(f)) for f in frames]

    # Find the threshold
    all_scores = [s for _, s in scores]
    threshold = np.percentile(all_scores, drop_percentile)

    # Keep frames above threshold
    kept = []
    removed = 0
    for frame_path, score in scores:
        if score >= threshold:
            kept.append(frame_path)
        else:
            os.remove(frame_path)
            removed += 1

    print(f"    Blur filter: removed {removed} frames (threshold={threshold:.1f})")
    return kept


# ============================================================
# Stage 4: Deduplication
# ============================================================
def deduplicate_frames(frames: List[str], ssim_threshold: float = 0.98) -> List[str]:
    """
    Remove near-identical consecutive frames using SSIM.

    Problem: if you hold the phone still for a moment (e.g. looking at a
    corner), you get many frames that are almost identical. These waste
    MASt3R-SLAM's compute budget without adding new information.

    Solution: compare each frame to the previous one using SSIM
    (Structural Similarity Index). If SSIM > 0.98 (i.e., 98% identical),
    drop the duplicate. Always keep the first and last frames.

    Args:
        frames: list of frame file paths (must be sorted chronologically)
        ssim_threshold: drop frame if SSIM with previous > this value

    Returns:
        Deduplicated list of frame paths (duplicates deleted from disk)
    """
    if len(frames) <= 2:
        return frames

    kept = [frames[0]]  # Always keep first frame
    prev_gray = cv2.imread(frames[0], cv2.IMREAD_GRAYSCALE)

    removed = 0
    for i in range(1, len(frames) - 1):
        curr_gray = cv2.imread(frames[i], cv2.IMREAD_GRAYSCALE)

        # Compute SSIM between consecutive frames
        # SSIM returns a value between -1 and 1; 1 = identical
        score = ssim(prev_gray, curr_gray)

        if score < ssim_threshold:
            # Frames are different enough — keep this one
            kept.append(frames[i])
            prev_gray = curr_gray
        else:
            # Too similar to previous kept frame — discard
            os.remove(frames[i])
            removed += 1

    # Always keep last frame
    kept.append(frames[-1])

    print(f"    Deduplication: removed {removed} frames (SSIM threshold={ssim_threshold})")
    return kept


# ============================================================
# Manifest Generation
# ============================================================
def write_manifest(frames: List[str], output_dir: str, video_path: str) -> str:
    """
    Write a frames.json manifest consumed by downstream stages.

    The manifest records:
    - Which frames survived filtering
    - Their filenames and order
    - The original video they came from
    - When preprocessing was run

    This is consumed by MASt3R-SLAM (to know which images to process)
    and by the semantic pipeline (to map masks back to frames).

    Args:
        frames: final list of frame paths after all filtering
        output_dir: directory containing the frames
        video_path: path to the original video

    Returns:
        Path to the generated manifest file
    """
    manifest = {
        "video_source": os.path.abspath(video_path),
        "frame_count": len(frames),
        "processed_at": datetime.now().isoformat(),
        "frames": [
            {
                "index": i,
                "filename": os.path.basename(f),
                "path": os.path.abspath(f),
            }
            for i, f in enumerate(frames)
        ],
    }

    manifest_path = os.path.join(output_dir, "frames.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"    Manifest written to {manifest_path}")
    return manifest_path


# ============================================================
# Full Pipeline
# ============================================================
def run_preprocessing(
    video_path: str,
    output_dir: str,
    fps: int = 3,
    height: int = 518,
    blur_drop_pct: float = 5.0,
    ssim_threshold: float = 0.98,
) -> Tuple[List[str], str]:
    """
    Run the complete 4-stage preprocessing pipeline.

    Args:
        video_path: path to input .mp4 video
        output_dir: where to save processed frames
        fps: extraction rate (frames per second)
        height: target frame height in pixels
        blur_drop_pct: percentage of blurriest frames to drop
        ssim_threshold: SSIM above this = duplicate (dropped)

    Returns:
        Tuple of (list of final frame paths, manifest path)
    """
    print(f"\n  Stage 1/4: Extracting frames...")
    frames = extract_frames_ffmpeg(video_path, output_dir, fps, height)

    print(f"\n  Stage 2/4: Normalising exposure...")
    frames = normalise_exposure(frames)

    print(f"\n  Stage 3/4: Filtering blurry frames...")
    frames = filter_blurry_frames(frames, blur_drop_pct)

    print(f"\n  Stage 4/4: Removing duplicates...")
    frames = deduplicate_frames(frames, ssim_threshold)

    # Rename surviving frames sequentially (000001.jpg, 000002.jpg, ...)
    # so MASt3R-SLAM gets a clean, gapless sequence
    print(f"\n  Renaming {len(frames)} frames sequentially...")
    renamed = []
    for i, old_path in enumerate(frames):
        new_name = f"{i + 1:06d}.jpg"
        new_path = os.path.join(output_dir, new_name)
        if old_path != new_path:
            os.rename(old_path, new_path)
        renamed.append(new_path)
    frames = renamed

    # Write manifest
    manifest_path = write_manifest(frames, output_dir, video_path)

    print(f"\n  ✓ Preprocessing complete: {len(frames)} frames ready")
    return frames, manifest_path


# ============================================================
# CLI Entry Point
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and filter frames from a video for 3D reconstruction"
    )
    parser.add_argument(
        "--video", type=str, required=True, help="Path to input video (.mp4)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for frames (default: data/<video_name>/frames/)",
    )
    parser.add_argument(
        "--fps", type=int, default=3, help="Frames per second to extract (default: 3)"
    )
    parser.add_argument(
        "--height", type=int, default=518, help="Target frame height in pixels (default: 518)"
    )
    parser.add_argument(
        "--blur_drop_pct",
        type=float,
        default=5.0,
        help="Percentage of blurriest frames to drop (default: 5.0)",
    )
    parser.add_argument(
        "--ssim_threshold",
        type=float,
        default=0.98,
        help="SSIM threshold for deduplication (default: 0.98)",
    )

    args = parser.parse_args()

    # Default output directory: data/<video_name>/frames/
    if args.output_dir is None:
        video_name = Path(args.video).stem
        args.output_dir = os.path.join("data", video_name, "frames")

    # Validate input
    if not os.path.isfile(args.video):
        print(f"Error: video not found: {args.video}")
        sys.exit(1)

    run_preprocessing(
        video_path=args.video,
        output_dir=args.output_dir,
        fps=args.fps,
        height=args.height,
        blur_drop_pct=args.blur_drop_pct,
        ssim_threshold=args.ssim_threshold,
    )


if __name__ == "__main__":
    main()
