"""
metrics.py — Evaluation metrics for the 3D reconstruction pipeline
===================================================================

Computes three categories of metrics:

1. **Rendering Quality** (PSNR, SSIM)
   - Render the Gaussian splat from held-out camera viewpoints
   - Compare against ground-truth frames
   - Higher is better for both metrics

2. **Geometry Quality** (Chamfer Distance)
   - Compare reconstructed point cloud against reference (if available)
   - Measures 3D accuracy in metres

3. **Semantic Quality** (Precision@K)
   - Given a text query and ground-truth labels, check if the
     top-K CLIP matches include the correct object
   - Evaluates open-vocabulary understanding

Usage:
  python -m eval.metrics \\
      --scene scene1 \\
      --data_dir data/ \\
      --output_dir outputs/ \\
      --save_results eval/results.json
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# 1. Rendering Quality Metrics
# ============================================================
def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two images.

    PSNR = 10 * log10(MAX^2 / MSE)

    Good reconstruction: > 25 dB
    Excellent: > 30 dB

    Args:
        img1: (H, W, 3) uint8 or float image
        img2: (H, W, 3) uint8 or float image (same size as img1)

    Returns:
        PSNR value in dB. Returns float('inf') for identical images.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")

    max_pixel = 255.0
    return float(10.0 * np.log10(max_pixel ** 2 / mse))


def compute_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int = 11,
    C1: float = 6.5025,
    C2: float = 58.5225,
) -> float:
    """
    Compute Structural Similarity Index between two images.

    SSIM measures perceived image quality based on luminance,
    contrast, and structure — better than raw pixel MSE.

    Range: [-1, 1] where 1 = identical images.
    Good reconstruction: > 0.85
    Excellent: > 0.92

    Args:
        img1, img2: (H, W, 3) uint8 images
        window_size: Gaussian window size for local statistics
        C1, C2: stability constants

    Returns:
        Mean SSIM across channels.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Compute per-channel and average
    ssim_channels = []
    for c in range(min(img1.shape[2], 3)):
        ssim_c = _ssim_single_channel(
            img1[:, :, c], img2[:, :, c], window_size, C1, C2
        )
        ssim_channels.append(ssim_c)

    return float(np.mean(ssim_channels))


def _ssim_single_channel(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int,
    C1: float,
    C2: float,
) -> float:
    """Compute SSIM for a single greyscale channel."""
    # Gaussian window
    sigma = 1.5
    gauss = cv2.getGaussianKernel(window_size, sigma)
    window = gauss @ gauss.T

    mu1 = cv2.filter2D(img1, -1, window)
    mu2 = cv2.filter2D(img2, -1, window)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


def evaluate_rendering(
    gt_frames_dir: str,
    rendered_frames_dir: str,
    holdout_fraction: float = 0.1,
) -> Dict[str, float]:
    """
    Evaluate rendering quality on held-out frames.

    Selects a fraction of frames as the test set, loads the
    corresponding rendered images, and computes PSNR and SSIM.

    Args:
        gt_frames_dir: directory of ground-truth frame images
        rendered_frames_dir: directory of rendered frame images
        holdout_fraction: fraction of frames to use as test set

    Returns:
        Dict with 'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std',
        'num_test_frames'
    """
    # Find matching frame pairs
    gt_files = sorted([
        f for f in os.listdir(gt_frames_dir)
        if f.endswith((".png", ".jpg"))
    ])
    rendered_files = set(os.listdir(rendered_frames_dir))

    # Select test frames (evenly spaced)
    n_test = max(1, int(len(gt_files) * holdout_fraction))
    step = max(1, len(gt_files) // n_test)
    test_indices = list(range(0, len(gt_files), step))[:n_test]

    psnr_scores = []
    ssim_scores = []

    for idx in test_indices:
        gt_name = gt_files[idx]

        # Try to find matching rendered frame
        base = os.path.splitext(gt_name)[0]
        rendered_name = None
        for ext in [".png", ".jpg"]:
            candidate = base + ext
            if candidate in rendered_files:
                rendered_name = candidate
                break

        if rendered_name is None:
            continue

        gt_img = cv2.imread(os.path.join(gt_frames_dir, gt_name))
        rendered_img = cv2.imread(os.path.join(rendered_frames_dir, rendered_name))

        if gt_img is None or rendered_img is None:
            continue

        # Resize rendered to match GT if needed
        if gt_img.shape[:2] != rendered_img.shape[:2]:
            rendered_img = cv2.resize(
                rendered_img, (gt_img.shape[1], gt_img.shape[0])
            )

        psnr_scores.append(compute_psnr(gt_img, rendered_img))
        ssim_scores.append(compute_ssim(gt_img, rendered_img))

    if not psnr_scores:
        return {
            "psnr_mean": 0.0,
            "psnr_std": 0.0,
            "ssim_mean": 0.0,
            "ssim_std": 0.0,
            "num_test_frames": 0,
        }

    return {
        "psnr_mean": float(np.mean(psnr_scores)),
        "psnr_std": float(np.std(psnr_scores)),
        "ssim_mean": float(np.mean(ssim_scores)),
        "ssim_std": float(np.std(ssim_scores)),
        "num_test_frames": len(psnr_scores),
    }


# ============================================================
# 2. Geometry Quality Metrics
# ============================================================
def compute_chamfer_distance(
    source: np.ndarray,
    target: np.ndarray,
    max_points: int = 50000,
) -> Dict[str, float]:
    """
    Compute Chamfer distance between two point clouds.

    Chamfer = mean(min_dist(source→target)) + mean(min_dist(target→source))

    Lower is better. Units match the input point cloud units (usually metres).

    Args:
        source: (M, 3) reconstructed point cloud
        target: (N, 3) reference point cloud
        max_points: subsample both clouds to this size

    Returns:
        Dict with 'chamfer', 'source_to_target', 'target_to_source' (in metres)
    """
    from scipy.spatial import cKDTree

    # Subsample for speed
    if len(source) > max_points:
        idx = np.random.choice(len(source), max_points, replace=False)
        source = source[idx]
    if len(target) > max_points:
        idx = np.random.choice(len(target), max_points, replace=False)
        target = target[idx]

    # Source → Target
    tree_target = cKDTree(target)
    dists_s2t, _ = tree_target.query(source)
    s2t = float(np.mean(dists_s2t))

    # Target → Source
    tree_source = cKDTree(source)
    dists_t2s, _ = tree_source.query(target)
    t2s = float(np.mean(dists_t2s))

    chamfer = (s2t + t2s) / 2.0

    return {
        "chamfer": chamfer,
        "source_to_target": s2t,
        "target_to_source": t2s,
        "source_points": len(source),
        "target_points": len(target),
    }


def load_point_cloud(path: str) -> np.ndarray:
    """Load a point cloud from .ply or .npy file."""
    if path.endswith(".npy"):
        return np.load(path).astype(np.float64)

    from plyfile import PlyData
    ply = PlyData.read(path)
    v = ply["vertex"]
    return np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)


# ============================================================
# 3. Semantic Quality Metrics
# ============================================================
def compute_precision_at_k(
    queries: List[Dict[str, str]],
    embeddings_path: str,
    k: int = 10,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Compute Precision@K for open-vocabulary text queries.

    Given a list of (query_text, expected_label) pairs, checks whether
    the CLIP-based retrieval ranks the correct label in the top-K results.

    Args:
        queries: list of dicts with 'query' and 'expected' keys
            e.g. [{"query": "red chair", "expected": "chair"}, ...]
        embeddings_path: path to CLIP embeddings .npz
        k: number of top results to consider
        device: compute device for CLIP

    Returns:
        Dict with 'precision_at_k', 'num_correct', 'num_queries',
        and per-query details.
    """
    if not os.path.exists(embeddings_path):
        return {
            "precision_at_k": 0.0,
            "num_correct": 0,
            "num_queries": len(queries),
            "details": [],
            "error": "embeddings not found",
        }

    try:
        from semantics.clip_embeddings import query_text
    except ImportError:
        return {
            "precision_at_k": 0.0,
            "num_correct": 0,
            "num_queries": len(queries),
            "details": [],
            "error": "CLIP module not available",
        }

    correct = 0
    details = []

    for q in queries:
        query_str = q["query"]
        expected = q["expected"].lower().strip()

        try:
            results = query_text(query_str, embeddings_path, device=device, top_k=k)
            top_labels = [label.lower().strip() for label, _ in results]
            is_correct = expected in top_labels

            if is_correct:
                correct += 1

            rank = top_labels.index(expected) + 1 if expected in top_labels else -1

            details.append({
                "query": query_str,
                "expected": expected,
                "correct": is_correct,
                "rank": rank,
                "top_results": [
                    {"label": label, "score": round(score, 4)}
                    for label, score in results
                ],
            })
        except Exception as e:
            details.append({
                "query": query_str,
                "expected": expected,
                "correct": False,
                "rank": -1,
                "error": str(e),
            })

    precision = correct / max(len(queries), 1)

    return {
        "precision_at_k": round(precision, 4),
        "k": k,
        "num_correct": correct,
        "num_queries": len(queries),
        "details": details,
    }


# ============================================================
# Results Aggregation
# ============================================================
def run_evaluation(
    scene_name: str,
    data_dir: str = "data",
    output_dir: str = "outputs",
    results_path: str = "eval/results.json",
    reference_ply: Optional[str] = None,
    queries: Optional[List[Dict]] = None,
) -> Dict:
    """
    Run all available evaluations for a scene.

    Checks which data is available and runs the appropriate metrics.

    Args:
        scene_name: name of the scene
        data_dir: base data directory
        output_dir: base outputs directory
        results_path: path to save/update results JSON
        reference_ply: optional reference point cloud for Chamfer
        queries: optional list of text queries for Precision@K

    Returns:
        Dict of all computed metrics
    """
    print(f"\n═══ Evaluation: {scene_name} ═══\n")

    scene_data = os.path.join(data_dir, scene_name)
    scene_output = os.path.join(output_dir, scene_name)

    results = {
        "scene": scene_name,
        "timestamp": datetime.now().isoformat(),
    }

    # 1. Rendering quality
    gt_frames = os.path.join(scene_data, "frames")
    rendered_frames = os.path.join(scene_output, "rendered")

    if os.path.exists(gt_frames) and os.path.exists(rendered_frames):
        print("  Computing rendering metrics (PSNR, SSIM)...")
        rendering = evaluate_rendering(gt_frames, rendered_frames)
        results["rendering"] = rendering
        print(f"    PSNR: {rendering['psnr_mean']:.2f} ± {rendering['psnr_std']:.2f} dB")
        print(f"    SSIM: {rendering['ssim_mean']:.4f} ± {rendering['ssim_std']:.4f}")
        print(f"    Test frames: {rendering['num_test_frames']}")
    else:
        print("  ⏭ Rendering metrics skipped (no rendered frames)")
        results["rendering"] = None

    # 2. Geometry quality
    recon_ply = os.path.join(scene_output, "splat.ply")
    slam_ply = os.path.join(scene_data, "slam_logs", "frames.ply")

    if reference_ply and os.path.exists(reference_ply) and os.path.exists(recon_ply):
        print("\n  Computing geometry metrics (Chamfer)...")
        source = load_point_cloud(recon_ply)
        target = load_point_cloud(reference_ply)
        geometry = compute_chamfer_distance(source, target)
        results["geometry"] = geometry
        print(f"    Chamfer: {geometry['chamfer']:.4f} m")
    else:
        print("  ⏭ Geometry metrics skipped (no reference point cloud)")
        results["geometry"] = None

    # 3. Semantic quality
    embeddings_path = os.path.join(scene_output, "embeddings.npz")

    if queries and os.path.exists(embeddings_path):
        print("\n  Computing semantic metrics (Precision@K)...")
        semantic = compute_precision_at_k(queries, embeddings_path)
        results["semantic"] = semantic
        print(f"    Precision@{semantic.get('k', 10)}: "
              f"{semantic['precision_at_k']:.2%} "
              f"({semantic['num_correct']}/{semantic['num_queries']})")
    else:
        print("  ⏭ Semantic metrics skipped (no queries or embeddings)")
        results["semantic"] = None

    # 4. Save results
    _save_results(results_path, scene_name, results)

    print(f"\n═══ Evaluation complete ═══\n")
    return results


def _save_results(results_path: str, scene_name: str, results: Dict) -> None:
    """Save or update the results JSON file."""
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)

    # Load existing results
    existing = {}
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            existing = json.load(f)

    if "scenes" not in existing:
        existing["scenes"] = {}

    existing["scenes"][scene_name] = results
    existing["generated_at"] = datetime.now().isoformat()

    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)

    print(f"  ✓ Results saved to {results_path}")


# ============================================================
# CLI
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate 3D reconstruction and semantic understanding"
    )
    parser.add_argument("--scene", type=str, required=True, help="Scene name")
    parser.add_argument("--data_dir", type=str, default="data", help="Base data dir")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Base output dir")
    parser.add_argument("--save_results", type=str, default="eval/results.json",
                        help="Path to save results JSON")
    parser.add_argument("--reference_ply", type=str, default=None,
                        help="Reference point cloud for Chamfer distance")
    parser.add_argument("--queries_file", type=str, default=None,
                        help="JSON file with text queries for Precision@K")

    args = parser.parse_args()

    queries = None
    if args.queries_file and os.path.exists(args.queries_file):
        with open(args.queries_file, "r") as f:
            queries = json.load(f)

    run_evaluation(
        scene_name=args.scene,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        results_path=args.save_results,
        reference_ply=args.reference_ply,
        queries=queries,
    )


if __name__ == "__main__":
    main()
