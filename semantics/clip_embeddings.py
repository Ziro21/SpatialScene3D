"""
clip_embeddings.py — Compute CLIP embeddings per semantic instance
==================================================================

For each labelled instance in the scene:
  1. Find all frames where that instance is visible
  2. Crop the image region covered by the instance mask
  3. Run CLIP ViT-L/14 on each crop
  4. Average the embeddings across all views → one embedding per instance

These embeddings enable open-vocabulary text queries at inference time:
"find the red chair" → CLIP(text) · CLIP(instance) → cosine similarity.

This runs on your Mac (MPS or CPU). CLIP ViT-L/14 needs ~1.5 GB memory.

Usage:
  python -m semantics.clip_embeddings \\
      --masks data/scene1/masks/ \\
      --frames data/scene1/colmap/images/ \\
      --output outputs/scene1/embeddings.npz
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def get_device() -> str:
    """Select best available device: MPS (Mac), CUDA, or CPU."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
    except (ImportError, AttributeError):
        pass
    return "cpu"


def load_clip_model(device: str = "cpu"):
    """
    Load CLIP ViT-L/14 model and preprocessing transform.

    Uses OpenAI's official CLIP implementation.
    Falls back to a smaller model if ViT-L/14 doesn't fit in memory.

    Args:
        device: "mps", "cuda", or "cpu"

    Returns:
        Tuple of (model, preprocess_fn, tokenize_fn)
    """
    try:
        import clip
        import torch

        # Try ViT-L/14 first (best quality)
        try:
            model, preprocess = clip.load("ViT-L/14", device=device)
            print(f"  ✓ CLIP ViT-L/14 loaded on {device}")
            return model, preprocess, clip.tokenize
        except RuntimeError:
            # Fall back to ViT-B/32 (smaller, fits in 8GB)
            model, preprocess = clip.load("ViT-B/32", device=device)
            print(f"  ✓ CLIP ViT-B/32 loaded on {device} (ViT-L/14 too large)")
            return model, preprocess, clip.tokenize

    except ImportError:
        raise ImportError(
            "OpenAI CLIP is required: pip install git+https://github.com/openai/CLIP.git"
        )


def crop_instance_from_frame(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    padding: int = 10,
    min_size: int = 32,
) -> Optional[np.ndarray]:
    """
    Crop the image region covered by a mask, with padding.

    Args:
        image_rgb: (H, W, 3) RGB image
        mask: (H, W) binary mask (255 = inside)
        padding: pixels of padding around the bounding box
        min_size: minimum crop size (skip tiny instances)

    Returns:
        Cropped RGB image, or None if the mask is too small
    """
    # Find bounding box of the mask
    coords = np.where(mask > 127)
    if len(coords[0]) == 0:
        return None

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    # Check minimum size
    if (y_max - y_min) < min_size or (x_max - x_min) < min_size:
        return None

    # Add padding
    h, w = image_rgb.shape[:2]
    y_min = max(0, y_min - padding)
    y_max = min(h, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(w, x_max + padding)

    # Crop
    crop = image_rgb[y_min:y_max, x_min:x_max].copy()

    # Optionally mask out the background (set to mean colour)
    crop_mask = mask[y_min:y_max, x_min:x_max]
    bg = crop_mask <= 127
    if np.any(bg):
        mean_colour = crop[~bg].mean(axis=0).astype(np.uint8)
        crop[bg] = mean_colour

    return crop


def compute_instance_embeddings(
    masks_dir: str,
    frames_dir: str,
    output_path: str,
    device: Optional[str] = None,
    max_crops_per_instance: int = 20,
) -> str:
    """
    Compute CLIP embeddings for each semantic instance.

    Steps:
    1. Load the masks manifest
    2. Group masks by label (all views of "chair" across frames)
    3. Crop each instance from each frame
    4. Run CLIP on each crop
    5. Average embeddings across views → one per instance
    6. Save as .npz file

    Args:
        masks_dir: path to masks/ directory
        frames_dir: path to frame images directory
        output_path: where to save the embeddings .npz
        device: compute device (auto-detected if None)
        max_crops_per_instance: max crops to average per instance

    Returns:
        Path to the output .npz file
    """
    import torch

    print("\n═══ CLIP Embedding Computation ═══\n")

    if device is None:
        device = get_device()

    # 1. Load CLIP
    print("  Loading CLIP model...")
    model, preprocess, tokenize = load_clip_model(device)

    # 2. Load mask manifest
    print("\n  Loading mask manifest...")
    manifest_path = os.path.join(masks_dir, "masks.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Group masks by label string
    # label_crops[label_string] = list of (frame_path, mask_path) tuples
    label_crops: Dict[str, List[Tuple[str, str]]] = {}

    for frame_info in manifest:
        frame_name = frame_info["frame"]
        frame_path = os.path.join(frames_dir, frame_name)

        if not os.path.exists(frame_path):
            # Try alternative extensions
            base = os.path.splitext(frame_name)[0]
            for ext in [".png", ".jpg"]:
                alt = os.path.join(frames_dir, base + ext)
                if os.path.exists(alt):
                    frame_path = alt
                    break

        for mask_info in frame_info.get("masks", []):
            label = mask_info["label"].lower().strip()
            mask_path = os.path.join(masks_dir, mask_info["mask_file"])

            if os.path.exists(frame_path) and os.path.exists(mask_path):
                if label not in label_crops:
                    label_crops[label] = []
                label_crops[label].append((frame_path, mask_path))

    print(f"  Found {len(label_crops)} unique labels:")
    for label, crops in sorted(label_crops.items()):
        print(f"    {label}: {len(crops)} mask crops available")

    # 3. Compute embeddings
    print("\n  Computing CLIP embeddings...")
    embeddings = {}
    label_to_id = {}
    label_id = 1

    from PIL import Image as PILImage

    for label, crop_pairs in sorted(label_crops.items()):
        label_to_id[label] = label_id

        # Subsample if too many crops
        if len(crop_pairs) > max_crops_per_instance:
            indices = np.linspace(0, len(crop_pairs) - 1,
                                  max_crops_per_instance, dtype=int)
            crop_pairs = [crop_pairs[i] for i in indices]

        all_features = []

        for frame_path, mask_path in crop_pairs:
            # Load image and mask
            image = cv2.imread(frame_path)
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            # Crop instance
            crop = crop_instance_from_frame(image_rgb, mask)
            if crop is None:
                continue

            # Preprocess for CLIP
            pil_crop = PILImage.fromarray(crop)
            clip_input = preprocess(pil_crop).unsqueeze(0).to(device)

            # Extract features
            with torch.no_grad():
                features = model.encode_image(clip_input)
                features = features / features.norm(dim=-1, keepdim=True)
                all_features.append(features.cpu().numpy().flatten())

        if all_features:
            # Average across all views
            avg_embedding = np.mean(all_features, axis=0)
            # Re-normalise to unit length
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            embeddings[label] = avg_embedding
            print(f"    {label}: averaged {len(all_features)} crops → "
                  f"dim={len(avg_embedding)}")
        else:
            print(f"    {label}: no valid crops found ⚠")

        label_id += 1

    # 4. Save
    print(f"\n  Saving embeddings...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Save as .npz with:
    #   - One array per label: embeddings[label_name] = (D,) float32
    #   - label_names: ordered list of label strings
    #   - label_ids: corresponding integer IDs
    save_dict = {}
    label_names_list = []
    label_ids_list = []

    for label, embedding in sorted(embeddings.items()):
        safe_key = label.replace(" ", "_").replace(".", "")
        save_dict[f"emb_{safe_key}"] = embedding.astype(np.float32)
        label_names_list.append(label)
        label_ids_list.append(label_to_id[label])

    save_dict["label_names"] = np.array(label_names_list, dtype=object)
    save_dict["label_ids"] = np.array(label_ids_list, dtype=np.int32)

    np.savez(output_path, **save_dict)

    # Also save a human-readable JSON sidecar
    json_path = output_path.replace(".npz", ".json")
    meta = {
        "num_instances": len(embeddings),
        "embedding_dim": len(next(iter(embeddings.values()))) if embeddings else 0,
        "device_used": device,
        "labels": {
            label: {
                "id": label_to_id[label],
                "num_crops": len(label_crops[label]),
            }
            for label in embeddings
        },
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  ✓ Saved embeddings: {output_path}")
    print(f"  ✓ Saved metadata: {json_path}")
    print(f"\n═══ CLIP embedding computation complete ═══")

    return output_path


def query_text(
    text: str,
    embeddings_path: str,
    device: Optional[str] = None,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Query the stored embeddings with a text prompt.

    Computes CLIP(text) · CLIP(instance) cosine similarity for each instance.

    Args:
        text: natural language query (e.g. "red chair")
        embeddings_path: path to .npz embeddings file
        device: compute device
        top_k: number of top matches to return

    Returns:
        List of (label_name, similarity_score) tuples, sorted by similarity
    """
    import torch

    if device is None:
        device = get_device()

    model, _, tokenize = load_clip_model(device)

    # Encode text
    with torch.no_grad():
        text_tokens = tokenize([text]).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_vec = text_features.cpu().numpy().flatten()

    # Load instance embeddings
    data = np.load(embeddings_path, allow_pickle=True)
    label_names = data["label_names"]

    results = []
    for i, label in enumerate(label_names):
        safe_key = f"emb_{str(label).replace(' ', '_').replace('.', '')}"
        if safe_key in data:
            emb = data[safe_key]
            sim = float(np.dot(text_vec, emb))
            results.append((str(label), sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ============================================================
# CLI
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute CLIP ViT-L/14 embeddings per semantic instance"
    )
    parser.add_argument(
        "--masks", type=str, required=True,
        help="Path to masks/ directory (must contain masks.json)",
    )
    parser.add_argument(
        "--frames", type=str, required=True,
        help="Path to frame images directory",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path for output embeddings .npz file",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'mps', 'cuda', or 'cpu' (auto-detected if omitted)",
    )

    args = parser.parse_args()
    compute_instance_embeddings(
        masks_dir=args.masks,
        frames_dir=args.frames,
        output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
