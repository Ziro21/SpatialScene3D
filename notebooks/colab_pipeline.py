# %% [markdown]
# # scene3d — Colab Pipeline: MASt3R-SLAM + gsplat + Grounded-SAM-2
#
# This notebook runs all GPU-dependent stages on Google Colab.
# Your Mac handles preprocessing and the viewer; Colab handles the heavy lifting.
#
# ## Workflow
# 1. Upload preprocessed frames to Google Drive
# 2. Run this notebook top-to-bottom
# 3. Download results (COLMAP workspace, .ply, masks) to your Mac
#
# **Runtime**: Select GPU → Runtime → Change runtime type → T4 GPU (free) or A100 (Pro)

# %% [markdown]
# ---
# ## 0. Mount Google Drive & Configure Scene


import os
import shutil

# ============================================================
# CONFIGURE THESE — change for each scene
# ============================================================
SCENE_NAME = "scene1"  # change this per scene

# Where your frames live on Google Drive (upload from Mac first)
DRIVE_FRAMES = f"/content/drive/MyDrive/scene3d/{SCENE_NAME}/frames"

# Working directories on Colab (local SSD = much faster I/O than Drive)
WORK_DIR = f"/content/scene3d_work/{SCENE_NAME}"
FRAMES_DIR = f"{WORK_DIR}/frames"
SLAM_DIR = "/content/MASt3R-SLAM"
SLAM_LOGS = f"{SLAM_DIR}/logs"
COLMAP_DIR = f"{WORK_DIR}/colmap"
GSPLAT_OUTPUT = f"{WORK_DIR}/gsplat_output"
MASKS_DIR = f"{WORK_DIR}/masks"

# Where results go back on Google Drive (download to Mac from here)
DRIVE_OUTPUT = f"/content/drive/MyDrive/scene3d/{SCENE_NAME}/outputs"

# %%
# Copy frames from Drive → Colab local SSD (much faster for GPU reads)
os.makedirs(WORK_DIR, exist_ok=True)

if os.path.exists(FRAMES_DIR):
    shutil.rmtree(FRAMES_DIR)

shutil.copytree(DRIVE_FRAMES, FRAMES_DIR)

num_frames = len([f for f in os.listdir(FRAMES_DIR) if f.endswith(('.jpg', '.png'))])
print(f"✓ Copied {num_frames} frames to {FRAMES_DIR}")

# %%
# ============================================================
# CRITICAL: Convert .jpg frames to .png
# ============================================================
# MASt3R-SLAM's RGBFiles dataset class ONLY loads *.png files.
# (Source: dataloader.py line: self.rgb_files = natsorted(list(self.dataset_path.glob("*.png"))))
# Our preprocessor outputs .jpg, so we must convert.
# ============================================================
import cv2
import glob

jpg_files = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
if jpg_files:
    print(f"Converting {len(jpg_files)} .jpg frames to .png for MASt3R-SLAM...")
    for jpg_path in jpg_files:
        png_path = jpg_path.replace('.jpg', '.png')
        img = cv2.imread(jpg_path)
        cv2.imwrite(png_path, img)
        os.remove(jpg_path)  # remove .jpg to avoid confusion
    print(f"✓ Converted {len(jpg_files)} frames to .png")
else:
    png_count = len(glob.glob(os.path.join(FRAMES_DIR, "*.png")))
    print(f"✓ Found {png_count} .png frames (no conversion needed)")

# %% [markdown]
# ---
# ## 1. Install MASt3R-SLAM
#
# MASt3R-SLAM has 3 submodule dependencies that must be installed in order.
# This takes ~5-8 minutes on first run. Restart runtime if you hit errors.

# %%
# Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"CUDA: {torch.version.cuda}")
print(f"PyTorch: {torch.__version__}")

# %%
# Clone MASt3R-SLAM (with submodules)
if not os.path.exists(SLAM_DIR):
    !git clone --recursive https://github.com/rmurai0610/MASt3R-SLAM.git {SLAM_DIR}
else:
    print("✓ MASt3R-SLAM already cloned")

# %%
# Install dependencies IN ORDER (order matters!)

import os
# Set environment for T4 GPU and force CUDA
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
os.environ["FORCE_CUDA"] = "1"
os.environ["MAX_JOBS"] = "4"  # Prevent Colab RAM crash during compile

# Step 0: Fix Python 3.12 build tools (distutils removal)
!pip install setuptools==69.5.1 wheel ninja

# Step 1: Pre-install lietorch manually (otherwise fails on Python 3.12)
!pip install --no-build-isolation git+https://github.com/princeton-vl/lietorch.git

# Step 2: Build curope manually
!cd {SLAM_DIR}/thirdparty/mast3r/dust3r/croco/models/curope && pip install .

# Step 3: MASt3R (the 3D matching backbone)
!pip install --no-build-isolation -e {SLAM_DIR}/thirdparty/mast3r

# Step 4: in3d (internal 3D utilities)
!pip install --no-build-isolation -e {SLAM_DIR}/thirdparty/in3d

# Step 5: MASt3R-SLAM itself (builds custom CUDA kernels)
!cd {SLAM_DIR} && rm -rf build && pip install --no-build-isolation -e .

# Step 6: extras
!pip install plyfile natsort

print("\n✓ MASt3R-SLAM installed")

# %%
# Download model checkpoints
CKPT_DIR = f"{SLAM_DIR}/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

# MASt3R backbone checkpoint (~1.3 GB)
MASTR_CKPT = f"{CKPT_DIR}/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
if not os.path.exists(MASTR_CKPT):
    print("Downloading MASt3R checkpoint (~1.3 GB)...")
    !wget -q --show-progress -O {MASTR_CKPT} \
        "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
else:
    print("✓ MASt3R checkpoint exists")

# Retrieval model (needed for loop closure / relocalization)
RETRIEVAL_CKPT = f"{CKPT_DIR}/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
if not os.path.exists(RETRIEVAL_CKPT):
    print("Downloading retrieval checkpoint...")
    !wget -q --show-progress -O {RETRIEVAL_CKPT} \
        "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
else:
    print("✓ Retrieval checkpoint exists")

ckpt_size = os.path.getsize(MASTR_CKPT) / 1e6
print(f"  MASt3R checkpoint: {ckpt_size:.0f} MB")

# %% [markdown]
# ---
# ## 2. Run MASt3R-SLAM
#
# ### Verified CLI (from reading main.py source code):
# ```
# python main.py --dataset PATH --config config/base.yaml --no-viz [--save-as NAME] [--calib FILE]
# ```
#
# **Exact flags** (from `main.py` lines 153-158):
# - `--dataset` : path to image folder (*.png) or .mp4 video
# - `--config`  : YAML config (default: `config/base.yaml`)
# - `--save-as` : subdirectory under `logs/` for outputs (default: `"default"`)
# - `--no-viz`  : headless mode (no GUI)
# - `--calib`   : optional calibration YAML
#
# **There is NO `--save_ply` flag!** Saving happens automatically.
#
# ### Output (from evaluate.py):
# ```
# logs/
# ├── {folder_name}.txt        ← TUM trajectory (timestamp x y z qx qy qz qw)
# ├── {folder_name}.ply        ← Dense point cloud (binary PLY)
# └── keyframes/{folder_name}/ ← Keyframe images as PNG
# ```
#
# ### Dataset requirement:
# `RGBFiles` class loads only `*.png` files via `glob("*.png")`.
# That's why we converted .jpg→.png in Step 0.

# %%
# Clean previous logs
if os.path.exists(SLAM_LOGS):
    shutil.rmtree(SLAM_LOGS)

%cd {SLAM_DIR}

# Run MASt3R-SLAM
# --dataset: our frames folder (containing .png files)
# --config: base configuration
# --no-viz: headless (no GUI on Colab)
# --save-as: where to put outputs under logs/
!python main.py \
    --dataset {FRAMES_DIR} \
    --config config/base.yaml \
    --no-viz \
    --save-as default

print("\n✓ MASt3R-SLAM finished")

# %%
# Inspect MASt3R-SLAM output
# The folder name becomes the "seq_name": dataset_path.stem
import pathlib

seq_name = pathlib.Path(FRAMES_DIR).stem  # "frames"
expected_traj = os.path.join(SLAM_LOGS, f"{seq_name}.txt")
expected_ply = os.path.join(SLAM_LOGS, f"{seq_name}.ply")
expected_kf = os.path.join(SLAM_LOGS, "keyframes", seq_name)

print(f"Expected seq_name: '{seq_name}'")
print(f"Trajectory file: {expected_traj} → exists: {os.path.exists(expected_traj)}")
print(f"Point cloud:     {expected_ply} → exists: {os.path.exists(expected_ply)}")
print(f"Keyframes dir:   {expected_kf} → exists: {os.path.exists(expected_kf)}")

print("\nAll files in logs/:")
if os.path.exists(SLAM_LOGS):
    for root, dirs, files in os.walk(SLAM_LOGS):
        for f in sorted(files):
            path = os.path.join(root, f)
            size_mb = os.path.getsize(path) / 1e6
            rel = os.path.relpath(path, SLAM_LOGS)
            print(f"  {rel}: {size_mb:.2f} MB")
else:
    print("  ⚠ No logs/ directory! Search for output files:")
    !find {SLAM_DIR} -name "*.ply" -o -name "*.txt" 2>/dev/null | head -20

# %%
# Load the trajectory (TUM format: timestamp x y z qx qy qz qw)
# These are CAMERA-TO-WORLD transforms (SE3)
import numpy as np

SLAM_TRAJ = None
SLAM_PLY = None

# Find the trajectory file
traj_candidates = [
    expected_traj,
    *glob.glob(os.path.join(SLAM_LOGS, "**/*.txt"), recursive=True),
]
for traj_path in traj_candidates:
    if os.path.exists(traj_path) and os.path.getsize(traj_path) > 50:
        try:
            traj_data = np.loadtxt(traj_path)
            if traj_data.ndim == 2 and traj_data.shape[1] == 8:
                SLAM_TRAJ = traj_data
                print(f"✓ Loaded trajectory: {traj_path}")
                print(f"  {len(SLAM_TRAJ)} keyframe poses")
                print(f"  Format: [timestamp, x, y, z, qx, qy, qz, qw]")
                break
        except Exception as e:
            continue

if SLAM_TRAJ is None:
    print("⚠ No trajectory file found — will use identity poses")

# Find the .ply file
ply_candidates = [
    expected_ply,
    *glob.glob(os.path.join(SLAM_DIR, "**/*.ply"), recursive=True),
]
for ply_path in ply_candidates:
    if os.path.exists(ply_path) and os.path.getsize(ply_path) > 1000:
        SLAM_PLY = ply_path
        size_mb = os.path.getsize(ply_path) / 1e6
        print(f"✓ Found point cloud: {ply_path} ({size_mb:.1f} MB)")
        break

if SLAM_PLY is None:
    print("⚠ No .ply found — COLMAP will have empty point cloud")

# %% [markdown]
# ---
# ## 3. Convert to COLMAP Format for gsplat
#
# MASt3R-SLAM outputs:
# - Trajectory: TUM format (timestamp x y z qx qy qz qw) — **camera-to-world**
# - Point cloud: binary PLY (x, y, z, red, green, blue)
#
# gsplat needs COLMAP format. We convert:
# - Trajectory → `images.bin` (inverting camera-to-world → world-to-camera)
# - Point cloud → `points3D.bin`
# - Camera params → `cameras.bin`

# %%
import struct
from plyfile import PlyData

os.makedirs(f"{COLMAP_DIR}/sparse/0", exist_ok=True)

# Copy images to COLMAP workspace (gsplat expects images/ directory)
colmap_images = f"{COLMAP_DIR}/images"
if os.path.exists(colmap_images):
    shutil.rmtree(colmap_images)
shutil.copytree(FRAMES_DIR, colmap_images)

# Get sorted frame names
frame_names = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith(('.png', '.jpg'))])
first_img = cv2.imread(os.path.join(FRAMES_DIR, frame_names[0]))
img_h, img_w = first_img.shape[:2]
print(f"Image size: {img_w} x {img_h}")
print(f"Frames: {len(frame_names)}")

# Estimate focal length (standard approximation)
focal = float(max(img_w, img_h))
cx = img_w / 2.0
cy = img_h / 2.0

# --- Write cameras.bin ---
cameras_path = f"{COLMAP_DIR}/sparse/0/cameras.bin"
with open(cameras_path, 'wb') as f:
    f.write(struct.pack('<Q', 1))  # 1 camera
    f.write(struct.pack('<i', 1))  # camera_id = 1
    f.write(struct.pack('<i', 1))  # model_id = 1 (PINHOLE)
    f.write(struct.pack('<Q', img_w))
    f.write(struct.pack('<Q', img_h))
    for p in [focal, focal, cx, cy]:
        f.write(struct.pack('<d', p))
print(f"✓ cameras.bin: PINHOLE focal={focal:.0f}, center=({cx:.0f}, {cy:.0f})")

# %%
# --- Write images.bin ---
# Convert MASt3R-SLAM trajectory (camera-to-world) to COLMAP (world-to-camera)

def quat_to_rot(qw, qx, qy, qz):
    """Quaternion (w,x,y,z) → 3x3 rotation matrix."""
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ])

def rot_to_quat(R):
    """3x3 rotation matrix → quaternion (w,x,y,z)."""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return w, x, y, z

images_path = f"{COLMAP_DIR}/sparse/0/images.bin"
with open(images_path, 'wb') as f:
    f.write(struct.pack('<Q', len(frame_names)))
    
    for i, fname in enumerate(frame_names):
        image_id = i + 1
        
        if SLAM_TRAJ is not None and i < len(SLAM_TRAJ):
            # MASt3R-SLAM format: timestamp x y z qx qy qz qw (camera-to-world)
            _, tx, ty, tz, qx, qy, qz, qw = SLAM_TRAJ[i]
            
            # Build camera-to-world 4x4 matrix
            R_c2w = quat_to_rot(qw, qx, qy, qz)
            t_c2w = np.array([tx, ty, tz])
            
            # Invert to world-to-camera (what COLMAP expects)
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w
            
            # Convert back to quaternion
            cqw, cqx, cqy, cqz = rot_to_quat(R_w2c)
            ctx, cty, ctz = t_w2c
        else:
            # Fallback: identity rotation, spaced translation
            cqw, cqx, cqy, cqz = 1.0, 0.0, 0.0, 0.0
            ctx, cty, ctz = i * 0.1, 0.0, 0.0
        
        f.write(struct.pack('<i', image_id))
        f.write(struct.pack('<d', cqw))
        f.write(struct.pack('<d', cqx))
        f.write(struct.pack('<d', cqy))
        f.write(struct.pack('<d', cqz))
        f.write(struct.pack('<d', ctx))
        f.write(struct.pack('<d', cty))
        f.write(struct.pack('<d', ctz))
        f.write(struct.pack('<i', 1))  # camera_id = 1
        f.write(fname.encode('utf-8') + b'\x00')
        f.write(struct.pack('<Q', 0))  # 0 keypoints

pose_source = "MASt3R-SLAM trajectory" if SLAM_TRAJ is not None else "identity (fallback)"
print(f"✓ images.bin: {len(frame_names)} images, poses from: {pose_source}")

# %%
# --- Write points3D.bin ---
# Load MASt3R-SLAM's PLY point cloud and convert to COLMAP format

points_path = f"{COLMAP_DIR}/sparse/0/points3D.bin"

if SLAM_PLY and os.path.exists(SLAM_PLY):
    ply = PlyData.read(SLAM_PLY)
    vertices = ply['vertex']
    xyz = np.column_stack([vertices['x'], vertices['y'], vertices['z']])
    
    if 'red' in vertices.data.dtype.names:
        rgb = np.column_stack([vertices['red'], vertices['green'], vertices['blue']]).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)
    
    print(f"MASt3R-SLAM point cloud: {len(xyz)} points")
    print(f"  X range: [{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}]")
    print(f"  Y range: [{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}]")
    print(f"  Z range: [{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]")
    
    # Filter invalid points
    valid = np.all(np.isfinite(xyz), axis=1)
    norms = np.linalg.norm(xyz, axis=1)
    valid &= (norms > 1e-6) & (norms < 100.0)
    xyz = xyz[valid]
    rgb = rgb[valid]
    
    # Subsample to max 100k points for COLMAP init
    max_pts = min(100_000, len(xyz))
    if len(xyz) > max_pts:
        indices = np.random.choice(len(xyz), max_pts, replace=False)
        xyz = xyz[indices]
        rgb = rgb[indices]
    
    with open(points_path, 'wb') as f:
        f.write(struct.pack('<Q', len(xyz)))
        for j in range(len(xyz)):
            f.write(struct.pack('<Q', j + 1))  # point3d_id
            for v in xyz[j]:
                f.write(struct.pack('<d', float(v)))
            for c in rgb[j]:
                f.write(struct.pack('<B', int(c)))
            f.write(struct.pack('<d', 0.0))  # error
            f.write(struct.pack('<Q', 0))     # empty track
    
    print(f"✓ points3D.bin: {len(xyz)} points (filtered + subsampled)")
else:
    with open(points_path, 'wb') as f:
        f.write(struct.pack('<Q', 0))
    print("✓ points3D.bin: 0 points (no PLY available)")

print(f"\n✓ COLMAP workspace ready at {COLMAP_DIR}/")

# %% [markdown]
# ---
# ## 4. Train 3D Gaussian Splatting (gsplat)
#
# gsplat uses `tyro` for CLI parsing. The `default` subcommand selects
# the default training configuration.
#
# **Verified command** (from simple_trainer.py source):
# ```bash
# python examples/simple_trainer.py default \
#     --data_dir PATH --data_factor 1 --result_dir PATH --disable_viewer
# ```
#
# **Expected time**: ~30-60 min on T4, ~15-30 min on A100.

# %%
# Install gsplat and all trainer dependencies (avoiding broken wheel builds)
!pip install gsplat==1.3.0
!pip install tyro viser imageio[ffmpeg] tensorboard torchmetrics[image] opencv-python tqdm scipy nerfview splines pycolmap PyYAML piexif

# Clone gsplat repo to access training scripts
GSPLAT_REPO = "/content/gsplat"
if not os.path.exists(GSPLAT_REPO):
    !git clone -b v1.3.0 https://github.com/nerfstudio-project/gsplat.git {GSPLAT_REPO}
    !touch {GSPLAT_REPO}/examples/datasets/__init__.py
    !wget -O {GSPLAT_REPO}/examples/datasets/colmap.py https://github.com/nerfstudio-project/gsplat/raw/9b9f98a5b440531376b4a5386aea49f8e820203b/examples/datasets/colmap.py
    !wget -O {GSPLAT_REPO}/examples/exif.py https://raw.githubusercontent.com/nerfstudio-project/gsplat/main/examples/exif.py
    !wget -O {GSPLAT_REPO}/examples/datasets/exif.py https://raw.githubusercontent.com/nerfstudio-project/gsplat/main/examples/exif.py
    !sed -i 's/align_principal_axes/align_principle_axes/g' {GSPLAT_REPO}/examples/datasets/colmap.py

# %%
# Pre-compile CUDA extensions (prevent Out-Of-Memory swap thrashing)
print("Pre-compiling gsplat CUDA extensions (this takes ~5 minutes)...")
!rm -rf ~/.cache/torch_extensions/
!MAX_JOBS=2 python -c "import gsplat.cuda._backend"
print("✓ Compilation finished!")

# %%
# Train gsplat
os.makedirs(GSPLAT_OUTPUT, exist_ok=True)

!cd {GSPLAT_REPO} && MAX_JOBS=1 python examples/simple_trainer.py default \
    --data_dir {COLMAP_DIR} \
    --data_factor 1 \
    --result_dir {GSPLAT_OUTPUT} \
    --disable_viewer

print("\n✓ gsplat training complete")

# %%
# Convert gsplat .pt to standard .ply format
import torch
import numpy as np
import glob
import os

print("Installing plyfile...")
!pip install -q plyfile
from plyfile import PlyData, PlyElement

# Find the latest checkpoint
ckpt_paths = sorted(glob.glob(f"{GSPLAT_OUTPUT}/**/ckpts/*.pt", recursive=True))
if not ckpt_paths:
    print("⚠ No gsplat checkpoints found!")
else:
    ckpt_path = ckpt_paths[-1]
    print(f"Loading checkpoint: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    splats = ckpt["splats"]
    
    means = splats["means"].numpy()
    scales = splats["scales"].numpy()
    quats = splats["quats"].numpy()
    opacities = splats["opacities"].numpy()
    sh0 = splats["sh0"].numpy()
    shN = splats["shN"].numpy() if "shN" in splats else None
    
    N = means.shape[0]
    sh0 = sh0.reshape(N, 3)
    if shN is not None:
        shN = shN.reshape(N, -1)
    
    # Create ply property datatypes
    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    ]
    for i in range(45):
        dtype_full.append((f'f_rest_{i}', 'f4'))
    dtype_full.extend([
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ])
    
    elements = np.empty(N, dtype=dtype_full)
    elements['x'] = means[:, 0]
    elements['y'] = means[:, 1]
    elements['z'] = means[:, 2]
    elements['nx'] = 0
    elements['ny'] = 0
    elements['nz'] = 0
    elements['f_dc_0'] = sh0[:, 0]
    elements['f_dc_1'] = sh0[:, 1]
    elements['f_dc_2'] = sh0[:, 2]
    
    if shN is not None:
        for i in range(min(45, shN.shape[1])):
            elements[f'f_rest_{i}'] = shN[:, i]
    else:
        for i in range(45):
            elements[f'f_rest_{i}'] = 0
            
    elements['opacity'] = opacities.flatten()
    elements['scale_0'] = scales[:, 0]
    elements['scale_1'] = scales[:, 1]
    elements['scale_2'] = scales[:, 2]
    elements['rot_0'] = quats[:, 0] # w
    elements['rot_1'] = quats[:, 1] # x
    elements['rot_2'] = quats[:, 2] # y
    elements['rot_3'] = quats[:, 3] # z
    
    out_ply = f"{GSPLAT_OUTPUT}/gsplat_final.ply"
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(out_ply)
    
    size_mb = os.path.getsize(out_ply) / 1e6
    print(f"\n✓ Successfully exported {N} Gaussians to .ply format!")
    print(f"Final Gaussian splat: {out_ply} ({size_mb:.1f} MB)")
    
    FINAL_PLY = out_ply

# %% [markdown]
# ---
# ## 5. Grounded-SAM-2 Semantic Masks
#
# ### Verified API (from official examples + HuggingFace):
# - **Grounding DINO**: via `transformers.AutoModelForZeroShotObjectDetection`
# - **SAM 2**: via `sam2.build_sam` + `sam2.sam2_image_predictor.SAM2ImagePredictor`
#
# **Expected time**: ~5-15 min for 100 frames on T4.

# %%
# Install dependencies
!pip install sam2 supervision

# Download SAM 2.1 checkpoint
SAM2_CKPT_DIR = "/content/checkpoints"
os.makedirs(SAM2_CKPT_DIR, exist_ok=True)

SAM2_CKPT = f"{SAM2_CKPT_DIR}/sam2.1_hiera_large.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

if not os.path.exists(SAM2_CKPT):
    !wget -q --show-progress -O {SAM2_CKPT} \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

print("✓ SAM 2.1 checkpoint ready")

# %%
# Run Grounded-SAM-2 on each frame
import json
import cv2
import numpy as np
import torch
from tqdm import tqdm

os.makedirs(MASKS_DIR, exist_ok=True)

# Grounding DINO text prompt: lowercase with periods
TEXT_PROMPT = "chair. table. sofa. monitor. laptop. cup. floor. wall. door. shelf. bed. lamp."

frame_paths = sorted([
    os.path.join(FRAMES_DIR, f)
    for f in os.listdir(FRAMES_DIR)
    if f.endswith(('.jpg', '.png'))
])

print(f"Processing {len(frame_paths)} frames")
print(f"Text prompt: {TEXT_PROMPT}")

try:
    # --- Load Grounding DINO (via HuggingFace Transformers) ---
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    gdino_id = "IDEA-Research/grounding-dino-tiny"
    gdino_processor = AutoProcessor.from_pretrained(gdino_id)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_id).to("cuda")
    print("✓ Grounding DINO loaded")

    # --- Load SAM 2 ---
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2_model = build_sam2(SAM2_CFG, SAM2_CKPT)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    print("✓ SAM 2 loaded")

    # --- Run inference on each frame ---
    all_masks_info = []

    for i, frame_path in enumerate(tqdm(frame_paths)):
        image = cv2.imread(frame_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Step 1: Detect objects with Grounding DINO
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(image_rgb)
        
        inputs = gdino_processor(images=pil_image, text=TEXT_PROMPT, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = gdino_model(**inputs)
        
        results = gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=0.3,
            text_threshold=0.25,
            target_sizes=[(h, w)],
        )

        boxes = results[0]["boxes"].cpu().numpy()  # (N, 4) xyxy pixel coords
        scores = results[0]["scores"].cpu().numpy()
        labels = results[0]["labels"]  # list of strings

        # Step 2: Generate masks with SAM 2
        frame_masks = []
        frame_name = os.path.basename(frame_path).rsplit('.', 1)[0]

        if len(boxes) > 0:
            sam2_predictor.set_image(image_rgb)
            masks, sam_scores, _ = sam2_predictor.predict(
                box=boxes,
                multimask_output=False,
            )

            for j in range(len(masks)):
                mask = masks[j].squeeze().astype(np.uint8) * 255
                mask_filename = f"{frame_name}_mask_{j:03d}.png"
                cv2.imwrite(os.path.join(MASKS_DIR, mask_filename), mask)

                frame_masks.append({
                    "mask_file": mask_filename,
                    "label": labels[j] if j < len(labels) else "unknown",
                    "confidence": float(scores[j]) if j < len(scores) else 0.0,
                    "instance_id": j,
                })

        all_masks_info.append({
            "frame": os.path.basename(frame_path),
            "frame_index": i,
            "masks": frame_masks,
        })

    # Save manifest
    with open(os.path.join(MASKS_DIR, "masks.json"), 'w') as f:
        json.dump(all_masks_info, f, indent=2)

    total_masks = sum(len(info["masks"]) for info in all_masks_info)
    print(f"\n✓ Saved {total_masks} masks across {len(frame_paths)} frames to {MASKS_DIR}/")

except ImportError as e:
    print(f"\n⚠ Import error: {e}")
    print("Install with: pip install sam2 transformers")
    print("Or check: https://github.com/IDEA-Research/Grounded-SAM-2")

except Exception as e:
    print(f"\n⚠ Error during segmentation: {e}")
    import traceback
    traceback.print_exc()
    print("Continuing without semantic masks — you can add them later.")

# %% [markdown]
# ---
# ## 6. Save Everything to Google Drive

# %%
os.makedirs(DRIVE_OUTPUT, exist_ok=True)

# 1. COLMAP workspace
colmap_drive = f"{DRIVE_OUTPUT}/colmap"
if os.path.exists(colmap_drive):
    shutil.rmtree(colmap_drive)
shutil.copytree(COLMAP_DIR, colmap_drive)
print(f"✓ COLMAP workspace → {colmap_drive}/")

# 2. Gaussian splat .ply
if FINAL_PLY and os.path.exists(FINAL_PLY):
    ply_drive = f"{DRIVE_OUTPUT}/splat.ply"
    shutil.copy2(FINAL_PLY, ply_drive)
    size_mb = os.path.getsize(ply_drive) / 1e6
    print(f"✓ Gaussian splat ({size_mb:.0f} MB) → {ply_drive}")

# 3. Semantic masks
if os.path.exists(MASKS_DIR) and len(os.listdir(MASKS_DIR)) > 0:
    masks_drive = f"{DRIVE_OUTPUT}/masks"
    if os.path.exists(masks_drive):
        shutil.rmtree(masks_drive)
    shutil.copytree(MASKS_DIR, masks_drive)
    num_masks = len([f for f in os.listdir(MASKS_DIR) if f.endswith('.png')])
    print(f"✓ Semantic masks ({num_masks} files) → {masks_drive}/")

# 4. Raw SLAM outputs (trajectory + pointcloud)
if SLAM_PLY and os.path.exists(SLAM_PLY):
    shutil.copy2(SLAM_PLY, f"{DRIVE_OUTPUT}/slam_pointcloud.ply")
    print(f"✓ SLAM pointcloud → {DRIVE_OUTPUT}/slam_pointcloud.ply")

if SLAM_TRAJ is not None:
    traj_drive = f"{DRIVE_OUTPUT}/slam_trajectory.txt"
    np.savetxt(traj_drive, SLAM_TRAJ, fmt='%.6f')
    print(f"✓ SLAM trajectory → {traj_drive}")

# 5. Copy SLAM logs directory as-is (for debugging)
if os.path.exists(SLAM_LOGS):
    logs_drive = f"{DRIVE_OUTPUT}/slam_logs"
    if os.path.exists(logs_drive):
        shutil.rmtree(logs_drive)
    shutil.copytree(SLAM_LOGS, logs_drive)
    print(f"✓ SLAM logs → {logs_drive}/")

print(f"""
{'='*60}
 ALL RESULTS SAVED TO GOOGLE DRIVE
{'='*60}

 Location: {DRIVE_OUTPUT}/

 On your Mac, download from Google Drive and place as:
   colmap/          → data/{SCENE_NAME}/colmap/
   splat.ply        → outputs/{SCENE_NAME}/splat.ply
   masks/           → data/{SCENE_NAME}/masks/
   slam_logs/       → data/{SCENE_NAME}/slam_logs/ (for debugging)

 Then run the local pipeline:
   python -m viewer.app --scene {SCENE_NAME}
{'='*60}
""")

# %% [markdown]
# ---
# ## Done!
#
# Download results from Google Drive to your Mac and run
# the local pipeline (semantic lifting + CLIP embeddings + viewer).
