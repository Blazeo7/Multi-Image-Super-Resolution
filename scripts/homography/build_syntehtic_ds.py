"""
build_dataset.py

Processes a directory of texture files with the naming convention:
    <sample_id>_deadon_HQ.png       — 1024×1024 high-resolution target
    <sample_id>_angle_NNN_LQ.png    — 512×512 low-resolution angled views
    <sample_id>_matrices.json       — (optional) transform metadata, unused

For each sample the script:
  1. Downscales the HQ deadon image to 512×512  → LR reference
  2. Aligns every angle_LQ image to the LR reference via:
       a. SIFT homography  (or LocalDescriptor model if --model is given)
       b. RAFT optical-flow refinement on top of the homography warp
  3. Copies the original HQ to the output directory
  4. Writes a manifest JSON describing the whole dataset

Output layout:
    output_dir/
      lr/               LR reference + aligned angle images
      hr/               copied original HQ files
      manifest.json

Usage:
    python build_dataset.py --input-dir /path/to/textures --output-dir /path/to/out

Optional flags:
    --scale-factor INT      Downscale factor recorded in the manifest  (default: 2)
    --model PATH            Local descriptor model for homography       (optional)
    --of-scale FLOAT        Scale factor for RAFT inference             (default: 0.5)
    --skip-existing         Skip samples whose LR reference already exists
    --device STR            Torch device for RAFT                       (default: cuda)
"""

import os

# Must be set before numpy, cv2, and torch are imported —
# these thread pools are initialized at import time.
os.environ["OMP_NUM_THREADS"]          = "1"
os.environ["OPENBLAS_NUM_THREADS"]     = "1"
os.environ["MKL_NUM_THREADS"]          = "1"
os.environ["NUMEXPR_NUM_THREADS"]      = "1"
os.environ["CV_NUM_THREADS"]           = "1"
os.environ["TORCH_NUM_THREADS"]        = "1"
os.environ["TORCH_NUM_INTEROP_THREADS"] = "1"

import argparse
import json
import logging
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from tqdm import tqdm

# Belt-and-suspenders: also set at runtime after torch is imported
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LR_SIZE = (512, 512)    # (width, height)
HR_SIZE = (1024, 1024)  # (width, height)

RE_HQ  = re.compile(r"^(.+)_deadon_HQ\.png$")
RE_LQ  = re.compile(r"^(.+)_angle_(\d{3})_LQ\.png$")
RE_MAT = re.compile(r"^(.+)_matrices\.json$")


# ---------------------------------------------------------------------------
# Homography  (ported verbatim from original script)
# ---------------------------------------------------------------------------

def compute_homography(img1: np.ndarray, img2: np.ndarray, local_descriptor_model=None, device: str = "cuda"):
    """
    Compute the homography matrix that maps points from img1 to img2.
    Tries multiple SIFT sigma values until enough inliers are found.
    Falls back to LocalDescriptor model when provided.
    Returns H (3x3 ndarray) or None on failure.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    for sigma in [1.6, 2.5, 3.5, 1.4, 1.2]:
        if local_descriptor_model is None:
            sift = cv2.SIFT_create(20000, 3, sigma=sigma)
            t1 = time.time()
            kp1, des1 = sift.detectAndCompute(gray1, None)
            kp2, des2 = sift.detectAndCompute(gray2, None)
            t2 = time.time()

            desc1_t = torch.from_numpy(des1).to(device).float()
            desc2_t = torch.from_numpy(des2).to(device).float()
            desc1_t = desc1_t / desc1_t.norm(dim=1)[:, None]
            desc2_t = desc2_t / desc2_t.norm(dim=1)[:, None]
        else:
            orb = cv2.ORB_create(nfeatures=2000)
            t1 = time.time()
            kp1 = orb.detect(gray1, None)
            kp2 = orb.detect(gray2, None)
            t2 = time.time()
            kp1 = local_descriptor_model.filter_points_kp(img1, kp1)
            kp2 = local_descriptor_model.filter_points_kp(img2, kp2)
            desc1_t = local_descriptor_model.process_page_kp(img1, kp1)
            desc2_t = local_descriptor_model.process_page_kp(img2, kp2)

        sim = torch.mm(desc1_t, desc2_t.t())
        matches = torch.topk(sim, k=2, dim=1)

        good_matches = []
        sim_threshold = 0.93
        match_indices      = matches.indices.cpu().numpy()
        match_similarities = matches.values.cpu().numpy()
        for i in range(match_indices.shape[0]):
            if match_similarities[i, 0] * sim_threshold > match_similarities[i, 1]:
                good_matches.append(
                    cv2.DMatch(i, match_indices[i, 0], 1 - match_similarities[i, 0])
                )
        t3 = time.time()

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        if len(src_pts) > 30:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 15.0, confidence=0.9999)
        else:
            H, mask = None, np.zeros(1, dtype=np.uint8)
        t4 = time.time()

        log.info(
            "MATCH ransac_matches: %d, good_matches: %d, kp1: %d, kp2: %d "
            "in %.2f / %.2f / %.2f s",
            mask.sum(), len(good_matches), len(kp1), len(kp2),
            t2 - t1, t3 - t1, t4 - t1,
        )

        if mask.sum() > 30:
            return H
        else:
            H = None
            if local_descriptor_model is not None:
                break  # model doesn't benefit from sigma retry

    return None


# ---------------------------------------------------------------------------
# RAFT helpers  (ported verbatim from original script)
# ---------------------------------------------------------------------------

def _preprocess(batch: torch.Tensor) -> torch.Tensor:
    transforms = T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=0.5, std=0.5),
    ])
    return transforms(batch)


def optical_flow_remap(
    img_warped: np.ndarray,
    reference: np.ndarray,
    of_model: torch.nn.Module,
    device: str,
    of_scale: float = 0.5,
):
    """
    Refine alignment of *img_warped* (already homography-warped) to *reference*
    using RAFT optical flow.

    Returns
    -------
    img_refined : np.ndarray   remapped image (same shape as reference)
    map_x, map_y : np.ndarray  full-resolution remap maps
    """
    of_imgs = [
        cv2.resize(img, (int(img.shape[1] * of_scale), int(img.shape[0] * of_scale)))
        for img in [reference, img_warped]
    ]

    img_stack = np.stack(of_imgs, 0)  # (2, H, W, C)
    img_torch = _preprocess(
        torch.from_numpy(np.transpose(img_stack, (0, 3, 1, 2)) / 255.0)
    ).to(device)

    img0_t = img_torch[0:1]
    img1_t = img_torch[1:2]

    with torch.no_grad():
        h, w = img0_t.shape[-2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        img0_t = torch.nn.functional.pad(img0_t, (0, pad_w, 0, pad_h))
        img1_t = torch.nn.functional.pad(img1_t, (0, pad_w, 0, pad_h))
        predicted_flows = of_model(img0_t, img1_t, num_flow_updates=8)
        predicted_flows = predicted_flows[-1][0]        # (2, H_pad, W_pad)
        predicted_flows = predicted_flows[:, :h, :w]   # remove padding
        predicted_flows = predicted_flows.cpu().numpy()

    flow_x = cv2.resize(predicted_flows[0] / of_scale,
                        (reference.shape[1], reference.shape[0]))
    flow_y = cv2.resize(predicted_flows[1] / of_scale,
                        (reference.shape[1], reference.shape[0]))

    map_x = (np.arange(0, reference.shape[1])[np.newaxis, :] + flow_x).astype(np.float32)
    map_y = (np.arange(0, reference.shape[0])[:, np.newaxis] + flow_y).astype(np.float32)

    img_refined = cv2.remap(img_warped, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return img_refined, map_x, map_y


# ---------------------------------------------------------------------------
# File grouping
# ---------------------------------------------------------------------------


def alignment_ncc(img_aligned: np.ndarray, img_ref: np.ndarray) -> float:
    """
    Compute normalized cross-correlation between two images.
    Only considers pixels where both images are non-zero (ignores black borders
    from warping). Returns a score in [-1, 1]; well-aligned images score > 0.9.
    """
    gray_a = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_r = cv2.cvtColor(img_ref,     cv2.COLOR_BGR2GRAY).astype(np.float32)

    mask = (gray_a > 0) & (gray_r > 0)
    if mask.sum() < 100:
        return 0.0

    a = gray_a[mask]
    r = gray_r[mask]
    a -= a.mean(); r -= r.mean()
    denom = np.sqrt((a ** 2).sum() * (r ** 2).sum())
    if denom < 1e-6:
        return 0.0
    return float(np.dot(a, r) / denom)

def group_samples(input_dir: Path) -> dict:
    """
    Scan *input_dir* and group files by sample prefix.

    Returns { sample_id: {"hq": Path, "lq": [Path, ...], "matrices": Path|None} }
    """
    samples: dict = defaultdict(lambda: {"hq": None, "lq": [], "matrices": None})

    for f in sorted(input_dir.iterdir()):
        name = f.name
        m = RE_HQ.match(name)
        if m:
            samples[m.group(1)]["hq"] = f
            continue
        m = RE_LQ.match(name)
        if m:
            samples[m.group(1)]["lq"].append(f)
            continue
        m = RE_MAT.match(name)
        if m:
            samples[m.group(1)]["matrices"] = f

    for s in samples.values():
        s["lq"].sort(key=lambda p: p.name)

    return {
        sid: data
        for sid, data in samples.items()
        if data["hq"] is not None and len(data["lq"]) >= 1
    }


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

def process_sample(
    sample_id: str,
    data: dict,
    output_dir: Path,
    of_model: torch.nn.Module,
    device: str,
    of_scale: float,
    local_descriptor_model,
    skip_existing: bool,
    ncc_threshold: float = 0.85,
    min_supporting: int = 1,
) -> dict | None:

    hq_src: Path  = data["hq"]
    lq_srcs: list = data["lq"]

    lr_dir = output_dir / "lr"
    hr_dir = output_dir / "hr"
    lr_dir.mkdir(parents=True, exist_ok=True)
    hr_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Generate LR reference by downscaling the HQ deadon image        #
    # ------------------------------------------------------------------ #
    lr_ref_path = lr_dir / f"{sample_id}_deadon_LR.png"

    if not (skip_existing and lr_ref_path.exists()):
        hq_img = cv2.imread(str(hq_src))
        if hq_img is None:
            log.error("[%s] cannot read HQ: %s", sample_id, hq_src)
            return None

        if hq_img.shape[:2] != (HR_SIZE[1], HR_SIZE[0]):
            log.warning(
                "[%s] HQ is %dx%d, expected %dx%d — resizing anyway.",
                sample_id, hq_img.shape[1], hq_img.shape[0], *HR_SIZE,
            )

        lr_ref_img = cv2.resize(hq_img, LR_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(lr_ref_path), lr_ref_img)
        log.info("[%s] wrote LR reference → %s", sample_id, lr_ref_path.name)
    else:
        log.info("[%s] LR reference exists, skipping generation", sample_id)

    lr_ref_img = cv2.imread(str(lr_ref_path))
    if lr_ref_img is None:
        log.error("[%s] cannot read LR reference: %s", sample_id, lr_ref_path)
        return None

    # ------------------------------------------------------------------ #
    # 2. Copy HQ to output/hr/                                           #
    # ------------------------------------------------------------------ #
    hr_target_path = hr_dir / hq_src.name
    if not (skip_existing and hr_target_path.exists()):
        shutil.copy2(hq_src, hr_target_path)

    # ------------------------------------------------------------------ #
    # 3. Align each angle LQ → LR reference                              #
    #    homography warp  →  RAFT optical-flow refinement                #
    # ------------------------------------------------------------------ #
    aligned_paths: list[str] = []
    failed = 0

    for lq_src in lq_srcs:
        out_path = lr_dir / f"{lq_src.stem}_aligned.png"

        if skip_existing and out_path.exists():
            log.info("[%s] %s already aligned, skipping", sample_id, lq_src.name)
            aligned_paths.append(str(out_path.relative_to(output_dir)))
            continue

        lq_img = cv2.imread(str(lq_src))
        if lq_img is None:
            log.warning("[%s] cannot read %s — skipping", sample_id, lq_src.name)
            failed += 1
            continue

        if lq_img.shape[:2] != (LR_SIZE[1], LR_SIZE[0]):
            lq_img = cv2.resize(lq_img, LR_SIZE, interpolation=cv2.INTER_AREA)

        # -- Step A: homography --
        H = compute_homography(lq_img, lr_ref_img, local_descriptor_model, device)

        if H is None:
            log.warning(
                "[%s] homography failed for %s — skipping",
                sample_id, lq_src.name,
            )
            failed += 1
            continue

        lq_warped = cv2.warpPerspective(lq_img, H, LR_SIZE)

        # -- Step B: RAFT optical-flow refinement --
        try:
            lq_refined, _, _ = optical_flow_remap(
                lq_warped, lr_ref_img, of_model, device, of_scale
            )
        except Exception as exc:
            log.warning(
                "[%s] optical flow failed for %s (%s) — using homography warp only",
                sample_id, lq_src.name, exc,
            )
            lq_refined = lq_warped

        ncc = alignment_ncc(lq_refined, lr_ref_img)
        if ncc < ncc_threshold:
            log.warning(
                "[%s] %s NCC=%.3f below threshold %.2f — dropping",
                sample_id, lq_src.name, ncc, ncc_threshold,
            )
            failed += 1
            continue
        cv2.imwrite(str(out_path), lq_refined)
        aligned_paths.append(str(out_path.relative_to(output_dir)))
        log.info("[%s] aligned %s  NCC=%.3f", sample_id, lq_src.name, ncc)

    if len(aligned_paths) < min_supporting:
        log.error(
            "[%s] only %d/%d images passed NCC check (min %d) — dropping sample",
            sample_id, len(aligned_paths), len(lq_srcs), min_supporting,
        )
        return None

    log.info(
        "[%s] done  (%d aligned, %d failed/unaligned)",
        sample_id, len(aligned_paths) - failed, failed,
    )

    return {
        "id": sample_id,
        "reference": str(lr_ref_path.relative_to(output_dir)),
        "supporting": aligned_paths,
        "target": str(hr_target_path.relative_to(output_dir)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build an aligned LR/HR dataset from texture files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir",    required=True, type=Path,
                   help="Directory containing raw texture files.")
    p.add_argument("--output-dir",   required=True, type=Path,
                   help="Root output directory.")
    p.add_argument("--scale-factor", type=int,   default=2,
                   help="Downscale factor written into the manifest.")
    p.add_argument("--model",        type=str,   default=None,
                   help="Path to a LocalDescriptor model for homography (optional).")
    p.add_argument("--of-scale",     type=float, default=0.5,
                   help="Scale factor applied to images before RAFT inference.")
    p.add_argument("--device",       type=str,   default="cuda",
                   help="Torch device for RAFT (cuda or cpu).")
    p.add_argument("--ncc-threshold", type=float, default=0.85,
                   help="Minimum NCC score to accept an aligned image (0-1).")
    p.add_argument("--min-supporting", type=int, default=1,
                   help="Drop sample if fewer than this many images pass NCC check.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip samples whose LR reference already exists.")
    return p.parse_args()


def main():
    args = parse_args()

    input_dir:  Path = args.input_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not input_dir.is_dir():
        log.error("Input directory does not exist: %s", input_dir)
        raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Restrict to a single GPU before any CUDA context is created.
    # Parses the device index from e.g. "cuda:2" -> sets CUDA_VISIBLE_DEVICES=2
    # so PyTorch sees exactly one device regardless of cluster config.
    if args.device.startswith("cuda"):
        parts = args.device.split(":")
        gpu_index = parts[1] if len(parts) > 1 else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
        args.device = "cuda:0"  # after restricting visibility it's always 0 internally
        log.info("Restricted to GPU index %s (CUDA_VISIBLE_DEVICES=%s)", gpu_index, gpu_index)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # cpu mode: hide all GPUs

    # Load models once — reused for every sample
    log.info("Loading RAFT large model on %s …", args.device)
    of_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
    of_model = of_model.to(args.device).eval()

    local_descriptor_model = None
    if args.model:
        from cnn_descriptor import LocalDescriptor
        local_descriptor_model = LocalDescriptor(args.model)
        log.info("Loaded LocalDescriptor model from %s", args.model)

    samples = group_samples(input_dir)
    log.info("Found %d valid samples in %s", len(samples), input_dir)

    if not samples:
        log.error("No valid samples found — check filenames match expected pattern.")
        raise SystemExit(1)

    # Sequential processing: RAFT + SIFT are GPU-bound; multiprocessing
    # would require per-worker model copies and add significant overhead.
    manifest_samples: list[dict] = []

    for sample_id, data in tqdm(samples.items(), desc='Samples', unit='sample'):
        try:
            entry = process_sample(
                sample_id=sample_id,
                data=data,
                output_dir=output_dir,
                of_model=of_model,
                device=args.device,
                of_scale=args.of_scale,
                local_descriptor_model=local_descriptor_model,
                skip_existing=args.skip_existing,
                ncc_threshold=args.ncc_threshold,
                min_supporting=args.min_supporting,
            )
        except Exception as exc:
            log.error("[%s] unhandled exception: %s", sample_id, exc, exc_info=True)
            entry = None

        if entry is not None:
            manifest_samples.append(entry)

    manifest_samples.sort(key=lambda e: e["id"])

    counts = [len(e["supporting"]) for e in manifest_samples] if manifest_samples else [0]
    num_aligned = max(set(counts), key=counts.count)

    manifest = {
        "scale_factor": args.scale_factor,
        "num_aligned_images": num_aligned,
        "samples": manifest_samples,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=4))

    log.info(
        "Manifest written → %s  (%d samples, mode %d supporting images)",
        manifest_path, len(manifest_samples), num_aligned,
    )


if __name__ == "__main__":
    main()
