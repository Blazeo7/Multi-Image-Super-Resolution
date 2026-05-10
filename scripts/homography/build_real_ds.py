import os

# Threading constraints for high-performance clusters
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reference selection — sharpest image by Laplacian variance
# ---------------------------------------------------------------------------

def laplacian_sharpness(img: np.ndarray) -> float:
    """Higher = sharper. Computed on luma at working resolution."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def select_reference(image_files: list[Path], work_scale: float) -> tuple[Path, list[Path]]:
    """
    Returns (reference_path, remaining_paths) where reference is the sharpest image.
    Images are evaluated at work_scale to keep this fast.
    """
    best_path, best_score = None, -1.0
    scores = {}
    for p in image_files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        small = cv2.resize(img, (0, 0), fx=work_scale, fy=work_scale,
                           interpolation=cv2.INTER_AREA)
        score = laplacian_sharpness(small)
        scores[p] = score
        if score > best_score:
            best_score = score
            best_path = p

    others = [p for p in image_files if p != best_path and p in scores]
    return best_path, others


# ---------------------------------------------------------------------------
# Homography estimation with full quality diagnostics
# ---------------------------------------------------------------------------

def compute_homography_with_quality(
    img_src: np.ndarray,
    img_ref: np.ndarray,
    work_scale: float,
) -> tuple[Optional[np.ndarray], dict]:
    """
    Estimate homography mapping img_src → img_ref at work_scale,
    then scale H back to full-image coordinates.

    Returns
    -------
    H_full : 3x3 ndarray in full-res pixel coords, or None on failure
    quality : dict of SIFT/RANSAC diagnostics
    """
    quality = dict(
        num_good_matches=0,
        num_inliers=0,
        inlier_ratio=0.0,
        reproj_error_mean=999.0,
        reproj_error_std=999.0,
    )

    def _resize(img):
        return cv2.resize(img, (0, 0), fx=work_scale, fy=work_scale,
                          interpolation=cv2.INTER_AREA)

    small_src = _resize(img_src)
    small_ref = _resize(img_ref)

    gray_src = cv2.cvtColor(small_src, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(small_ref, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(8000, 3, sigma=1.6)
    kp1, des1 = sift.detectAndCompute(gray_src, None)
    kp2, des2 = sift.detectAndCompute(gray_ref, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, quality

    bf = cv2.BFMatcher()
    raw = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    quality["num_good_matches"] = len(good)

    if len(good) < 20:
        return None, quality

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H_small, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, confidence=0.9999)
    if H_small is None:
        return None, quality

    inlier_mask = mask.ravel().astype(bool)
    num_inliers = int(inlier_mask.sum())
    quality["num_inliers"] = num_inliers
    quality["inlier_ratio"] = round(num_inliers / len(good), 4)

    if num_inliers < 20:
        return None, quality

    # Reprojection error (at working resolution)
    src_in = src_pts[inlier_mask]
    dst_in = dst_pts[inlier_mask]
    errors = np.linalg.norm(
        cv2.perspectiveTransform(src_in, H_small) - dst_in, axis=2
    ).ravel()
    quality["reproj_error_mean"] = round(float(errors.mean()) / work_scale, 3)  # in full-res px
    quality["reproj_error_std"]  = round(float(errors.std())  / work_scale, 3)

    # Scale H from working-res coords → full-res coords
    # H_full = S_inv · H_small · S   where S = diag(work_scale, work_scale, 1)
    S     = np.diag([work_scale, work_scale, 1.0])
    S_inv = np.diag([1.0 / work_scale, 1.0 / work_scale, 1.0])
    H_full = S_inv @ H_small @ S

    return H_full, quality


# ---------------------------------------------------------------------------
# Optical flow quality measurement (no output image saved)
# ---------------------------------------------------------------------------

def measure_flow_quality(
    img_src: np.ndarray,
    img_ref: np.ndarray,
    H_full: np.ndarray,
    of_model,
    device: str,
    of_scale: float,
) -> tuple[float, float, float, float]:
    """
    Warp img_src with H_full, run RAFT, return
    (ncc_before_flow, ncc_after_flow, flow_mean, flow_max).
    Nothing is written to disk.
    """
    h, w = img_ref.shape[:2]
    warped = cv2.warpPerspective(img_src, H_full, (w, h))

    ncc_before = _ncc(warped, img_ref)

    # Resize for OF
    sh, sw = int(h * of_scale), int(w * of_scale)
    ref_s  = cv2.resize(img_ref, (sw, sh), interpolation=cv2.INTER_AREA)
    warp_s = cv2.resize(warped,  (sw, sh), interpolation=cv2.INTER_AREA)

    inp = np.stack([ref_s, warp_s])
    inp_t = torch.from_numpy(inp.transpose(0, 3, 1, 2)).float().to(device)
    inp_t = (inp_t / 127.5) - 1.0

    with torch.no_grad():
        pad_h = (8 - inp_t.shape[-2] % 8) % 8
        pad_w = (8 - inp_t.shape[-1] % 8) % 8
        inp_p = torch.nn.functional.pad(inp_t, (0, pad_w, 0, pad_h))
        flow  = of_model(inp_p[0:1], inp_p[1:2], num_flow_updates=12)[-1][0]
        flow  = flow[:, :sh, :sw].cpu().numpy()

    # Scale flow magnitudes to full-res pixels for interpretability
    flow_x = cv2.resize(flow[0], (w, h)) / of_scale
    flow_y = cv2.resize(flow[1], (w, h)) / of_scale
    mag    = np.sqrt(flow_x ** 2 + flow_y ** 2)

    # Apply flow to get refined warp (only used for NCC measurement)
    mx = (np.arange(w).reshape(1, -1) + cv2.resize(flow[0], (w, h))).astype(np.float32)
    my = (np.arange(h).reshape(-1, 1) + cv2.resize(flow[1], (w, h))).astype(np.float32)
    refined = cv2.remap(warped, mx, my, interpolation=cv2.INTER_LANCZOS4)

    ncc_after = _ncc(refined, img_ref)

    return (
        round(ncc_before, 4),
        round(ncc_after,  4),
        round(float(mag.mean()), 3),
        round(float(mag.max()),  3),
    )


def _ncc(img_a: np.ndarray, img_b: np.ndarray) -> float:
    ga = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mask = (ga > 0) & (gb > 0)
    if mask.sum() < 1000:
        return 0.0
    a, b = ga[mask], gb[mask]
    a -= a.mean(); b -= b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float(np.dot(a, b) / denom) if denom > 1e-6 else 0.0


# ---------------------------------------------------------------------------
# Tile homography derivation (stored as utility, used by dataloader)
# ---------------------------------------------------------------------------

def tile_homography(H_full: np.ndarray, src_tile_x: int, src_tile_y: int,
                    dst_tile_x: int, dst_tile_y: int) -> np.ndarray:
    """
    Derive homography for a specific tile pair from the full-image H.

        H_tile = T_dst_inv · H_full · T_src

    T_src shifts origin to top-left of the source tile.
    T_dst shifts origin to top-left of the reference tile.
    Call this in your dataloader — no need to store per-tile matrices.
    """
    T_src = np.array([[1, 0, src_tile_x],
                      [0, 1, src_tile_y],
                      [0, 0, 1        ]], dtype=np.float64)
    T_dst_inv = np.array([[1, 0, -dst_tile_x],
                          [0, 1, -dst_tile_y],
                          [0, 0,  1         ]], dtype=np.float64)
    return T_dst_inv @ H_full @ T_src


# ---------------------------------------------------------------------------
# Per-scene processing
# ---------------------------------------------------------------------------

def process_scene(
    scene_id: str,
    scene_path: Path,
    args,
    of_model,
) -> Optional[dict]:

    image_files = sorted(scene_path.glob("*.jpg"))
    image_files = [p for p in image_files if cv2.haveImageReader(str(p))]

    if len(image_files) < 2:
        log.warning(f"Scene {scene_id}: fewer than 2 readable images, skipping.")
        return None

    # ── Select reference (sharpest) ─────────────────────────────────────────
    ref_path, other_paths = select_reference(image_files, args.work_scale)
    log.info(f"Scene {scene_id}: reference = {ref_path.name}  ({len(other_paths)} supporting)")

    img_ref = cv2.imread(str(ref_path))
    h_full, w_full = img_ref.shape[:2]

    # ── Build renders list ───────────────────────────────────────────────────
    renders = [
        {
            "type": "HQ",
            "filename": ref_path.name,
            "homography_matrix": np.eye(3).tolist(),
        }
    ]

    for render_id, src_path in enumerate(other_paths):
        img_src = cv2.imread(str(src_path))
        if img_src is None:
            log.warning(f"  {src_path.name}: could not read, skipping.")
            continue

        # 1. Homography + RANSAC quality
        H_full, ransac_q = compute_homography_with_quality(img_src, img_ref, args.work_scale)

        if H_full is None:
            log.info(f"  {src_path.name}: homography failed — {ransac_q['num_good_matches']} good matches")
            renders.append({
                "type":      "LQ",
                "render_id": render_id,
                "filename":  src_path.name,
                "passed":    False,
                "fail_reason": "homography_failed",
                "homography_matrix": None,
                "quality": {**ransac_q,
                            "ncc_before_flow":     None,
                            "ncc_after_flow":      None,
                            "flow_magnitude_mean": None,
                            "flow_magnitude_max":  None},
            })
            continue

        # 2. Early reproj gate (skip expensive OF if already bad)
        if ransac_q["reproj_error_mean"] > args.max_reproj_error:
            log.info(f"  {src_path.name}: reproj {ransac_q['reproj_error_mean']:.1f}px > {args.max_reproj_error}")
            renders.append({
                "type":      "LQ",
                "render_id": render_id,
                "filename":  src_path.name,
                "passed":    False,
                "fail_reason": "reproj_error_too_high",
                "homography_matrix": H_full.tolist(),
                "quality": {**ransac_q,
                            "ncc_before_flow":     None,
                            "ncc_after_flow":      None,
                            "flow_magnitude_mean": None,
                            "flow_magnitude_max":  None},
            })
            continue

        # 3. Optical flow quality measurement
        try:
            ncc_before, ncc_after, flow_mean, flow_max = measure_flow_quality(
                img_src, img_ref, H_full, of_model, args.device, args.of_scale
            )
        except Exception as e:
            log.warning(f"  {src_path.name}: OF failed ({e}), using NCC-only.")
            ncc_before = _ncc(cv2.warpPerspective(img_src, H_full, (w_full, h_full)), img_ref)
            ncc_after, flow_mean, flow_max = ncc_before, 999.0, 999.0

        quality = {
            **ransac_q,
            "ncc_before_flow":     ncc_before,
            "ncc_after_flow":      ncc_after,
            "flow_magnitude_mean": flow_mean,
            "flow_magnitude_max":  flow_max,
        }

        # 4. Remaining gates
        fail_reason = None
        if flow_mean > args.max_flow_mean:
            fail_reason = "flow_magnitude_too_high"
        elif ncc_after < args.ncc_threshold:
            fail_reason = "ncc_too_low"

        passed = fail_reason is None
        if not passed:
            log.info(f"  {src_path.name}: failed — {fail_reason}")

        renders.append({
            "type":             "LQ",
            "render_id":        render_id,
            "filename":         src_path.name,
            "passed":           passed,
            **({"fail_reason": fail_reason} if not passed else {}),
            "homography_matrix": H_full.tolist(),
            "quality":          quality,
        })

    # ── Assemble scene metadata ──────────────────────────────────────────────
    n_passed = sum(1 for r in renders if r.get("type") == "LQ" and r.get("passed"))
    log.info(f"Scene {scene_id}: {n_passed}/{len(renders)-1} supporting images passed.")

    return {
        "scene_id":  scene_id,
        "reference": ref_path.name,
        "image_size": {"width": w_full, "height": h_full},
        "tiling": {
            "tile_size": args.tile_size,
            "stride":    args.tile_stride,
        },
        "renders": renders,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build per-scene homography metadata for real captured images. "
                    "No images are copied or modified."
    )
    parser.add_argument("--input-dir",        type=Path,  required=True,
                        help="Root directory containing one subdirectory per scene/texture.")
    parser.add_argument("--output-dir",       type=Path,  required=True,
                        help="Where to write *_metadata.json files (mirrors input structure).")
    # Tiling
    parser.add_argument("--tile-size",        type=int,   default=256)
    parser.add_argument("--tile-stride",      type=int,   default=224,
                        help="Stride between tiles (tile_size - stride = overlap).")
    # Computation
    parser.add_argument("--work-scale",       type=float, default=0.5,
                        help="Scale at which SIFT runs. H is rescaled to full-res coords.")
    parser.add_argument("--of-scale",         type=float, default=0.5,
                        help="Scale at which RAFT runs for flow quality measurement.")
    parser.add_argument("--device",           type=str,   default="cuda")
    # Quality gates
    parser.add_argument("--ncc-threshold",    type=float, default=0.85)
    parser.add_argument("--max-reproj-error", type=float, default=3.0,
                        help="Max mean reprojection error in full-res pixels.")
    parser.add_argument("--max-flow-mean",    type=float, default=8.0,
                        help="Max mean OF correction in full-res pixels.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading RAFT...")
    of_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(args.device).eval()

    scene_dirs = sorted([d for d in args.input_dir.iterdir() if d.is_dir()])
    log.info(f"Found {len(scene_dirs)} scene directories.")

    total_scenes   = 0
    total_passing  = 0
    total_attempts = 0

    for scene_path in tqdm(scene_dirs):
        # Mirror the input subdirectory structure in output
        out_scene_dir = args.output_dir / scene_path.name
        out_scene_dir.mkdir(parents=True, exist_ok=True)

        meta = process_scene(scene_path.name, scene_path, args, of_model)
        if meta is None:
            continue

        meta_path = out_scene_dir / f"{scene_path.name}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

        lq_renders   = [r for r in meta["renders"] if r.get("type") == "LQ"]
        n_passed     = sum(1 for r in lq_renders if r.get("passed"))
        total_scenes   += 1
        total_passing  += n_passed
        total_attempts += len(lq_renders)

    log.info(
        f"Done. {total_scenes} scenes processed. "
        f"{total_passing}/{total_attempts} supporting images passed quality gates."
    )


if __name__ == "__main__":
    main()
