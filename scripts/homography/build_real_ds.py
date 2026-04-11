import os

# Threading constraints for high-performance clusters
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

import argparse
import json
import logging
import random
import shutil
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

def alignment_ncc(img_aligned, img_ref):
    gray_a = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_r = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mask = (gray_a > 0) & (gray_r > 0)
    if mask.sum() < 1000: return 0.0
    a, r = gray_a[mask], gray_r[mask]
    a -= a.mean(); r -= r.mean()
    denom = np.sqrt((a ** 2).sum() * (r ** 2).sum())
    return float(np.dot(a, r) / denom) if denom > 1e-6 else 0.0

def compute_homography(img1, img2, device="cuda"):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(8000, 3, sigma=1.6)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None: return None

    # Brute force matching with ratio test
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) > 20:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H if mask.sum() > 20 else None
    return None

def optical_flow_refine(img_warped, reference, of_model, device, scale=0.5):
    h, w = reference.shape[:2]
    img_stack = np.stack([
        cv2.resize(reference, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA),
        cv2.resize(img_warped, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    ])
    
    # Preprocessing
    inp = torch.from_numpy(img_stack.transpose(0, 3, 1, 2)).float().to(device)
    inp = (inp / 127.5) - 1.0 # RAFT normalization
    
    with torch.no_grad():
        pad_h, pad_w = (8 - inp.shape[-2] % 8) % 8, (8 - inp.shape[-1] % 8) % 8
        inp = torch.nn.functional.pad(inp, (0, pad_w, 0, pad_h))
        flow = of_model(inp[0:1], inp[1:2], num_flow_updates=12)[-1][0][:, :h, :w].cpu().numpy()

    flow_x = cv2.resize(flow[0] / scale, (w, h))
    flow_y = cv2.resize(flow[1] / scale, (w, h))
    mx = (np.arange(w).reshape(1, -1) + flow_x).astype(np.float32)
    my = (np.arange(h).reshape(-1, 1) + flow_y).astype(np.float32)
    return cv2.remap(img_warped, mx, my, interpolation=cv2.INTER_LANCZOS4)

def process_scene_folder(scene_id, scene_path, output_dir, args, of_model):
    # Find all JPGs (using JPG as base, assuming DNGs are sidecars)
    image_files = sorted(list(scene_path.glob("*.jpg")))
    if not image_files: return []

    lr_dir = output_dir / "lr"
    hr_dir = output_dir / "hr"
    lr_dir.mkdir(exist_ok=True); hr_dir.mkdir(exist_ok=True)

    scene_samples = []
    
    # Create X samples from this one directory
    for i in range(args.samples_per_dir):
        # 1. Select random image as Target/Reference
        target_path = random.choice(image_files)
        other_files = [f for f in image_files if f != target_path]
        
        # Select N supporting images
        if len(other_files) < args.n_supporting:
            log.warning(f"Scene {scene_id} has insufficient images for {args.n_supporting} supporting.")
            continue
        supporting_srcs = random.sample(other_files, args.n_supporting)
        
        sample_key = f"{scene_id}_s{i:02d}_{target_path.stem}"
        
        # 2. Create HR Target and LR Reference
        full_img = cv2.imread(str(target_path))
        if full_img is None: continue

        # Best downsampling: INTER_AREA for downscaling
        hr_target = cv2.resize(full_img, (args.hr_res, args.hr_res), interpolation=cv2.INTER_AREA)
        lr_ref = cv2.resize(full_img, (args.lr_res, args.lr_res), interpolation=cv2.INTER_AREA)
        
        hr_path = hr_dir / f"{sample_key}_HR.png"
        lr_ref_path = lr_dir / f"{sample_key}_ref_LR.png"
        cv2.imwrite(str(hr_path), hr_target)
        cv2.imwrite(str(lr_ref_path), lr_ref)

        # 3. Align Supporting Images
        aligned_paths = []
        for s_idx, s_src in enumerate(supporting_srcs):
            s_img_raw = cv2.imread(str(s_src))
            s_img = cv2.resize(s_img_raw, (args.lr_res, args.lr_res), interpolation=cv2.INTER_AREA)
            
            H = compute_homography(s_img, lr_ref, device=args.device)
            if H is None: continue
            
            warped = cv2.warpPerspective(s_img, H, (args.lr_res, args.lr_res))
            
            # Refine
            try:
                refined = optical_flow_refine(warped, lr_ref, of_model, args.device, args.of_scale)
            except:
                refined = warped
                
            ncc = alignment_ncc(refined, lr_ref)
            if ncc > args.ncc_threshold:
                out_name = f"{sample_key}_supp_{s_idx:02d}.png"
                cv2.imwrite(str(lr_dir / out_name), refined)
                aligned_paths.append(str(Path("lr") / out_name))

        if len(aligned_paths) >= args.min_supporting:
            scene_samples.append({
                "id": sample_key,
                "reference": str(Path("lr") / lr_ref_path.name),
                "supporting": aligned_paths,
                "target": str(Path("hr") / hr_path.name)
            })
            
    return scene_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-dir", type=int, default=3, help="How many times to reuse a scene with diff references")
    parser.add_argument("--n-supporting", type=int, default=5, help="Attempt to find this many images")
    parser.add_argument("--min-supporting", type=int, default=3, help="Minimum needed to keep the sample")
    parser.add_argument("--hr-res", type=int, default=1024)
    parser.add_argument("--lr-res", type=int, default=512)
    parser.add_argument("--ncc-threshold", type=float, default=0.85)
    parser.add_argument("--of-scale", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("Loading RAFT...")
    of_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(args.device).eval()

    # Get all numbered subdirectories
    scene_dirs = sorted([d for d in args.input_dir.iterdir() if d.is_dir()])
    
    all_manifest_entries = []
    for scene_path in tqdm(scene_dirs):
        entries = process_scene_folder(scene_path.name, scene_path, args.output_dir, args, of_model)
        all_manifest_entries.extend(entries)

    manifest = {
        "scale_factor": args.hr_res // args.lr_res,
        "samples": all_manifest_entries
    }
    
    with open(args.output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
    
    log.info(f"Done. Generated {len(all_manifest_entries)} samples.")

if __name__ == "__main__":
    main()
