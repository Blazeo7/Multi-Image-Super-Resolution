import logging
import os
import time

import cv2
import lmdb
import numpy as np
import torch
import torchvision.transforms as T
from cnn_descriptor import LocalDescriptor
from torchvision.models.optical_flow import (Raft_Large_Weights,
                                             Raft_Small_Weights, raft_large,
                                             raft_small)


def extract_patch(img: np.ndarray, kp: cv2.KeyPoint, patch_size: int) -> np.ndarray:
    """
    Extracts a patch from the image using the keypoint's scale and orientation.
    The patch is a rotated and scaled crop such that the keypoint's neighborhood
    is normalized to a canonical coordinate system.
    """
    x, y = kp.pt
    scale = kp.size / patch_size

    theta = np.deg2rad(kp.angle)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    M = np.zeros((2, 3), dtype=np.float32)
    M[:, :2] = scale * np.array([[cos_theta, -sin_theta],
                                 [sin_theta, cos_theta]])
    center_patch = np.array([patch_size / 2, patch_size / 2], dtype=np.float32)
    M[:, 2] = np.array([x, y], dtype=np.float32) - M[:, :2].dot(center_patch)

    corners = np.array([[0, 0], [patch_size, 0], [patch_size, patch_size], [0, patch_size]], dtype=np.float32)
    transformed_corners = cv2.transform(np.array([corners]), M)[0]

    h, w = img.shape[:2]
    if (transformed_corners[:, 0].min() < 0 or transformed_corners[:, 0].max() >= w or
            transformed_corners[:, 1].min() < 0 or transformed_corners[:, 1].max() >= h):
        return None

    M_inverse = cv2.invertAffineTransform(M)
    patch = cv2.warpAffine(img, M_inverse, (patch_size, patch_size), flags=cv2.INTER_LINEAR)
    return patch


def compute_homography(img1: np.ndarray, img2: np.ndarray, local_descriptor_model=None) -> np.ndarray:
    """
    Compute the homography matrix that maps points from img1 to img2.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    for sigma in [1.6, 2.5, 3.5, 1.4, 1.2]:
        if local_descriptor_model is None:
            sift = cv2.SIFT_create(20000, 3, sigma=sigma)

            t1 = time.time()
            kp1, des1 = sift.detectAndCompute(gray1, None)
            kp2, des2 = sift.detectAndCompute(gray2, None)
            t2 = time.time()

            desc1_t = torch.from_numpy(des1).to('cuda').float()
            desc2_t = torch.from_numpy(des2).to('cuda').float()
            desc1_t = desc1_t / desc1_t.norm(dim=1)[:, None]
            desc2_t = desc2_t / desc2_t.norm(dim=1)[:, None]
        else:
            t1 = time.time()
            orb = cv2.ORB_create(nfeatures=2000)
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
        match_indices = matches.indices.cpu().numpy()
        match_similarities = matches.values.cpu().numpy()
        for i in range(match_indices.shape[0]):
            if match_similarities[i, 0] * sim_threshold > match_similarities[i, 1]:
                good_matches.append(cv2.DMatch(i, match_indices[i, 0], 1 - match_similarities[i, 0]))
        t3 = time.time()

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        if len(src_pts) > 30:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 15.0, confidence=0.9999)
        else:
            H = None
            mask = np.zeros(1, dtype=np.uint8)
        t4 = time.time()
        logging.info(f"MATCH ransac_matches: {mask.sum()}, good_matches: {len(good_matches)}, kp1: {len(kp1)}, kp2: {len(kp2)} in {t2 - t1:.2f}, {t3 - t1:.2f}, {t4 - t1:.2f} s")
        if mask.sum() > 30:
            break
        else:
            H = None
            if local_descriptor_model is not None:
                break

    return H


def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),
        ]
    )
    batch = transforms(batch)
    return batch


def main():
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser(description='Extract homography patches from two images.')
    parser.add_argument('--images', nargs="+", type=str, required=True, help='Paths to the images to process.')
    parser.add_argument('--patch-size', type=int, default=80, help='Size of the extracted patches.')
    parser.add_argument('--output-path', type=str, default='./output/', help='Path to save the extracted patches.')
    parser.add_argument('--output-prefix', type=str, default='patch', help='Prefix for the output patch filenames.')
    parser.add_argument('--output-lmdb', type=str, help='Path to save the extracted patches.')
    parser.add_argument("--grid-step", type=int, default=32, help="Step size for the grid points.")
    parser.add_argument("--model", type=str, help="Local descriptor model to use.")

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    reference_image = cv2.imread(args.images[0])
    reference_image = cv2.resize(reference_image, (reference_image.shape[1] // 2, reference_image.shape[0] // 2),
                                 interpolation=cv2.INTER_AREA)

    local_descriptor_model = LocalDescriptor(args.model) if args.model else None

    DEVICE = 'cuda'
    of_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(DEVICE)
    of_model = of_model.eval()

    grid_points = np.array([[x, y] for x in range(0, reference_image.shape[1], args.grid_step)
                            for y in range(0, reference_image.shape[0], args.grid_step)], dtype=np.float32)

    # Create a LMDB database to store the patches
    if args.output_lmdb:
        env = lmdb.open(args.output_lmdb, map_size=1024 ** 4)

    # --- Mosaic accumulators ---
    accumulator = np.zeros_like(reference_image, dtype=np.float32)
    weight_map = np.zeros(reference_image.shape[:2], dtype=np.float32)

    # Add the reference image itself first
    ref_mask = (reference_image > 0).any(axis=2).astype(np.float32)
    accumulator += reference_image.astype(np.float32) * ref_mask[:, :, np.newaxis]
    weight_map += ref_mask

    for image_id, img_path in enumerate(args.images[1:]):
        img2 = cv2.imread(img_path)
        img2 = cv2.resize(img2, (img2.shape[1] // 2, img2.shape[0] // 2), interpolation=cv2.INTER_AREA)

        H = compute_homography(img2, reference_image, local_descriptor_model)

        if H is None:
            logging.info(f"Failed to compute homography for {img_path}")
            continue

        # Map image 2 to image 1 using the computed homography.
        img2_warped = cv2.warpPerspective(img2, H, reference_image.shape[:2][::-1])
        of_images = [reference_image, img2_warped]

        of_scale = 0.5
        of_images = [cv2.resize(img, (int(img.shape[1] * of_scale), int(img.shape[0] * of_scale))) for img in of_images]

        img = np.stack(of_images, 0)
        img_torch = preprocess(torch.from_numpy(np.transpose(img, (0, 3, 1, 2)) / 255.0)).to(DEVICE)
        img0_torch = img_torch[0:1]
        img1_torch = img_torch[1:2]
        with torch.no_grad():
            h, w = img0_torch.shape[-2:]
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            img0_torch = torch.nn.functional.pad(img0_torch, (0, pad_w, 0, pad_h))
            img1_torch = torch.nn.functional.pad(img1_torch, (0, pad_w, 0, pad_h))
            predicted_flows = of_model(img0_torch, img1_torch, num_flow_updates=8)
            predicted_flows = predicted_flows[-1][0]
            predicted_flows = predicted_flows[:, :h, :w]
            predicted_flows = predicted_flows.cpu().numpy()

        flow_x = predicted_flows[0, :, :] / of_scale
        flow_y = predicted_flows[1, :, :] / of_scale
        flow_x = cv2.resize(flow_x, (reference_image.shape[1], reference_image.shape[0]))
        flow_y = cv2.resize(flow_y, (reference_image.shape[1], reference_image.shape[0]))

        map_x = (np.arange(0, reference_image.shape[1], 1)[np.newaxis, :] + flow_x).astype(np.float32)
        map_y = (np.arange(0, reference_image.shape[0], 1)[:, np.newaxis] + flow_y).astype(np.float32)
        img2_warped_of = cv2.remap(img2_warped, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        # Transform the grid points using the predicted flow and homography into the second image
        grid_points_2 = np.stack(
            [map_x[grid_points[:, 1].astype(int), grid_points[:, 0].astype(int)],
             map_y[grid_points[:, 1].astype(int), grid_points[:, 0].astype(int)]], axis=1)
        H_inv = np.linalg.inv(H)
        grid_points_2 = cv2.perspectiveTransform(grid_points_2[np.newaxis, :, :], H_inv)[0]

        # Crop patches from img2 and save to LMDB
        if args.output_lmdb:
            txn = env.begin(write=True)
            patch_size = args.patch_size
            for i, (x, y) in enumerate(grid_points_2):
                x1 = int(x + 0.5) - patch_size // 2
                y1 = int(y + 0.5) - patch_size // 2
                x2 = x1 + patch_size
                y2 = y1 + patch_size
                if x1 < 0 or y1 < 0 or x2 >= img2.shape[1] or y2 >= img2.shape[0]:
                    continue
                patch = img2[y1:y2, x1:x2]
                output_file = f'{args.output_prefix}-{i:04d}_{image_id:02d}.jpg'
                patch_img = cv2.imencode('.jpg', patch, [int(cv2.IMWRITE_JPEG_QUALITY), 98])[1].tobytes()
                txn.put(output_file.encode(), patch_img)
            txn.commit()
            logging.info(f"Saved patches for {img_path}, {len(grid_points_2)} patches")

        # --- Accumulate into mosaic ---
        mask = (img2_warped_of > 0).any(axis=2).astype(np.float32)
        accumulator += img2_warped_of.astype(np.float32) * mask[:, :, np.newaxis]
        weight_map += mask

        logging.info(f"Accumulated image {image_id + 1}/{len(args.images) - 1}: {img_path}")

    # --- Save final mosaic ---
    weight_map = np.maximum(weight_map, 1)
    mosaic = (accumulator / weight_map[:, :, np.newaxis]).astype(np.uint8)
    cv2.imwrite('mosaic_final.jpg', mosaic)
    logging.info("Saved final mosaic to mosaic_final.jpg")


if __name__ == '__main__':
    main()
