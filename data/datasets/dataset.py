import json
import logging
import os
import random
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MISRDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        scale_factor: int = 4,
        hq_size: Tuple[int, int] = (1024, 1024),
        num_lr_images: int = 4,
        augment_cfg: Optional[dict] = None,
        samples_per_scene: int = 5,
    ):
        """
        This class loads sample metadata from a JSON manifest. Key parameters
        (num_lr_images, lr_size) can be explicitly passed to the constructor
        to override the defaults stored in the manifest file.
        Expects a manifest file in JSON format with the following structure:
            {
                "scale_factor": 2,
                "num_aligned_images": 4,
                "data_dir": "dataset/",
                "samples": [
                    {
                        "texture_name": "acg_grass_01",
                        "scenes": ["metadata1.json", ...]
                    },
                    ...
                ]
            }
        """
        self.scale_factor = scale_factor
        self.num_lr_images = num_lr_images
        self.hq_size = hq_size
        self.lr_size = self._calculate_lr_size()

        self.logger = logging.getLogger("MISRDataset")
        self.manifest = self._load_manifest(manifest_path)

        self.data_root = self.manifest.get("data_dir", "")
        texture_groups = self.manifest.get("samples", [])

        self.samples = self._build_sample_list(texture_groups, self.data_root, samples_per_scene)
        self.photo_transform = self._build_augmentations(augment_cfg)

        self._log_configuration()

    def __len__(self):
        return len(self.samples)

    def _build_sample_list(self, texture_groups: List[dict], data_root: str, samples_per_scene: int) -> List[tuple]:
        """For each scene, generates set of N HR-LR sets and stores the paths in a shuffled list."""
        samples = []

        for group in texture_groups:
            texture_name = group["texture_name"]

            for scene_meta in group["scenes"]:
                renders = self._get_renders_metadata(scene_meta, os.path.join(data_root, texture_name))
                hr_candidates = renders.copy()

                for _ in range(samples_per_scene):
                    hr = random.choice(hr_candidates)
                    hr_candidates.remove(hr)

                    lr_candidates = [r for r in renders if r != hr]
                    lrs = random.sample(lr_candidates, min(self.num_lr_images - 1, len(lr_candidates)))

                    hr["filename"] = os.path.join(data_root, texture_name, hr["filename"])
                    for lr in lrs:
                        lr["filename"] = os.path.join(data_root, texture_name, lr["filename"])

                    samples.append((hr, lrs))

        print(f"Total samples generated: {len(samples)}")
        random.shuffle(samples)
        return samples

    def __getitem__(self, idx):
        hr, lrs = self.samples[idx]

        lr_list, lr_masks, hr_hsv = self._process_frames(hr, lrs)
        lr_list = self._apply_photometric_augmentations(lr_list, lr_masks)

        shuffle_idx = np.random.permutation(len(lr_list))
        lr_list = [lr_list[i] for i in shuffle_idx]
        lr_masks = [lr_masks[i] for i in shuffle_idx]

        lr_stack = np.stack(lr_list)
        mask_stack = np.stack(lr_masks)

        lr_hsv = torch.from_numpy(lr_stack).permute(0, 3, 1, 2).float()
        target_tensor = torch.from_numpy(hr_hsv).permute(2, 0, 1).float()
        lr_masks_tensor = torch.from_numpy(mask_stack) < 128

        return self._normalize_hsv(lr_hsv), self._normalize_hsv(target_tensor), lr_masks_tensor

    def _load_manifest(self, path: str) -> dict:
        with open(path, "r") as f:
            return json.load(f)

    def _determine_scale(self, override_scale: Optional[int]) -> int:
        """Returns override scale if provided, else manifest scale, defaulting to 2."""
        return override_scale if override_scale is not None else self.manifest.get("scale_factor", 2)

    def _calculate_lr_size(self) -> Tuple[int, int]:
        """Calculates symmetric LR resolution based on HQ size and scale factor."""
        h, w = self.hq_size
        return (h // self.scale_factor, w // self.scale_factor)

    def _log_configuration(self):
        config = {
            "dataset_scale_factor": self.scale_factor,
            "dataset_num_lr_images": self.num_lr_images,
            "dataset_hq_size": str(self.hq_size),
            "data_root": self.data_root,
        }
        self.logger.info(f"Dataset Configured: {config}")

    def _get_renders_metadata(self, metadata: str, base_dir: str) -> List[dict]:
        """Loads metadata and handles padding/slicing of render list."""
        meta_path = os.path.join(base_dir, metadata)
        with open(meta_path, "r") as f:
            meta = json.load(f)

        renders = meta["renders"]
        available = len(renders)

        if available < self.num_lr_images:
            renders += [renders[0]] * (self.num_lr_images - available)
        return renders[: self.num_lr_images]

    def _align_resize_hsv(self, img_src, H) -> Tuple[np.ndarray, np.ndarray]:
        """Warp -> Resize -> HSV, returns (hsv, mask) separately."""
        warped_img = cv2.warpPerspective(
            img_src, H, self.hq_size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

        mask = np.full(img_src.shape[:2], 255, dtype=np.uint8)
        warped_mask = cv2.warpPerspective(mask, H, self.hq_size, flags=cv2.INTER_NEAREST)

        lr_img = cv2.resize(warped_img, self.lr_size, interpolation=cv2.INTER_AREA)
        lr_mask = cv2.resize(warped_mask, self.lr_size, interpolation=cv2.INTER_AREA)
        _, lr_mask = cv2.threshold(lr_mask, 128, 255, cv2.THRESH_BINARY)

        return cv2.cvtColor(lr_img, cv2.COLOR_BGR2HSV), lr_mask

    def _process_frames(self, hr, lrs) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """Loads and processes HR target and all LR input frames."""
        ref_img = cv2.imread(hr["filename"])
        if ref_img is None:
            raise FileNotFoundError(f"Reference image not found: {hr['filename']}")
        hr_target = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)

        H_ref = np.array(hr["homography_matrix"], dtype=np.float32)
        try:
            H_ref_inv = np.linalg.inv(H_ref)
        except np.linalg.LinAlgError:
            H_ref_inv = np.eye(3)

        lr_list, mask_list = [], []

        hsv, mask = self._align_resize_hsv(ref_img, np.eye(3, dtype=np.float32))
        lr_list.append(hsv)
        mask_list.append(mask)

        for lr in lrs:
            img = cv2.imread(lr["filename"])
            if img is None:
                continue

            H_i = np.array(lr["homography_matrix"], dtype=np.float32)
            H_warp = self._compute_homography(H_ref_inv, H_i)

            hsv, mask = self._align_resize_hsv(img, H_warp)
            lr_list.append(hsv)
            mask_list.append(mask)

        return lr_list, mask_list, hr_target

    def _compute_homography(self, H_ref_inv, H_i):
        H_warp = H_ref_inv @ H_i
        if H_warp[2, 2] != 0:
            H_warp /= H_warp[2, 2]
        return H_warp

    def _normalize_hsv(self, tensor):
        """Normalizes HSV tensor: H/179, SV/255."""
        tensor[..., 0, :, :] /= 179.0
        tensor[..., 1:, :, :] /= 255.0
        return tensor

    def _build_augmentations(self, cfg: Optional[dict]) -> Optional[A.Compose]:
        if not cfg:
            return None

        extra_targets = {f"image_{i}": "image" for i in range(1, self.num_lr_images)}
        transforms = [getattr(A, n)(**p) for n, p in cfg.items() if hasattr(A, n)]

        return A.Compose(transforms, additional_targets=extra_targets) if transforms else None

    def _apply_photometric_augmentations(
        self, lr_list: List[np.ndarray], lr_masks: List[np.ndarray]
    ) -> List[np.ndarray]:
        if self.photo_transform is None or not lr_list:
            return lr_list

        aug_input = {}
        for i, hsv in enumerate(lr_list):
            key = "image" if i == 0 else f"image_{i}"
            aug_input[key] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        main_image = aug_input.pop("image")
        augmented_output = self.photo_transform(image=main_image, **aug_input)

        final_lrs = []
        for i, mask in enumerate(lr_masks):
            key = "image" if i == 0 else f"image_{i}"
            aug_rgb = augmented_output[key]

            aug_hsv = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2HSV)
            normalized_mask = (mask > 128).astype(np.uint8)[..., np.newaxis]
            final_lrs.append(aug_hsv * normalized_mask)

        return final_lrs
