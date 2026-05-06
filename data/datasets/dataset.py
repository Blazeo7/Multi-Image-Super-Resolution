import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import os
import json
from torch.utils.data import Dataset


class MISRDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        scale_factor: Optional[int] = None,
        hq_size: Tuple[int, int] = (1024, 1024),
        num_lr_images: Optional[int] = None,
    ):
        """
        This class loads sample metadata from a JSON manifest. Key parameters
        (num_lr_images, lr_size) can be explicitly passed to the constructor
        to override the defaults stored in the manifest file.

        Expects a manifest file in JSON format with the following structure:
            {
                "scale_factor": 2,
                "num_aligned_images": 4,
                "split_method": "texture",
                "samples": [
                    {
                        "texture_id": "texture_01",
                        "base_dir": "path/to/images_dir",
                        "metadata_path": "metadata.json"
                    },
                    ...
                ]
            }
        """
        self.logger = logging.getLogger("MISRDataset")
        self.manifest = self._load_manifest(manifest_path)
        self.samples = self.manifest.get("samples", [])

        # Setup Hyperparameters
        self.hq_size = hq_size
        self.scale_factor = self._determine_scale(scale_factor)
        self.num_lr_images = self._determine_num_images(num_lr_images)
        self.lr_size = self._calculate_lr_size()

        self._log_configuration()

    def _load_manifest(self, path: str) -> dict:
        with open(path, "r") as f:
            return json.load(f)

    def _determine_scale(self, override_scale: Optional[int]) -> int:
        """Returns override scale if provided, else manifest scale, defaulting to 2."""
        return override_scale if override_scale is not None else self.manifest.get("scale_factor", 2)

    def _determine_num_images(self, override_num: Optional[int]) -> int:
        """Returns override image count if provided, else manifest count."""
        return override_num if override_num is not None else self.manifest.get("num_aligned_images")

    def _calculate_lr_size(self) -> Tuple[int, int]:
        """Calculates symmetric LR resolution based on HQ size and scale factor."""
        h, w = self.hq_size
        return (h // self.scale_factor, w // self.scale_factor)

    def _log_configuration(self):
        config = {
            "dataset_scale_factor": self.scale_factor,
            "dataset_num_lr_images": self.num_lr_images,
            "dataset_hq_size": str(self.hq_size),
            "dataset_lr_size": str(self.lr_size),
        }
        self.logger.info(f"Dataset Configured: {config}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        renders = self._get_renders_metadata(sample)

        lr_list, hr_hsv = self._process_frames(sample["base_dir"], renders)

        np.random.shuffle(lr_list)
        input_stack = np.stack(lr_list)

        input_tensor = torch.from_numpy(input_stack).permute(0, 3, 1, 2).float()
        target_tensor = torch.from_numpy(hr_hsv).permute(2, 0, 1).float()

        return self._normalize_hsv(input_tensor), self._normalize_hsv(target_tensor)

    def _get_renders_metadata(self, sample):
        """Loads metadata and handles padding/slicing of render list."""
        meta_path = os.path.join(sample["base_dir"], sample["metadata_path"])
        with open(meta_path, "r") as f:
            meta = json.load(f)

        renders = meta["renders"]
        available = len(renders)

        if available < self.num_lr_images:
            renders += [renders[0]] * (self.num_lr_images - available)
        return renders[: self.num_lr_images]

    def _align_resize_hsv(self, img_src, H):
        """Warp -> Resize -> HSV -> Mask."""
        # Spatial transformations
        warped_img = cv2.warpPerspective(img_src, H, self.hq_size, flags=cv2.INTER_LINEAR)
        mask = np.full(img_src.shape[:2], 255, dtype=np.uint8)
        warped_mask = cv2.warpPerspective(mask, H, self.hq_size, flags=cv2.INTER_NEAREST)

        # Downsampling
        lr_img = cv2.resize(warped_img, self.lr_size, interpolation=cv2.INTER_AREA)
        lr_mask = cv2.resize(warped_mask, self.lr_size, interpolation=cv2.INTER_AREA)
        _, lr_mask = cv2.threshold(lr_mask, 128, 255, cv2.THRESH_BINARY)

        # Build 4-channel result
        combined = np.empty((*self.lr_size[::-1], 4), dtype=np.uint8)
        combined[..., :3] = cv2.cvtColor(lr_img, cv2.COLOR_BGR2HSV)
        combined[..., 3] = lr_mask
        return combined

    def _process_frames(self, base_dir, renders):
        """Loads and processes HR target and all LR input frames."""
        lr_inputs = []
        hr_target = None

        for i, info in enumerate(renders):
            img_path = os.path.join(base_dir, info["filename"])
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Failed to load: {img_path}")

            H = np.array(info["homography_matrix"], dtype=np.float32)
            lr_inputs.append(self._align_resize_hsv(img, H))

            if i == 0:  # The first render is always the anchor/target
                hr_target = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        return lr_inputs, hr_target

    def _normalize_hsv(self, tensor):
        """Normalizes HSV tensor: H/179, SV/255."""
        tensor[..., 0, :, :] /= 179.0
        tensor[..., 1:, :, :] /= 255.0
        return tensor
