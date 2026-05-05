import cv2
import numpy as np
import torch
import os
import json
from torch.utils.data import Dataset


class MISRDataset(Dataset):
    def __init__(self, manifest_path, lr_size=(256, 256), hq_size=(1024, 1024), num_lr_images=8):
        """
        Expects a manifest file in JSON format with the following structure:
            {
                "scale_factor": 2,
                "num_aligned_images": 4,
                "samples": [
                    {
                        "path": "path/to/lr/images_dir",
                        "path_to_metadata": "path/to/metadata.json"
                    },
                    ...
                ]
            }
        """

        with open(manifest_path, "r") as f:
            self.samples = json.load(f)["samples"]
        self.lr_size = lr_size
        self.hq_size = hq_size
        self.num_lr_images = num_lr_images

    def __len__(self):
        return len(self.samples)

    def _get_renders_metadata(self, sample):
        """Loads metadata and handles padding/slicing of render list."""
        meta_path = os.path.join(sample["path"], sample["path_to_metadata"])
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

    def __getitem__(self, idx):
        sample = self.samples[idx]
        renders = self._get_renders_metadata(sample)

        lr_list, hr_hsv = self._process_frames(sample["path"], renders)

        np.random.shuffle(lr_list)
        input_stack = np.stack(lr_list)

        # 3. Convert to Tensors and Normalize
        input_tensor = torch.from_numpy(input_stack).permute(0, 3, 1, 2).float()
        target_tensor = torch.from_numpy(hr_hsv).permute(2, 0, 1).float()

        return self._normalize_hsv(input_tensor), self._normalize_hsv(target_tensor)
