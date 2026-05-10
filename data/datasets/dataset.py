import json
import logging
import os
import random
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class ColorMode(str, Enum):
    RGB = "rgb"
    HSV = "hsv"
    YCBCR = "ycbcr"
    LAB = "lab"
    GRAY = "gray"


def _make_color_pipeline(mode: ColorMode) -> Tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
    Callable[[torch.Tensor], torch.Tensor],
]:
    if mode == ColorMode.RGB:
        return (
            lambda bgr: cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            lambda img: img,
            lambda img: img,
            lambda t: t / 255.0,
        )

    if mode == ColorMode.HSV:
        return (
            lambda bgr: cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV),
            lambda hsv: cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB),
            lambda rgb: cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV),
            _normalize_hsv,
        )

    if mode == ColorMode.YCBCR:

        def bgr_to_ycbcr(bgr: np.ndarray) -> np.ndarray:
            ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
            return ycc[:, :, [0, 2, 1]]

        def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
            ycc = ycbcr[:, :, [0, 2, 1]]
            return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)

        def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
            ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
            return ycc[:, :, [0, 2, 1]]

        return (bgr_to_ycbcr, ycbcr_to_rgb, rgb_to_ycbcr, lambda t: t / 255.0)

    if mode == ColorMode.LAB:
        return (
            lambda bgr: cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab),
            lambda lab: cv2.cvtColor(lab, cv2.COLOR_Lab2RGB),
            lambda rgb: cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab),
            lambda t: t / 255.0,
        )

    if mode == ColorMode.GRAY:

        def bgr_to_gray(bgr: np.ndarray) -> np.ndarray:
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            return g[:, :, np.newaxis]

        def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
            return cv2.cvtColor(gray[:, :, 0], cv2.COLOR_GRAY2RGB)

        def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
            g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            return g[:, :, np.newaxis]

        return (bgr_to_gray, gray_to_rgb, rgb_to_gray, lambda t: t / 255.0)

    raise ValueError(f"Unsupported ColorMode: {mode!r}")


def _normalize_hsv(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.clone()
    tensor[..., 0, :, :] = tensor[..., 0, :, :] / 179.0
    tensor[..., 1:, :, :] = tensor[..., 1:, :, :] / 255.0
    return tensor

def estimate_sharpness(img_path: str) -> float:
    img = cv2.imread(img_path)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    img_var = gray.var()
    if img_var < 1e-6:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_32F).var() / img_var)


class MISRDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        scale_factor: int = 4,
        hq_size: Tuple[int, int] = (1024, 1024),
        num_lr_images: int = 4,
        color_mode: Union[ColorMode, str] = ColorMode.HSV,
        augment_cfg: Optional[dict] = None,
        samples_per_scene: int = 5,
        alignment_corruption_coeff: float = 0.0,
        alignment_corruption_p: float = 0.0,
        exclusive_hr=False,
        random_hr=True,
    ):
        self.scale_factor = scale_factor
        self.num_lr_images = num_lr_images
        self.hq_size = hq_size
        self.lr_size = self._calculate_lr_size()

        # This converts "rgb" -> ColorMode.RGB or keeps ColorMode.RGB as is
        self.color_mode = ColorMode(color_mode)

        (
            self._to_space,
            self._to_rgb,
            self._from_rgb,
            self._normalize,
        ) = _make_color_pipeline(self.color_mode)

        self.exclusive_hr = exclusive_hr
        self.random_hr = random_hr
        self.logger = logging.getLogger("MISRDataset")
        self.manifest = self._load_manifest(manifest_path)
        self.data_root = self.manifest.get("data_dir", "")
        self.meta_root = self.manifest.get("meta_dir", self.data_root)
        texture_groups = self.manifest.get("samples", [])
        self.samples = self._build_sample_list(texture_groups, self.data_root, samples_per_scene)
        self.photo_transform = self._build_augmentations(augment_cfg)
        self._log_configuration()
        self.h_corr_coeff = alignment_corruption_coeff
        self.h_corr_p = alignment_corruption_p

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        hr_meta, lrs_meta = self.samples[idx]
        lr_list, lr_masks, hr_img = self._process_frames(hr_meta, lrs_meta)

        while len(lr_list) < self.num_lr_images:
            lr_list.append(np.zeros_like(lr_list[0]))
            lr_masks.append(np.ones_like(lr_masks[0]))

        lr_list = self._apply_photometric_augmentations(lr_list, lr_masks)


        # shuffle_idx = np.random.permutation(len(lr_list))
        # lr_list = [lr_list[i] for i in shuffle_idx]
        # lr_masks = [lr_masks[i] for i in shuffle_idx]

        lr_stack = np.stack(lr_list)
        mask_stack = np.stack(lr_masks)

        lr_tensor = torch.from_numpy(lr_stack).permute(0, 3, 1, 2).float()
        hr_tensor = torch.from_numpy(hr_img).permute(2, 0, 1).float()

        if self.color_mode == ColorMode.GRAY:
            hr_tensor = hr_tensor.unsqueeze(0) if hr_tensor.ndim == 2 else hr_tensor

        lr_masks_tensor = torch.from_numpy(mask_stack) < 128

        return (
            self._normalize(lr_tensor),
            self._normalize(hr_tensor),
            lr_masks_tensor,
        )

    def _load_manifest(self, path: str) -> dict:
        with open(path, "r") as f:
            return json.load(f)

    def _calculate_lr_size(self) -> Tuple[int, int]:
        h, w = self.hq_size
        return (h // self.scale_factor, w // self.scale_factor)

    def _log_configuration(self):
        config = {
            "color_mode": self.color_mode.value,
            "scale_factor": self.scale_factor,
            "num_lr_images": self.num_lr_images,
            "hq_size": str(self.hq_size),
            "lr_size": str(self.lr_size),
            "data_root": self.data_root,
            "total_samples": len(self.samples),
        }
        self.logger.info(f"Dataset configured: {config}")

    def _build_sample_list(
        self,
        texture_groups: List[dict],
        data_root: str,
        samples_per_scene: int,
    ) -> List[tuple]:
        samples = []

        for group in tqdm(texture_groups, desc="Building scenes..."):
            texture_name = group["texture_name"]

            for scene_meta in group["scenes"]:
                base_dir = os.path.join(self.meta_root, texture_name)
                renders  = self._get_renders_metadata(scene_meta, base_dir)

                renders = [r for r in renders if r.get("passed", True)]
                if not renders:
                    continue

                if self.exclusive_hr:
                    hr_candidates = [r for r in renders if r.get("type") == "HQ"]
                    n_samples = len(hr_candidates)
                else:
                    hr_candidates = renders.copy()

                    if not self.random_hr:
                        hr_candidates.sort(
                            key=lambda r: r.get("quality", {}).get("ncc_after_flow", 0.0),
                        )
                    n_samples = samples_per_scene

                if not hr_candidates:
                    self.logger.warning(f"Scene {scene_meta}: no HR candidates, skipping.")
                    continue

                for _ in range(n_samples):
                    if not hr_candidates:
                        break

                    hr = hr_candidates[0] if not self.random_hr else random.choice(hr_candidates)
                    hr_candidates.remove(hr)

                    lr_candidates = [r for r in renders if r != hr]
                    lrs = random.sample(
                        lr_candidates,
                        min(self.num_lr_images - 1, len(lr_candidates)),
                    )

                    hr  = {**hr,  "filename": os.path.join(data_root, texture_name, hr["filename"])}
                    lrs = [{**lr, "filename": os.path.join(data_root, texture_name, lr["filename"])}
                           for lr in lrs]

                    samples.append((hr, lrs))

        random.shuffle(samples)
        return samples

    def _get_renders_metadata(self, metadata: str, base_dir: str) -> List[dict]:
        meta_path = os.path.join(base_dir, metadata)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        renders = meta["renders"]
        return renders

    def _align_resize_to_space(self, img_bgr: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        warped = cv2.warpPerspective(
            img_bgr,
            H,
            self.hq_size,
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        mask_src = np.full(img_bgr.shape[:2], 255, dtype=np.uint8)
        warped_mask = cv2.warpPerspective(mask_src, H, self.hq_size, flags=cv2.INTER_NEAREST)
        lr_img = cv2.resize(warped, self.lr_size, interpolation=cv2.INTER_AREA)
        lr_mask = cv2.resize(warped_mask, self.lr_size, interpolation=cv2.INTER_AREA)
        _, lr_mask = cv2.threshold(lr_mask, 128, 255, cv2.THRESH_BINARY)
        return self._to_space(lr_img), lr_mask

    def corrupt_homography(self, H: np.ndarray, noise_scale: float, image_size) -> np.ndarray:
        if np.random.random() > self.h_corr_p:
            return H

        H = H.copy()
        w, h = image_size
        cx, cy = w / 2, h / 2
        tx = np.random.normal(0, noise_scale)
        ty = np.random.normal(0, noise_scale)
        T_to = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
        T_fr = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float32)
        P = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
        Delta = T_fr @ P @ T_to
        H_noisy = Delta @ H
        if H_noisy[2, 2] != 0:
            H_noisy = H_noisy / H_noisy[2, 2]
        return H_noisy

    def _process_frames(self, hr: dict, lrs: List[dict]) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        ref_bgr = cv2.imread(hr["filename"])
        if ref_bgr is None:
            raise FileNotFoundError(f"Reference image not found: {hr['filename']}")
        hr_img = self._to_space(ref_bgr)
        H_ref = np.array(hr["homography_matrix"], dtype=np.float32)
        try:
            H_ref_inv = np.linalg.inv(H_ref)
        except np.linalg.LinAlgError:
            H_ref_inv = np.eye(3, dtype=np.float32)
        lr_list, mask_list = [], []
        img_space, mask = self._align_resize_to_space(ref_bgr, np.eye(3, dtype=np.float32))
        lr_list.append(img_space)
        mask_list.append(mask)
        for lr in lrs:
            img_bgr = cv2.imread(lr["filename"])
            if img_bgr is None:
                continue
            H_i = np.array(lr["homography_matrix"], dtype=np.float32)
            H_warp = _compute_relative_homography(H_ref_inv, H_i)

            H_warp = self.corrupt_homography(H_warp, self.h_corr_coeff, img_bgr.shape[1::-1])

            img_space, mask = self._align_resize_to_space(img_bgr, H_warp)
            lr_list.append(img_space)
            mask_list.append(mask)
        return lr_list, mask_list, hr_img

    def _build_augmentations(self, cfg: Optional[dict]) -> Optional[A.Compose]:
        if not cfg:
            return None
        extra_targets = {f"image_{i}": "image" for i in range(1, self.num_lr_images)}
        transforms = [getattr(A, name)(**params) for name, params in cfg.items() if hasattr(A, name)]
        return A.Compose(transforms, additional_targets=extra_targets) if transforms else None

    def _apply_photometric_augmentations(
        self, lr_list: List[np.ndarray], lr_masks: List[np.ndarray]
    ) -> List[np.ndarray]:
        if self.photo_transform is None or not lr_list:
            return lr_list
        aug_input: dict = {}
        for i, img in enumerate(lr_list):
            key = "image" if i == 0 else f"image_{i}"
            aug_input[key] = self._to_rgb(img)
        main_image = aug_input.pop("image")
        augmented = self.photo_transform(image=main_image, **aug_input)
        final_lrs = []
        for i, mask in enumerate(lr_masks):
            key = "image" if i == 0 else f"image_{i}"
            aug_rgb = augmented[key]
            aug_img = self._from_rgb(aug_rgb)
            valid_mask = (mask > 128).astype(np.uint8)[..., np.newaxis]
            final_lrs.append(aug_img * valid_mask)
        return final_lrs


def _compute_relative_homography(H_ref_inv: np.ndarray, H_i: np.ndarray) -> np.ndarray:
    H = H_ref_inv @ H_i
    if H[2, 2] != 0:
        H = H / H[2, 2]
    return H
