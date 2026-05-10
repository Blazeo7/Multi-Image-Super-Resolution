import logging
import math
import os
import secrets

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from data.datasets.dataset import ColorMode, _make_color_pipeline
from loggers.base_logger import BaseLogger

log = logging.getLogger(__name__)


class Visualizer:
    def __init__(self, save_dir: str, cfg: DictConfig, logger: BaseLogger):
        self.cfg = cfg
        self.viz_cfg = cfg.visualization
        self.logger = logger
        self.color_mode = ColorMode(self.viz_cfg.color_mode)

        _, self._to_rgb, _, _ = _make_color_pipeline(self.color_mode)

        self.viz_dir = os.path.join(save_dir, cfg.paths.vizulization_dir)
        os.makedirs(self.viz_dir, exist_ok=True)
        self.all_samples = []

    def add_batch(self, lr_stack, sr, hr):
        for i in range(sr.size(0)):
            self.all_samples.append((lr_stack[i].cpu(), sr[i].cpu(), hr[i].cpu()))

    def save(self, indices, metrics):
        log.info(f"Saving visualizations to {self.viz_dir}")

        # Determine the primary metric to show first
        primary_metric = self.viz_cfg.get("order_by_metric", "psnr").lower()

        for idx, sample_idx in enumerate(indices):
            if sample_idx >= len(self.all_samples):
                continue

            lr_stack, sr, hr = self.all_samples[sample_idx]

            n_lr_images = lr_stack.shape[0]

            n_show = n_lr_images if self.viz_cfg.lr_show is None else min(self.viz_cfg.lr_show, n_lr_images)
            lr_indices = self._get_lr_indices(n_lr_images, n_show)

            lr_cols = self.viz_cfg.images_in_row
            lr_rows = math.ceil(n_show / lr_cols)

            fig, axes = plt.subplots(
                lr_rows + 1,
                lr_cols,
                figsize=(lr_cols * 4, (lr_rows + 1) * 4),
                squeeze=False,
            )

            for pos, stack_idx in enumerate(lr_indices):
                row, col = divmod(pos, lr_cols)
                axes[row][col].imshow(self._tensor_to_rgb(lr_stack[stack_idx]))
                axes[row][col].set_title(f"LR frame {stack_idx}", fontsize=9)
                axes[row][col].axis("off")

            for pos in range(n_show, lr_rows * lr_cols):
                row, col = divmod(pos, lr_cols)
                axes[row][col].set_visible(False)

            sr_rgb = self._tensor_to_rgb(sr)
            hr_rgb = self._tensor_to_rgb(hr)
            err = np.abs(sr_rgb - hr_rgb).mean(axis=-1)

            bottom = axes[lr_rows]

            bottom[0].imshow(sr_rgb)
            bottom[0].set_title("SR Output", fontsize=9)
            bottom[0].axis("off")

            bottom[1].imshow(hr_rgb)
            bottom[1].set_title("HR Ground Truth", fontsize=9)
            bottom[1].axis("off")

            im = bottom[2].imshow(err, cmap="hot", vmin=0.0, vmax=max(err.max(), 1e-6))
            bottom[2].set_title("Error Map", fontsize=9)
            bottom[2].axis("off")
            plt.colorbar(im, ax=bottom[2], fraction=0.046, pad=0.04)

            bottom[3].set_visible(False)

            m = metrics[sample_idx]
            metric_str = f"PSNR: {m['psnr']:.2f} | SSIM: {m['ssim']:.3f} | LPIPS: {m['lpips']:.3f}"
            fig.suptitle(
                f"{self.viz_cfg.type.upper()} [{self.color_mode.value}] | Index: {sample_idx}\n{metric_str}",
                fontsize=12,
            )
            plt.tight_layout()

            filename = self.viz_cfg.order_by_metric if self.viz_cfg.type == "metric" else self.viz_cfg.type
            out_path = os.path.join(self.viz_dir, f"{filename}_{idx + 1}_{secrets.token_hex(2)}.png")

            self.logger.log_figure(fig, os.path.basename(out_path))
            plt.savefig(out_path, bbox_inches="tight", dpi=150)
            plt.close(fig)

    def _get_lr_indices(self, n_frames: int, n_show: int):
        if self.viz_cfg.lr_select == "random":
            rng = np.random.default_rng(seed=self.cfg.random_seed)
            return sorted(rng.choice(n_frames, size=n_show, replace=False).tolist())
        if self.viz_cfg.lr_select == "first":
            return list(range(n_show))
        raise ValueError(f"Unknown LR selection method: {self.viz_cfg.lr_select}")

    def _tensor_to_rgb(self, tensor) -> np.ndarray:
        hwc = tensor.permute(1, 2, 0).numpy().clip(0.0, 1.0)
        # Tensors are normalized floats; scale back to uint8 range for cv2 conversions
        uint8 = (hwc * 255).astype(np.uint8)
        rgb = self._to_rgb(uint8)
        return rgb.astype(np.float32) / 255.0
