import os
import math
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig


class Visualizer:
    def __init__(self, save_dir: str, cfg: DictConfig):
        self.cfg = cfg
        self.viz_cfg = self.cfg.visualization

        self.viz_dir = os.path.join(save_dir, self.cfg.paths.vizulization_dir)
        os.makedirs(self.viz_dir, exist_ok=True)
        self.all_samples = []

    def add_batch(self, lr_stack, sr, hr):
        for i in range(sr.size(0)):
            self.all_samples.append((lr_stack[i].cpu(), sr[i].cpu(), hr[i].cpu()))

    def save(self, indices, metrics):
        """
        Save visualization grids for selected samples.

        Args:
            indices: Sample indices to visualize.
            metrics: Per-sample dicts with PSNR, SSIM, LPIPS.
        """

        print(f"Saving visualizations to {self.viz_dir}")

        for idx, sample_idx in enumerate(indices):
            if sample_idx >= len(self.all_samples):
                continue

            lr_stack, sr, hr = self.all_samples[sample_idx]
            n_lr_images = lr_stack.shape[0]

            # Create a string of metrics for the title
            m = metrics[sample_idx]
            metric_str = f"PSNR: {m['psnr']:.2f} | SSIM: {m['ssim']:.3f} | LPIPS: {m['lpips']:.3f}"

            # ── choose which LR frames to show ──────────────────────────────────
            n_show = (
                n_lr_images if self.viz_cfg.lr_show is None else min(self.viz_cfg.lr_show, n_lr_images)
            )

            indices_lr = self._get_image_indices(n_lr_images, n_show)
            selected_images = [lr_stack[i] for i in indices_lr]

            # ── grid dimensions: strict 4-column rows ───────────────────────────
            lr_cols = self.viz_cfg.images_in_row
            lr_rows = math.ceil(n_show / lr_cols)
            n_rows = lr_rows + 1  # extra row for SR/HR/error

            fig, axes = plt.subplots(
                n_rows,
                lr_cols,
                figsize=(lr_cols * 4, n_rows * 4),
                squeeze=False,
            )

            # ── LR images ───────────────────────────────────────────────────────
            for pos, images in enumerate(selected_images):
                row, col = divmod(pos, lr_cols)
                ax = axes[row][col]
                ax.imshow(self._to_numpy(images))
                ax.set_title(f"LR frame {indices_lr[pos]}", fontsize=9)
                ax.axis("off")

            # Clean up empty cells in LR rows
            for pos in range(n_show, lr_rows * lr_cols):
                row, col = divmod(pos, lr_cols)
                axes[row][col].set_visible(False)

            # ── bottom row: SR | HR | error heatmap | blank ────────────────────
            sr_np = self._to_numpy(sr)
            hr_np = self._to_numpy(hr)
            err = np.abs(sr_np - hr_np).mean(axis=-1)

            bottom = axes[lr_rows]

            bottom[0].imshow(sr_np)
            bottom[0].set_title("SR Output", fontsize=9)
            bottom[0].axis("off")

            bottom[1].imshow(hr_np)
            bottom[1].set_title("HR Ground Truth", fontsize=9)
            bottom[1].axis("off")

            im = bottom[2].imshow(err, cmap="hot", vmin=0.0, vmax=max(err.max(), 1e-6))
            bottom[2].set_title("Error Map", fontsize=9)
            bottom[2].axis("off")
            plt.colorbar(im, ax=bottom[2], fraction=0.046, pad=0.04)

            bottom[3].set_visible(False)

            fig.suptitle(f"{self.viz_cfg.type.upper()} | Index: {sample_idx}\n{metric_str}", fontsize=12)
            plt.tight_layout()

            filename = (
                f"{self.viz_cfg.order_by_metric}"
                if self.viz_cfg.type == "metric"
                else f"{self.viz_cfg.type}"
            )
            out_path = os.path.join(self.viz_dir, f"{filename}_{idx +1}.png")
            plt.savefig(out_path, bbox_inches="tight", dpi=150)
            plt.close(fig)

    def _get_image_indices(self, n_frames, n_show):
        if self.viz_cfg.lr_select == "random":
            rng = np.random.default_rng(seed=self.cfg.random_seed)
            indices_lr = sorted(rng.choice(n_frames, size=n_show, replace=False))
        elif self.viz_cfg.lr_select == "first":
            indices_lr = list(range(n_show))
        else:
            raise ValueError(f"Unknown LR selection method: {self.viz_cfg.lr_select}")
        return indices_lr

    def _to_numpy(self, tensor):
        """CHW tensor → HWC numpy, clamped to [0, 1]."""
        return tensor.permute(1, 2, 0).numpy().clip(0.0, 1.0)

    def _error_map(self, sr_np, hr_np):
        """Absolute error collapsed to a single-channel heatmap."""
        return np.abs(sr_np - hr_np).mean(axis=-1)
