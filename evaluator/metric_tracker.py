import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as lpips
from torchmetrics.image import PeakSignalNoiseRatio as psnr
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim

log = logging.getLogger(__name__)


class MetricTracker:
    def __init__(self, cfg: DictConfig, accelerator: Accelerator):
        self.cfg = cfg
        self.device = accelerator.device
        self.metrics_instances = torch.nn.ModuleDict()

        self._setup_metrics()
        self.reset()

    def _setup_metrics(self):
        """Dynamically initializes metrics based on YAML config."""
        m_cfg = self.cfg.metrics

        if m_cfg.psnr.compute:
            self.metrics_instances["psnr"] = psnr(data_range=1.0).to(self.device)

        if m_cfg.ssim.compute:
            self.metrics_instances["ssim"] = ssim(data_range=1.0).to(self.device)

        if m_cfg.lpips.compute:
            # LPIPS requires a specific network type from config
            self.metrics_instances["lpips"] = lpips(net_type=m_cfg.lpips.net_type).to(self.device)

    def reset(self):
        self.results_history = []  # Stores dicts of per-sample metrics

    def _apply_metric(self, name, sr, hr):
        metric_fn = self.metrics_instances[name]

        if name == "lpips":
            # 1. Scale from [0, 1] to [-1, 1]
            sr_lpips = sr * 2 - 1
            hr_lpips = hr * 2 - 1

            # 2. Convert Grayscale [B, 1, H, W] to Pseudo-RGB [B, 3, H, W]
            if sr_lpips.shape[1] == 1:
                sr_lpips = sr_lpips.repeat(1, 3, 1, 1)
                hr_lpips = hr_lpips.repeat(1, 3, 1, 1)

            return metric_fn(sr_lpips, hr_lpips).item()

        # Default behavior for PSNR, SSIM, etc.
        return metric_fn(sr, hr).item()

    def update(self, sr: torch.Tensor, hr: torch.Tensor):
        sr, hr = sr.to(self.device), hr.to(self.device)

        # Calculate per-sample in the batch
        for i in range(sr.size(0)):
            s, h = sr[i : i + 1], hr[i : i + 1]
            sample_results = {}

            # Dynamically compute only active metrics
            for name in self.metrics_instances.keys():
                sample_results[name] = self._apply_metric(name, s, h)

            self.results_history.append(sample_results)

    def compute_averages(self):
        if not self.results_history:
            return {}

        # Get keys from the first entry to see what was computed
        active_metrics = self.results_history[0].keys()
        return {m.upper(): np.mean([x[m] for x in self.results_history]) for m in active_metrics}

    def get_top_n_by_metric(self, metric_name=None, n=None, descending=True):
        """
        Retrieves top N sample indices sorted by a specific metric.

        Args:
            metric_name: Key in results_history (e.g., 'psnr', 'lpips').
            n: Number of samples to return.
            descending: If True, returns highest values first. If False, lowest first.
        """
        if not self.results_history:
            return []

        # Standardize key access (keys are lowercase in update loop)
        metric_key = metric_name.lower()
        available_metrics = list(self.results_history[0].keys())

        if metric_key not in available_metrics:
            raise ValueError(f"Warning: Metric '{metric_key}' not found in history. Available: {available_metrics}")

        # Extract scores with their original indices
        indexed_scores = enumerate([x[metric_key] for x in self.results_history])

        # Sort based on the config parameters
        sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=descending)

        return [idx for idx, score in sorted_scores[:n]]

    def save_metrics_csv(self, save_dir: str):
        if not self.results_history:
            raise ValueError("No metrics to save or save directory not set.")

        # 1. Save Per-Sample CSV
        df = pd.DataFrame(self.results_history)
        csv_path = os.path.join(save_dir, "per_sample_metrics.csv")
        df.to_csv(csv_path, index_label="index")

        averages = self.compute_averages()
        self._print_final_scores(averages)

        # Add some metadata like total samples
        final_data = {
            "metadata": {
                "total_samples": len(self.results_history),
            },
            "averages": averages,
        }

        json_path = os.path.join(save_dir, "final_metrics.json")
        with open(json_path, "w") as f:
            json.dump(final_data, f, indent=4)

        log.info(f"Results saved to {save_dir}")

    def _print_final_scores(self, averages):
        log.info("Final Evaluation Results:")
        for k, v in averages.items():
            log.info(f"{k}: {v:.4f}")
