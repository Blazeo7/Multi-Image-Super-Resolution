from datetime import datetime as dt
import os
import numpy as np
from omegaconf import DictConfig
from torch import nn
import torch
from torch.utils.data import DataLoader

from evaluator import InferenceEngine, MetricTracker, Visualizer
from tqdm import tqdm


class Evaluator:
    def __init__(self, model: nn.Module, dataset: torch.utils.data.Dataset, cfg: DictConfig):
        self.cfg = cfg
        self.setup_logging_directory(model, self.cfg.paths.base_save_dir)

        self.loader = DataLoader(dataset, batch_size=self.cfg.hardware.batch_size, shuffle=False)
        self.viz_count = self.cfg.visualization.num_samples

        self.inference = InferenceEngine(model, cfg)
        self.metrics = MetricTracker(cfg)
        self.visualizer = Visualizer(self.model_eval_dir, cfg)

    def setup_logging_directory(self, model: nn.Module, save_dir: str):
        model_name = model.__class__.__name__
        timestamp = dt.now().strftime("%m%d_%H%M")
        self.model_eval_dir = os.path.join(save_dir, f"{model_name}_{timestamp}")
        os.makedirs(self.model_eval_dir)

    def run(self):
        self.metrics.reset()

        # Progress Bar
        pbar = tqdm(self.loader, desc="Evaluating", unit="batch")


        for batch in pbar:
            lr_stack, hr = batch
            sr = self.inference.forward(lr_stack)

            self.metrics.update(sr, hr)
            self.visualizer.add_batch(lr_stack, sr, hr)

            # Update pbar postfix with running average
            avg_metrics = self.metrics.compute_averages()
            pbar.set_postfix(PSNR=f"{avg_metrics['PSNR']:.2f}")

        self.metrics.save_metrics_csv(self.model_eval_dir)

        # Visualizations
        indices = self._get_indices_by_viz_type()
        metrics_history = self.metrics.results_history
        self.visualizer.save(indices, metrics_history)

        return avg_metrics

    def _get_indices_by_viz_type(self):
        if self.cfg.visualization.type == "metric":
            return self.metrics.get_top_n_by_metric(
                self.cfg.visualization.order_by_metric,
                n=self.viz_count,
                descending=self.cfg.visualization.order_descending,
            )
        elif self.cfg.visualization.type == "random":
            total_samples = len(self.metrics.results_history)
            rng = np.random.default_rng(seed=42)
            return sorted(rng.choice(total_samples, size=self.viz_count, replace=False))
        elif self.cfg.visualization.type == "in_order":
            return list(range(min(self.viz_count, len(self.metrics.results_history))))
        else:
            raise ValueError(f"Unknown visualization type: {self.cfg.visualization.type}")
