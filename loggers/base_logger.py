import logging
from abc import ABC, abstractmethod


class BaseLogger(ABC):
    def __init__(self, logger_name="BaseLogger", accelerator=None):
        self.local_logger = logging.getLogger(logger_name)
        self.accelerator = accelerator

    def setup_local_logger(self, logger_name):
        self.local_logger = logging.getLogger(logger_name)

    def info(self, message):
        self.local_logger.info(message)

    def warning(self, message):
        self.local_logger.warning(message)

    def error(self, message):
        self.local_logger.error(message)

    def start(self):
        """Initiliaze the logger"""
        pass

    def log_metrics(self, metrics: dict, step: int):
        """Log a dictionary of metrics at a specific step."""
        if self.accelerator is None or self.accelerator.is_main_process:
            self._log_metrics(metrics, step)

    def log_hyperparameters(self, params: dict):
        if self.accelerator is None or self.accelerator.is_main_process:
            self._log_hyperparameters(params)

    def log_image(self, image, name: str):
        if self.accelerator is None or self.accelerator.is_main_process:
            self._log_image(image, name)

    def log_figure(self, figure, name: str):
        if self.accelerator is None or self.accelerator.is_main_process:
            self._log_figure(figure, name)

    def log_model(self, model, name: str):
        if self.accelerator is None or self.accelerator.is_main_process:
            self._log_model(model, name)

    @abstractmethod
    def _log_metrics(self, metrics: dict, step: int):
        """Log a dictionary of metrics at a specific step."""
        pass

    @abstractmethod
    def _log_hyperparameters(self, params: dict):
        pass

    @abstractmethod
    def _log_figure(self, figure, name: str):
        pass

    @abstractmethod
    def _log_image(self, image, name: str):
        pass

    @abstractmethod
    def _log_model(self, model, name: str):
        pass

    def finish(self):
        pass
