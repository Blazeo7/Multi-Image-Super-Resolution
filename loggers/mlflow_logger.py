import mlflow
import mlflow.pytorch as mlflow_pytorch

from .base_logger import BaseLogger


class MLflowLogger(BaseLogger):
    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        tracking_uri: str,
        resume_run_id=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run_id = resume_run_id

    def start(self):
        run = mlflow.start_run(run_name=self.run_name, run_id=self.run_id)
        self.run_id = run.info.run_id
        return self.run_id

    def _log_hyperparameters(self, params: dict):
        mlflow.log_params(params)

    def _log_metrics(self, metrics: dict, step: int):
        mlflow.log_metrics(metrics, step=step)

    def _log_params(self, params: dict):
        mlflow.log_params(params)

    def _log_image(self, image, name: str):
        if not name.endswith((".png", ".jpg", ".jpeg")):
            name = f"{name}.png"
        mlflow.log_image(image, name)

    def _log_figure(self, figure, name: str):
        if not name.endswith((".png", ".jpg", ".jpeg")):
            name = f"{name}.png"
        mlflow.log_figure(figure, name)

    def _log_model(self, model, name: str):
        mlflow_pytorch.log_model(model, name)

    def finish(self):
        if mlflow.active_run():
            mlflow.end_run()
