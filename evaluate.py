import hydra
from omegaconf import DictConfig
from evaluator import Evaluator


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    model = hydra.utils.instantiate(cfg.model)
    dataset = hydra.utils.instantiate(cfg.dataset, split="test")
    ml_logger = hydra.utils.instantiate(cfg.logger)

    evaluator = Evaluator(model, dataset, cfg=cfg.evaluator, logger=ml_logger)
    evaluator.run()


if __name__ == "__main__":
    main()
