import hydra
from omegaconf import DictConfig
from evaluator import Evaluator


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    model = hydra.utils.instantiate(cfg.model)
    dataset = hydra.utils.instantiate(cfg.dataset, split="test")

    evaluator = Evaluator(model, dataset, cfg.evaluator)
    evaluator.run()


if __name__ == "__main__":
    main()
