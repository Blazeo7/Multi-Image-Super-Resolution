import accelerate
import hydra
import torch
from omegaconf import DictConfig
from safetensors.torch import load_file

from evaluator import Evaluator


@hydra.main(version_base=None, config_path="configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    """
    Main function to evaluate a model on a test dataset.
    """
    accelerator = accelerate.Accelerator()
    model = hydra.utils.instantiate(cfg.model)
    state_dict = load_file(cfg.checkpoint_path)
    model.load_state_dict(state_dict)

    dataset = hydra.utils.instantiate(cfg.dataset, split="test")
    dataloader = hydra.utils.instantiate(cfg.dataloader, dataset=dataset)
    # pass accelerator to logger for distributed logging
    ml_logger = hydra.utils.instantiate(cfg.logger, accelerator=accelerator)

    model, dataset, dataloader = accelerator.prepare(model, dataset, dataloader)

    evaluator = Evaluator(model, dataset, cfg=cfg.evaluator, logger=ml_logger, loader=dataloader)
    evaluator.run()


if __name__ == "__main__":
    main()
