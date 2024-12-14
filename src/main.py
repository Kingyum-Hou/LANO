import torch
import hydra
import pytorch_lightning as pl
import gc
import requests

from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import (Optional, List)


def train(cfg: DictConfig) -> Optional[float]:
    # Set seed
    pl.seed_everything(cfg.seed)

    # Set precision
    torch.set_float32_matmul_precision('high')

    # Init Lightning datamodule
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)
    datamodule.prepare_data()
    
    # Init Lightning model
    model: pl.LightningDataModule = instantiate(cfg.model)

    # Init callbacks
    callbacks: List[pl.Callback] = []
    for _, cfg_callback in cfg.callback.items():
        if "_target_" in cfg_callback:
            callbacks.append(instantiate(cfg_callback))

    # Init logger
    for _, cfg_logger in cfg.logger.items():
        if "_target_" in cfg_logger:
            logger: pl.loggers.LightningLoggerBase = instantiate(cfg_logger)
    # Init trainer
    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)
    trainer.fit (model=model, datamodule=datamodule)
    testLoss_dict = trainer.test(model=model, datamodule=datamodule, ckpt_path="best")[0]

    logger.experiment.finish()

    del datamodule, model, callbacks, trainer, logger
    gc.collect()
    torch.cuda.empty_cache()
    return testLoss_dict


@hydra.main(version_base="1.2", config_path="configs", config_name="default")
def main(cfg: DictConfig) -> Optional[float]:
    testLoss_dict = train(cfg)
    requests.get(
        cfg.bark_url +
        f"ANOT_{cfg.model.params_model.name}_{cfg.tag}/" +
        f"full_loss={testLoss_dict['test/loss']:.4f}" +
        f"?sound={cfg.sound}"
    )
    

if __name__ == '__main__':
    main()
