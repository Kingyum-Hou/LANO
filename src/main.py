import torch
import hydra
import pytorch_lightning as pl
import gc
import requests
import os
from omegaconf import OmegaConf

from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import (Optional, List)
import logging
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.utilities import rank_zero_only

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('cfgrib').setLevel(logging.WARNING)
logging.getLogger("fsspec.local").setLevel(logging.WARNING)


def train(cfg: DictConfig) -> Optional[float]:
    # Set seed
    pl.seed_everything(cfg.seed)
    print(f"Current PID: {os.getpid()}")

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
    logger.watch(model, log="all")
    logger.experiment.config.update(
        OmegaConf.to_container(cfg, resolve=True),
        allow_val_change=True
    )

    # Init profiler
    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)
    #trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks, logger=logger, strategy=DDPStrategy(find_unused_parameters=True))
    
    trainer.fit(model=model, datamodule=datamodule)
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
        f"AAAI26_{cfg.datamodule.params_data.name}_{cfg.datamodule.params_data.task}_{cfg.datamodule.params_data.missing_rate}_{cfg.model.params_model.name}_{cfg.tag}/" +
        f"full_loss={testLoss_dict['test/full_loss']:.4f}" +
        f"?sound={cfg.sound}"
    )
 

if __name__ == '__main__':
    main()
