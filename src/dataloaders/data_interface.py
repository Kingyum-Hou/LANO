import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from typing import Optional
from dataloaders.ns import get_NS, NSDataset


class NS_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        name: str           = "NS_v-3",
        data_dir: str       = "",
        task: str           = "task1",
        missing_rate: float = 0.,
        n_train_test: list  = [128, 32],
        b_train_test: list  = [16, 16],
        num_workers: int    = 4,
        space_size: list    = [64, 64],
        space_dim: int      = 2,
        T_all: int          = 50,
        downsample: int     = 1,
        ref: int            = 64,
    ):
        super().__init__()
        self.name         = name
        self.data_dir     = data_dir
        self.task         = task
        self.missing_rate = missing_rate
        self.n_train, self.n_test = n_train_test
        self.b_train, self.b_test = b_train_test
        self.num_workers  = num_workers
        self.T_all        = T_all
        self.ref          = ref
        self.space_size   = space_size

    def setup(self, stage: Optional[str]=None):
        train_data, test_data = get_NS(self.name, self.data_dir, self.n_train, self.n_test, self.space_size, self.task, self.T_all, self.missing_rate, self.ref)
        self.train_data = NSDataset(*train_data, self.task, is_train=True)
        self.test_data  = NSDataset(*test_data,  self.task, is_train=False)
        del train_data, test_data

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.b_train, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.b_test, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.b_test, num_workers=self.num_workers, shuffle=False)
