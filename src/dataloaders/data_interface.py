import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

from torch.utils.data import DataLoader
from typing import Optional
from dataloaders.ns import get_NS, NSDataset, NSDataset4test
from dataloaders.era5 import get_ERA5, ERA5Dataset, ERA5Dataset4test
from dataloaders.swe import get_SWE, SWEDataset, SWEDataset4test
from dataloaders.diffusion import get_DIFFUSION, DIFFUSIONDataset, DIFFUSIONDataset4test
from dataloaders.elas import get_ELAS, ELASDataset, ELASDataset4test
from dataloaders.hadcrut5 import get_HADCRUT5, HADCRUT5Dataset, HADCRUT5Dataset4test


class NS_DataModule(pl.LightningDataModule):
    def __init__(self, params_data: DictConfig):
        super().__init__()
        self.name         = params_data.name
        self.data_dir     = params_data.data_dir
        self.task         = params_data.task
        self.missing_rate = params_data.missing_rate
        self.n_train, self.n_test = params_data.n_train_test
        self.b_train, self.b_test = params_data.b_train_test
        self.num_workers  = params_data.num_workers
        self.T_all        = params_data.T_all
        self.ref          = params_data.ref
        self.space_size   = params_data.space_size
        self.patch_size   = params_data.patch_size
        self.patch_num    = params_data.patch_num

    def setup(self, stage: Optional[str]=None):
        train_data, test_data, test_data_high = get_NS(self.name, self.data_dir, self.n_train, self.n_test, self.space_size, 
                                                       self.task, self.T_all, self.missing_rate, self.ref, self.patch_size)
        self.train_data  = NSDataset(*train_data, self.task)
        self.valid_data  = NSDataset(*test_data,  self.task)
        self.test_data   = NSDataset4test(test_data, test_data_high, self.task)
        del train_data, test_data, test_data_high

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.b_train, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.b_test, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data,  batch_size=self.b_test, num_workers=self.num_workers, shuffle=False)


class ERA5_DataModule(pl.LightningDataModule):
    def __init__(self, params_data: DictConfig):
        super().__init__()
        self.name         = params_data.name
        self.data_dir     = params_data.data_dir
        self.np_data_dir  = params_data.np_data_dir
        self.data_mask_dir= params_data.data_mask_dir
        self.task         = params_data.task
        self.missing_rate = params_data.missing_rate
        self.n_train, self.n_test = params_data.n_train_test
        self.b_train, self.b_test = params_data.b_train_test
        self.num_workers  = params_data.num_workers
        self.T_all        = params_data.T_all
        self.ref          = params_data.ref
        self.space_size   = params_data.space_size
        self.downsample   = params_data.downsample

    def setup(self, stage: Optional[str]=None):
        train_data, test_data, test_data_high = get_ERA5(self.name, self.data_dir, self.np_data_dir, self.data_mask_dir, self.n_train, self.n_test, self.space_size, self.task, self.T_all, self.missing_rate, self.ref, self.downsample)
        self.train_data = ERA5Dataset(*train_data, self.task)
        self.valid_data = ERA5Dataset(*test_data,  self.task)
        self.test_data  = ERA5Dataset4test(test_data, test_data_high, self.task)
        del train_data, test_data

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.b_train, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.b_test, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data,  batch_size=self.b_test, num_workers=self.num_workers, shuffle=False)


class DIFFUSION_DataModule(pl.LightningDataModule):
    def __init__(self, params_data: DictConfig):
        super().__init__()
        self.name         = params_data.name
        self.data_dir     = params_data.data_dir
        self.np_data_dir  = params_data.np_data_dir
        self.task         = params_data.task
        self.missing_rate = params_data.missing_rate
        self.n_train, self.n_test = params_data.n_train_test
        self.b_train, self.b_test = params_data.b_train_test
        self.num_workers  = params_data.num_workers
        self.T_all        = params_data.T_all
        self.ref          = params_data.ref
        self.space_size   = params_data.space_size
        self.downsample   = params_data.downsample

    def setup(self, stage: Optional[str]=None):
        train_data, test_data, test_data_high = get_DIFFUSION(self.name, self.data_dir, self.np_data_dir, self.n_train, self.n_test, self.space_size, self.task, self.T_all, self.missing_rate, self.ref, self.downsample)
        self.train_data = DIFFUSIONDataset(*train_data, self.task)
        self.valid_data = DIFFUSIONDataset(*test_data,  self.task)
        self.test_data  = DIFFUSIONDataset4test(test_data, test_data_high, self.task)
        del train_data, test_data, test_data_high

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.b_train, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.b_test, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data,  batch_size=self.b_test, num_workers=self.num_workers, shuffle=False)
    