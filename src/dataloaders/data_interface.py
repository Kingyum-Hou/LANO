import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from typing import Optional
from dataloaders.ns import get_NS, NSDataset, NSDataset4test
from dataloaders.era5 import get_ERA5, ERA5Dataset, ERA5Dataset4test
from dataloaders.swe import get_SWE, SWEDataset, SWEDataset4test
from dataloaders.diffusion import get_DIFFUSION, DIFFUSIONDataset, DIFFUSIONDataset4test
from dataloaders.elas import get_ELAS, ELASDataset, ELASDataset4test
from dataloaders.hadcurt5 import get_HADCRUT5, HADCRUT5Dataset, HADCRUT5Dataset4test


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
        train_data, test_data, test_data_high = get_NS(self.name, self.data_dir, self.n_train, self.n_test, self.space_size, self.task, self.T_all, self.missing_rate, self.ref)
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
    def __init__(
        self,
        name: str           = "ERA5",
        data_dir: str       = "",
        data_mask_dir: str  = "",
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
        self.data_mask_dir= data_mask_dir
        self.task         = task
        self.missing_rate = missing_rate
        self.n_train, self.n_test = n_train_test
        self.b_train, self.b_test = b_train_test
        self.num_workers  = num_workers
        self.T_all        = T_all
        self.ref          = ref
        self.space_size   = space_size
        self.downsample   = downsample

    def setup(self, stage: Optional[str]=None):
        train_data, test_data, test_data_high = get_ERA5(self.name, self.data_dir, self.data_mask_dir, self.n_train, self.n_test, self.space_size, self.task, self.T_all, self.missing_rate, self.ref, self.downsample)
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


class SWE_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        name: str           = "shallow_water",
        data_dir: str       = "/data/jingren/repository/dataset/PDEBench/2D/shallow-water/2D_rdb_NA_NA.h5",
        task: str           = "task3",
        missing_rate: float = 0.,
        n_train_test: list  = [1000, 100],
        b_train_test: list  = [16, 16],
        num_workers: int    = 4,
        space_size: list    = [64, 64],
        space_dim: int      = 2,
        T_all: int          = 20,
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
        self.downsample   = downsample

    def setup(self, stage: Optional[str]=None):
        train_data, test_data, test_data_high = get_SWE(self.name, self.data_dir, self.n_train, self.n_test, self.space_size, self.task, self.T_all, self.missing_rate, self.ref, self.downsample)
        self.train_data = SWEDataset(*train_data, self.task)
        self.valid_data = SWEDataset(*test_data,  self.task)
        self.test_data  = SWEDataset4test(test_data, test_data_high, self.task)
        del train_data, test_data, test_data_high

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.b_train, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.b_test, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data,  batch_size=self.b_test, num_workers=self.num_workers, shuffle=False)


class DIFFUSION_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        name: str,           
        data_dir: str,    
        task: str           = "task3",
        missing_rate: float = 0.,
        n_train_test: list  = [1000, 100],
        b_train_test: list  = [16, 16],
        num_workers: int    = 4,
        space_size: list    = [64, 64],
        space_dim: int      = 2,
        T_all: int          = 20,
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
        self.downsample   = downsample

    def setup(self, stage: Optional[str]=None):
        train_data, test_data, test_data_high = get_DIFFUSION(self.name, self.data_dir, self.n_train, self.n_test, self.space_size, self.task, self.T_all, self.missing_rate, self.ref, self.downsample)
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
    

class ELAS_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        name: str,           
        data_dir: str,    
        task: str           = "task3",
        missing_rate: float = 0.,
        n_train_test: list  = [1000, 100],
        b_train_test: list  = [16, 16],
        num_workers: int    = 4,
        space_dim: int     = 2,
        space_size: list    = [1, 1],
        downsample: int     = 1,
        T_all: int          = 1,
    ):
        super().__init__()
        self.name         = name
        self.data_dir     = data_dir
        self.task         = task
        self.missing_rate = missing_rate
        self.n_train, self.n_test = n_train_test
        self.b_train, self.b_test = b_train_test
        self.num_workers  = num_workers
        self.space_dim    = space_dim

    def setup(self, stage: Optional[str]=None):
        train_data, test_data, test_data_high = get_ELAS(self.name, self.data_dir, self.n_train, self.n_test, self.space_dim, self.task, self.missing_rate)
        self.train_data = ELASDataset(*train_data, self.task)
        self.valid_data = ELASDataset(*test_data,  self.task)
        self.test_data  = ELASDataset4test(test_data, test_data_high, self.task)
        del train_data, test_data, test_data_high

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.b_train, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.b_test, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data,  batch_size=self.b_test, num_workers=self.num_workers, shuffle=False)


class HADCRUT5_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        name: str           = "ERA5",
        data_dir: str       = "",
        data_dir_mask: str  = "",
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
        self.data_dir_mask= data_dir_mask
        self.task         = task
        self.missing_rate = missing_rate
        self.n_train, self.n_test = n_train_test
        self.b_train, self.b_test = b_train_test
        self.num_workers  = num_workers
        self.T_all        = T_all
        self.ref          = ref
        self.space_size   = space_size
        self.downsample   = downsample

    def setup(self, stage: Optional[str]=None):
        train_data, test_data, test_data_high = get_HADCRUT5(self.name, self.data_dir, self.data_dir_mask, self.n_train, self.n_test, self.space_size, self.task, self.T_all, self.missing_rate, self.ref, self.downsample)
        self.train_data = HADCRUT5Dataset(*train_data, self.task)
        self.valid_data = HADCRUT5Dataset(*test_data,  self.task)
        self.test_data  = HADCRUT5Dataset4test(test_data, test_data_high, self.task)
        del train_data, test_data

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.b_train, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.b_test, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data,  batch_size=self.b_test, num_workers=self.num_workers, shuffle=False)
