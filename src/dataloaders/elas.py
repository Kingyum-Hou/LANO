import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
from tools import reshape2blocks, reshape2data
from einops import rearrange
from scipy.interpolate import griddata
from scipy import io as scio


def add_point_missing(data, num_sampling):
    B, N = data.shape

    valid_data = data.clone()
    valid_mask = torch.ones_like(valid_data)
    for i in range(B):
        indices_addMissing = torch.randperm(N)[:num_sampling]
        valid_data[i, indices_addMissing] = 0.
        valid_mask[i, indices_addMissing] = 0.
    return valid_data, valid_mask


def add_patch_missing(data, missing_rate, space_size, patch_size=4):
    patch_num = [space_size[0]//patch_size, space_size[1]//patch_size]
    total_patches = np.prod(patch_num)
    patch_holes_num = int(np.round(total_patches * missing_rate))
    B = data.shape[0]

    valid_data = reshape2blocks(data, patch_size, patch_num)
    valid_mask = torch.ones_like(valid_data)
    for i in range(B):
        indices_addHoles = torch.randperm(total_patches)[:patch_holes_num]
        valid_data[i, indices_addHoles, ...] = 0.
        valid_mask[i, indices_addHoles, ...] = 0.
    
    valid_data = reshape2data(valid_data, patch_size, patch_num)
    valid_mask = reshape2data(valid_mask, patch_size, patch_num)
    return valid_data, valid_mask


def get_ELAS(
        name, data_dir, num_train, num_test, 
        space_dim, task, missing_rate
    ):
    # load data
    PATH_Sigma = data_dir + '/Random_UnitCell_sigma_10.npy'
    PATH_XY    = data_dir + '/Random_UnitCell_XY_10.npy'
    s = np.load(PATH_Sigma)
    s = torch.tensor(s, dtype=torch.float).permute(1, 0)
    pos = np.load(PATH_XY)
    pos = torch.tensor(pos, dtype=torch.float).permute(2, 0, 1)
    
    train_xy = s[         :num_train]
    test_xy  = s[-num_test:         ]
    train_pos = pos[         :num_train]
    test_pos  = pos[-num_test:         ]
    
    #y_normalizer = UnitTransformer(train_xy)
    #train_xy     = y_normalizer.encode(train_xy)
    

    if missing_rate == 0.05:
        missing_rate_high = 0.25
    elif missing_rate == 0.25:
        missing_rate_high = 0.5
    elif missing_rate == 0.5:
        missing_rate_high = 0.75
    else:
        missing_rate_high = min(0.75, missing_rate*2)

    if task == "task3":
        N = train_xy.shape[1]
        num_sampling      = int(np.round(missing_rate      * N))
        num_sampling_high = int(np.round(missing_rate_high * N))
        # train
        train_xy, train_mask = add_point_missing(train_xy, num_sampling)
        # test
        _, test_mask      = add_point_missing(test_xy, num_sampling)
        # test high
        _, test_mask_high = add_point_missing(test_xy, num_sampling_high)
    elif task == "task4":
        # train
        train_xy, train_mask = add_patch_missing(train_xy, missing_rate, [H, W], patch_size=4)
        train_x = train_xy[..., :10]
        train_y = train_xy[..., 10:]
        # test
        test_xy_, test_mask = add_patch_missing(test_xy, missing_rate, [H, W], patch_size=4)
        test_x  = test_xy_[..., :10]
        test_y  = test_xy [..., 10:]
        # test high
        test_xy_high_, test_mask_high = add_patch_missing(test_xy, missing_rate_high, [H, W], patch_size=4)
        test_x_high   = test_xy_high_[...,   :10]
    else:
        raise NotImplementedError
    return (train_mask, train_pos, train_xy), \
           (test_mask, test_pos, test_xy), \
           (test_mask_high,)


class ELASDataset(Dataset):
    def __init__(self, mask, pos, xy, task):
        self.mask     = mask
        self.pos      = pos
        self.xy       = xy
        self.task     = task
    
    def __len__(self):
        return self.xy .shape[0]

    def __getitem__(self, idx):
        mask = self.mask[idx]
        pos  = self.pos[idx]
        xy   = self.xy[idx]
        return mask, pos, xy, self.task


class ELASDataset4test(Dataset):
    def __init__(self, test_data, test_data_high, task):
        self.mask     = test_data[0]
        self.pos      = test_data[1]
        self.xy       = test_data[2]
        self.mask_high= test_data_high[0]
        self.task     = task

    def __len__(self):
        return self.xy.shape[0]
    
    def __getitem__(self, idx):
        mask    = self.mask   [idx]
        pos     = self.pos    [idx]
        xy      = self.xy     [idx]
        mask_high = self.mask_high[idx]
        return (mask,      pos, xy, self.task), \
               (mask_high, pos, xy, self.task)
