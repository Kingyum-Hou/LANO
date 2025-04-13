import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "2"

import torch
import numpy as np
from torch.utils.data import Dataset
from tools import get_pos, get_pos_ref, reshape2blocks, reshape2data
import xarray as xr


def add_point_missing(data, num_sampling):
    B, HW, _   = data.shape
    valid_data = data.clone()
    valid_mask = torch.ones_like(valid_data)
    for i in range(B):
        indices_addMissing = torch.randperm(HW)[:num_sampling]
        valid_data[i, indices_addMissing, ...] = 0.
        valid_mask[i, indices_addMissing, ...] = 0.
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


def get_ERA5(
        name, data_dir, num_train, num_test, space_size,
        task, T_all, missing_rate, ref, downsample
    ):
    H, W = space_size[0], space_size[1]
    # load_data
    ds = xr.open_dataset(data_dir)
    data = torch.tensor(np.array(ds["t"]))
    h = int((H / downsample))
    w = int((W / downsample))
    Tn = 7 * int(data.shape[0] / 7)
    data = data[:, :720, :]
    data = data[:, ::downsample, ::downsample]
    data_list = []
    for i in range(0, data.shape[0]-14, 7):
        data_list.append(data[i:i+14, ...])
    data = torch.stack(data_list, dim=0).permute(0, 2, 3, 1)
    
    # train & test
    train_xy = data[:num_train, ...].reshape(num_train, -1, 14)
    test_xy  = data[-num_test:, ...].reshape(num_test,  -1, 14)
    
    pos = get_pos(h, w).unsqueeze(0).contiguous()
    train_pos = pos.repeat(num_train, 1, 1, 1).reshape(num_train, -1, 2)
    test_pos  = pos.repeat(num_test,  1, 1, 1).reshape(num_test, -1, 2)

    pos_ref = get_pos_ref(h, w, ref).contiguous()
    train_pos_ref = pos_ref.repeat(num_train, 1, 1, 1).reshape(num_train, -1, ref*ref)
    test_pos_ref  = pos_ref.repeat(num_test,  1, 1, 1).reshape(num_test,  -1, ref*ref)

    if task == "task3":
        num_sampling = int(np.round(missing_rate * h * w))
        # train
        train_xy, train_mask = add_point_missing(train_xy, num_sampling)
        train_x = train_xy[...,  :7]
        train_y = train_xy[..., 7: ]
        # test
        test_xy_, test_mask = add_point_missing(test_xy, num_sampling)
        test_x  = test_xy_[...,  :7]
        test_y  = test_xy[..., 7: ]
    elif task == "task4":
        # train
        train_xy, train_mask = add_patch_missing(train_xy, missing_rate, (h, w), patch_size=3)
        train_x = train_xy[...,  :7]
        train_y = train_xy[..., 7: ]
        # test
        test_xy_, test_mask = add_patch_missing(test_xy, missing_rate, (h, w), patch_size=3)
        test_x = test_xy_[...,  :7]
        test_y = test_xy [..., 7: ]
    else:
        raise NotImplementedError
    return (train_mask, train_pos, train_x, train_y, train_pos_ref), (test_mask, test_pos, test_x, test_y, test_pos_ref)


class ERA5Dataset(Dataset):
    def __init__(self, mask, pos, x, y, pos_ref, task, is_train):
        self.mask = mask
        self.pos  = pos
        self.x    = x
        self.y    = y
        self.pos_ref = pos_ref
        self.task = task
        self.is_train = is_train

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        mask = self.mask[idx]
        pos  = self.pos[idx]   
        x    = self.x[idx] 
        y    = self.y[idx]
        pos_ref = self.pos_ref[idx]
        return mask, pos, x, y, pos_ref, self.task
