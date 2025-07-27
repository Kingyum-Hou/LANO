import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
from tools import get_pos, get_pos_ref, add_point_missing, add_patch_missing
from einops import rearrange
from scipy.interpolate import griddata
from scipy import io as scio


def get_DIFFUSION(
        name, data_dir, np_data_dir, num_train, num_test, space_size,
        task, T_all, missing_rate, ref, downsample
    ):
    # load data
    H, W = space_size[0], space_size[1]
    if np_data_dir is not None:
        _data = np.load(np_data_dir)
        _data = torch.tensor(_data, dtype=torch.float32)
        B, T, H, W, C = 1000, 101, 128, 128, 2
        train_xy = _data[:num_train]
        test_xy  = _data[-num_test:]
    else:
        with h5py.File(data_dir, 'r') as f:
            keys = list(f.keys())
            keys.sort()
            data_arrays = [
                np.array(f[key]["data"], dtype=np.float32) for key in keys
            ]
            _data = torch.from_numpy(
                np.stack(data_arrays, axis=0)
            )  # [B, nt, nx, ny, nc]
            B, T, H, W, C = _data.shape
            _data = _data[..., 0].permute(0, 2, 3, 1).reshape(B, H, W, T)  # for activator
            _data = _data[..., 50:]
            train_xy = _data[:num_train, ::downsample, ::downsample, ::2][..., :T_all]
            test_xy  = _data[-num_test:, ::downsample, ::downsample, ::2][..., :T_all]
    train_xy = train_xy.reshape(num_train, -1, T_all)
    test_xy  = test_xy. reshape(num_test,  -1, T_all)
    H, W = int(H//downsample), int(W//downsample)
    pos       = get_pos(H, W).unsqueeze(0).contiguous()
    train_pos = pos.repeat(num_train, 1, 1, 1).reshape(num_train, -1, 2)
    test_pos  = pos.repeat(num_test,  1, 1, 1).reshape(num_test,  -1, 2)

    pos_ref   = get_pos_ref(H, W, ref).contiguous()
    train_pos_ref = pos_ref.repeat(num_train, 1, 1, 1).reshape(num_train, -1, ref*ref)
    test_pos_ref  = pos_ref.repeat(num_test,  1, 1, 1).reshape(num_test,  -1, ref*ref)

    if missing_rate == 0.05:
        missing_rate_high = 0.25
    elif missing_rate == 0.25:
        missing_rate_high = 0.5
    elif missing_rate == 0.5:
        missing_rate_high = 0.75
    else:
        missing_rate_high = min(0.75, missing_rate*2)

    if task == "task3":
        num_sampling      = int(np.round(missing_rate      * H * W))
        num_sampling_high = int(np.round(missing_rate_high * H * W))
        # train
        train_xy, train_mask = add_point_missing(train_xy, num_sampling)
        train_x = train_xy[...,   :10]
        train_y = train_xy[..., 10:  ]
        # test
        test_xy_, test_mask = add_point_missing(test_xy, num_sampling)
        test_x  = test_xy_[...,   :10]
        test_y  = test_xy [..., 10:  ]
        # test high
        test_xy_high_, test_mask_high = add_point_missing(test_xy, num_sampling_high)
        test_x_high   = test_xy_high_[...,   :10]
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
    return (train_mask, train_pos, train_x, train_y, train_pos_ref), \
           (test_mask, test_pos, test_x, test_y, test_pos_ref), \
           (test_mask_high, test_x_high)


class DIFFUSIONDataset(Dataset):
    def __init__(self, mask, pos, x, y, pos_ref, task):
        self.mask     = mask
        self.pos      = pos
        self.x        = x
        self.y        = y
        self.pos_ref  = pos_ref
        self.task     = task
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        mask = self.mask[idx]
        pos  = self.pos[idx]
        x    = self.x[idx]
        y    = self.y[idx]
        pos_ref = self.pos_ref[idx]
        return mask, pos, x, y, pos_ref, self.task


class DIFFUSIONDataset4test(Dataset):
    def __init__(self, test_data, test_data_high, task):
        self.mask     = test_data[0]
        self.pos      = test_data[1]
        self.x        = test_data[2]
        self.y        = test_data[3]
        self.pos_ref  = test_data[4]
        self.mask_high     = test_data_high[0]
        self.x_high        = test_data_high[1]
        self.task     = task

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        mask    = self.mask   [idx]
        pos     = self.pos    [idx]
        x       = self.x      [idx]
        y       = self.y      [idx]
        pos_ref = self.pos_ref[idx]
        mask_high = self.mask_high[idx]
        x_high    = self.x_high[idx]
        return (mask,      pos, x,      y, pos_ref, self.task), \
               (mask_high, pos, x_high, y, pos_ref, self.task)