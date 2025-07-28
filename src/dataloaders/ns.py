import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
from tools import get_pos, get_pos_ref, add_patch_missing, add_point_missing
from einops import rearrange
from scipy.interpolate import griddata
from scipy import io as scio


def pad_periodic(data, padding):
    # padding H
    data = torch.concat([data,               data[:, :padding[0], :, :]], dim=1)  # right
    data = torch.concat([data[:, -2*padding[0]:-padding[0], :, :], data], dim=1)  # left
    # padding W
    data = torch.concat([data,               data[:, :, :padding[1], :]], dim=2)  # right
    data = torch.concat([data[:, :, -2*padding[1]:-padding[1], :], data], dim=2)  # left
    return data


def cubicInterp(data, mask):
    padding = [1, 1]
    data = rearrange(data.clone(), 'B (H W) T -> B H W T', H=64, W=64)
    mask = rearrange(mask.clone(), 'B (H W) T -> B H W T', H=64, W=64)

    B, H, W, T = data.shape
    x = torch.arange(64 + 2*padding[0])
    y = torch.arange(64 + 2*padding[0])
    xx, yy = torch.meshgrid(x, y, indexing='xy')

    coordinates_padding = torch.stack([xx, yy], dim=2).unsqueeze(0).repeat(B, 1, 1, 1)
    data_padding        = pad_periodic(data, padding=padding)
    mask_padding        = pad_periodic(mask, padding=padding)

    data_interp = torch.zeros_like(data)
    mask_interp = torch.ones_like(mask)
    for i in range(B):
        coor_padding_i = coordinates_padding[i, ...]
        data_padding_i = data_padding       [i, ...]
        mask_padding_i = mask_padding       [i, ...]
        
        coor_had_i = coor_padding_i[mask_padding_i[..., 0:2]==1.].reshape(-1, 2)
        coor_fit_i = coor_padding_i[mask_padding_i[..., 0:2]==0.].reshape(-1, 2)
        data_had_i = data_padding_i[mask_padding_i          ==1.].reshape(-1, T)

        data_fit_i = griddata(coor_had_i, data_had_i, coor_fit_i, method='cubic', fill_value=0.)
        data_padding_i[mask_padding_i==0.] = torch.from_numpy(data_fit_i).reshape(-1).float()
        data_interp[i, ...] = data_padding_i[padding[0]:padding[0]+H, padding[1]:padding[1]+W, :]
        mask_interpInvalid_i = ((mask_padding_i==0.) & (data_padding_i==np.inf))[padding[0]:padding[0]+H, padding[1]:padding[1]+W, :]
        mask_interp[i, ...][mask_interpInvalid_i] = 0.
    data_interp = rearrange(data_interp, 'B H W T -> B (H W) T')
    return data_interp, mask_interp


def get_NS(
        name, data_dir, num_train, num_test, space_size,
        task, T_all, missing_rate, ref, patch_size
    ):
    # load data
    H, W = space_size[0], space_size[1]
    if   name == "NS_v-5":
        data      = scio.loadmat(data_dir)['u']
        train_au  = torch.tensor(data[:num_train,   ::1, ::1, :T_all], dtype=torch.float).\
            reshape(num_train, -1, T_all)
        test_au   = torch.tensor(data[-num_test:,   ::1, ::1, :T_all], dtype=torch.float).\
            reshape(num_test,  -1, T_all)
    elif name == "NS_v-3":
        data      = h5py.File(data_dir)['u']
        train_au  = torch.tensor(data[:T_all, ::1, ::1, :num_train], dtype=torch.float).transpose(0, 3).reshape(num_train, -1, T_all)
        test_au   = torch.tensor(data[:T_all, ::1, ::1, -num_test:], dtype=torch.float).transpose(0, 3).reshape(num_test,  -1, T_all)
    
    pos       = get_pos(H, W).unsqueeze(0).contiguous()
    train_pos = pos.repeat(num_train, 1, 1, 1).reshape(num_train, -1, 2)
    test_pos  = pos.repeat(num_test,  1, 1, 1).reshape(num_test,  -1, 2)

    pos_ref   = get_pos_ref(H, W, ref).contiguous()
    train_pos_ref = pos_ref.repeat(num_train, 1, 1, 1).reshape(num_train, -1, ref*ref)
    test_pos_ref  = pos_ref.repeat(num_test,  1, 1, 1).reshape(num_test,  -1, ref*ref)

    if missing_rate == 0.0:
        missing_rate_high = 0.05
    elif missing_rate == 0.05:
        missing_rate_high = 0.25
    elif missing_rate == 0.25:
        missing_rate_high = 0.5
    elif missing_rate == 0.5:
        missing_rate_high = 0.75
    else:
        missing_rate_high = min(0.75, missing_rate*2)

    if task == "task0":
        train_mask = torch.ones_like(train_au)
        train_a    = train_au[...,   :10]
        train_u    = train_au[..., 10:  ]
        # test
        test_mask = torch.ones_like(test_au)
        test_a    = test_au[...,   :10]
        test_u    = test_au[..., 10:  ]
        # test high
        test_au_high_, test_mask_high = add_point_missing(test_au, int(np.round(missing_rate_high * 4096)))
        test_a_high   = test_au_high_[...,   :10]
    elif task == "task1":
        num_sampling = int(np.round(missing_rate * 4096))
        # train
        train_au, train_mask = add_point_missing(train_au, num_sampling)
        train_a  = train_au[...,   :10]
        train_u  = train_au[..., 10:  ]
        # test
        test_au_, test_mask = add_point_missing(test_au, num_sampling)
        test_a   = test_au_[...,   :10]
        test_u   = test_au [..., 10:  ]
        # cubic interp for train
        train_u, _ = cubicInterp(train_u, train_mask[..., 10:])
    elif task == "task2":
        # train
        train_au, train_mask = add_patch_missing(train_au, missing_rate, space_size, patch_size=patch_size)
        train_a  = train_au[...,   :10]
        train_u  = train_au[..., 10:  ]
        # test
        test_au_, test_mask = add_patch_missing(test_au, missing_rate, space_size, patch_size=patch_size)
        test_a   = test_au_[...,   :10]
        test_u   = test_au [..., 10:  ]
        # cubic interp for train
        train_u, _ = cubicInterp(train_u, train_mask[..., 10:])
        # test high
        test_au_high_, test_mask_high = add_patch_missing(test_au, missing_rate_high, space_size, patch_size=patch_size)
        test_a_high   = test_au_high_[...,   :10]
    elif task == "task3":
        num_sampling      = int(np.round(missing_rate      * 4096))
        num_sampling_high = int(np.round(missing_rate_high * 4096))
        # train
        train_au, train_mask = add_point_missing(train_au, num_sampling)
        train_a  = train_au[...,   :10]
        train_u  = train_au[..., 10:  ]
        # test
        test_au_, test_mask = add_point_missing(test_au, num_sampling)
        test_a   = test_au_[...,   :10]
        test_u   = test_au [..., 10:  ]
        # test high
        test_au_high_, test_mask_high = add_point_missing(test_au, num_sampling_high)
        test_a_high   = test_au_high_[...,   :10]
    elif task == "task4":
        # train
        train_au, train_mask = add_patch_missing(train_au, missing_rate, space_size, patch_size=patch_size)
        train_a  = train_au[...,   :10]
        train_u  = train_au[..., 10:  ]
        # test
        test_au_, test_mask = add_patch_missing(test_au, missing_rate, space_size, patch_size=patch_size)
        test_a   = test_au_[...,   :10]
        test_u   = test_au [..., 10:  ]
        # test high
        test_au_high_, test_mask_high = add_patch_missing(test_au, missing_rate_high, space_size, patch_size=patch_size)
        test_a_high   = test_au_high_[...,   :10]
    else:
        raise NotImplementedError

    return (train_mask, train_pos, train_a, train_u, train_pos_ref), \
           (test_mask, test_pos, test_a, test_u, test_pos_ref), \
           (test_mask_high, test_a_high)


class NSDataset(Dataset):
    def __init__(self, mask, pos, a, u, pos_ref, task):
        self.mask     = mask
        self.pos      = pos
        self.a        = a
        self.u        = u
        self.pos_ref  = pos_ref
        self.task     = task
    
    def __len__(self):
        return self.a.shape[0]
    
    def __getitem__(self, idx):
        mask    = self.mask   [idx]
        pos     = self.pos    [idx]
        a       = self.a      [idx]
        u       = self.u      [idx]
        pos_ref = self.pos_ref[idx]
        return mask, pos, a, u, pos_ref, self.task


class NSDataset4test(Dataset):
    def __init__(self, test_data, test_data_high, task):
        self.mask     = test_data[0]
        self.pos      = test_data[1]
        self.a        = test_data[2]
        self.u        = test_data[3]
        self.pos_ref  = test_data[4]
        self.mask_high     = test_data_high[0]
        self.a_high        = test_data_high[1]
        self.task     = task

    def __len__(self):
        return self.a.shape[0]
    
    def __getitem__(self, idx):
        mask    = self.mask   [idx]
        pos     = self.pos    [idx]
        a       = self.a      [idx]
        u       = self.u      [idx]
        pos_ref = self.pos_ref[idx]
        mask_high = self.mask_high[idx]
        a_high    = self.a_high[idx]
        return (mask,      pos, a,      u, pos_ref, self.task), \
               (mask_high, pos, a_high, u, pos_ref, self.task)
