import torch
from scipy import io as scio
from model_interface import get_model
from tools import torch2dgrid
from dataloaders.ns import add_point_missing, add_patch_missing
import numpy as np
import matplotlib.pyplot as plt
from tools import LpLoss
import math
import h5py
from einops import rearrange

# env
missing_rate = 0.25
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
alpha = 1e-5


class ModelParams():
    def __init__(self):
        self.name = 'OFormer'
        self.in_channels = 12
        self.encoder_emb_dim = 96
        self.out_seq_emb_dim = 192
        self.encoder_heads = 1
        self.encoder_depth = 5
        self.decoder_emb_dim = 384
        self.out_channels = 1
        self.out_step = 1
        self.propagator_depth = 1
        self.fourier_frequency = 8
        self.ntrain = 500
        self.ntest = 100
        self.r = 64

model = get_model(ModelParams()).to(device)
model_path = 'save/SWE/OFormer/task3_mr=5/best-v4.ckpt'
#model_path = 'save/SWE/OFormer/task3_mr=25/best-v1.ckpt'
#model_path = 'save/SWE/OFormer/task3_mr=50/best-v1.ckpt'
#model_path = 'save/SWE/OFormer/task4_mr=5/best-v3.ckpt'
#model_path = 'save/SWE/OFormer/task4_mr=25/best-v2.ckpt'
#model_path = 'save/SWE/OFormer/task4_mr=50/best.ckpt'
model_dict = torch.load(model_path, map_location=device)['state_dict']
model_dict = {k.replace('model.', ''): v for k, v in model_dict.items()}
print(model.load_state_dict(model_dict, strict=False))

model.eval()
test_aPos_had = torch.rand(1, 4096, 12).to(device)
test_pos_had  = torch.rand(1, 4096,  2).to(device)
test_pos_pred = torch.rand(1, 4096,  2).to(device)
test_To       = 10
test_pred = model(test_aPos_had, test_pos_had, test_pos_pred, test_To)


# load data
data_path = '/data/jingren/repository/dataset/PDEBench/2D/shallow-water/2D_rdb_NA_NA.h5'
H, W = 128, 128
num_train, num_test = 500, 100
downsample = 2
T_all = 20
with h5py.File(data_path, 'r') as f:
    keys = list(f.keys())
    keys.sort()
    data_arrays = [
        np.array(f[key]["data"], dtype=np.float32) for key in keys
    ]
    _data = torch.from_numpy(
        np.stack(data_arrays, axis=0)
    )  # [B, nt, nx, ny, nc]
    B, T, H, W, C = _data.shape
    _data = _data.permute(0, 2, 3, 1, 4).reshape(B, H, W, T)
    train_xy = _data[:num_train, ::downsample, ::downsample, ::5][..., :T_all]
    test_xy  = _data[-num_test:, ::downsample, ::downsample, ::5][..., :T_all]
test_xy = test_xy.reshape(num_test, -1, T_all)
H, W = int(H//downsample), int(W//downsample)
test_y = test_xy[..., 10:]
#test_xy_withMissing, test_mask = add_point_missing(test_xy, int(np.round(4096*missing_rate)))
test_xy_withMissing, test_mask = add_patch_missing(test_xy, missing_rate, [64, 64], patch_size=4)
test_x = test_xy_withMissing[..., :10]

pos = torch2dgrid(64, 64).unsqueeze(0).contiguous()
test_pos = pos.repeat(num_test, 1, 1, 1).reshape(num_test, -1, 2)

test_data = [test_mask, test_pos, test_x, test_y]
test_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*test_data), batch_size=10, shuffle=False)

# eval
loss_func = LpLoss(size_average=False)
loss = 0.
plot_flag = True
with torch.no_grad():
    for batch_idx, batch in enumerate(test_dataloader):
        mask, pos, a, u = [x.to(device) for x in batch]
        B, _, Ti = a.shape
        _, _, To = u.shape
        a_had    = a  [mask[..., :Ti].bool()].reshape(B, -1, Ti)
        pos_had  = pos[mask[..., : 2].bool()].reshape(B, -1,  2)
        pos_pred = pos
        aPos_had = torch.concat([a_had, pos_had], dim=-1)
        pred     = model(aPos_had, pos_had, pos_pred, To)
        loss    += loss_func(pred.reshape(B, -1), u.reshape(B, -1))
        if plot_flag:
            # Plot the predictions
            fig, axes = plt.subplots(3, 10, figsize=(40, 12))
            yy = u.reshape(10, 64, 64, 10)
            pred = pred.reshape(10, 64, 64, 10)
            error = torch.abs(pred - yy)
            # ground truth
            for i in range(10):
                axes[0, i].imshow(yy[0, ..., i].cpu().numpy(), cmap='viridis')
                axes[0, i].set_title(f'Time {i+1}_gt')
                axes[0, i].axis('off')
            # pred
            for i in range(10):
                axes[1, i].imshow(pred[0, ..., i].cpu().numpy(), cmap='viridis')
                axes[1, i].set_title(f'Time {i+1}_pred')
                axes[1, i].axis('off')
            # error
            for i in range(10):
                axes[2, i].imshow(error[0, ..., i].cpu().numpy(), cmap='viridis')
                axes[2, i].set_title(f'Time {i+1}_error')
                axes[2, i].axis('off')
            plt.show()
            plt.savefig(f'imgs/OFormer_swe_{batch_idx}.png')
            plt.clf()
            #plot_flag = False
print('ok!')
print(f"full loss = {loss/num_test}")
