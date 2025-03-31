import torch
from scipy import io as scio
from model_interface import get_model
from tools import torch2dgrid, reshape2blocks, reshape2data
from dataloaders.ns import add_point_missing, add_patch_missing
import numpy as np
import matplotlib.pyplot as plt
from tools import LpLoss
import math
from einops import rearrange
import xarray as xr

# env
missing_rate = 0.75
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
alpha = 1e-5


"""
class ModelParams():
    def __init__(self):
        self.name = 'OFormer'
        self.in_channels = 12
        self.encoder_emb_dim = 64
        self.out_seq_emb_dim = 128
        self.encoder_heads = 1
        self.encoder_depth = 5
        self.decoder_emb_dim = 256
        self.out_channels = 1
        self.out_step = 1
        self.propagator_depth = 1
        self.fourier_frequency = 8
        self.ntrain = 1000
        self.ntest = 200
        self.r = 64
"""


class ModelParams():
    def __init__(self):
        self.name = 'OFormer'
        self.in_channels = 9
        self.encoder_emb_dim = 96
        self.out_seq_emb_dim = 192
        self.encoder_heads = 1
        self.encoder_depth = 5
        self.decoder_emb_dim = 384
        self.out_channels = 1
        self.out_step = 1
        self.propagator_depth = 1
        self.fourier_frequency = 8
        self.ntrain = 1000
        self.ntest = 200
        self.r = 64

model = get_model(ModelParams()).to(device)
#model_path = 'save/NS/OFormer/task3_mr=5/best-v1.ckpt'
#model_path = 'save/NS/OFormer/task3_mr=25/best-v1.ckpt'
#model_path = 'save/NS/OFormer/task3_mr=50/best.ckpt'
#model_path = 'save/NS/OFormer/task4_mr=5/best-v2.ckpt'
#model_path = 'save/NS/OFormer/task4_mr=25/best-v2.ckpt'
#model_path = 'save/NS/OFormer/task4_mr=50/best-v1.ckpt'
#model_path = 'save/ERA5/OFormer/task3_mr=5/best-v4.ckpt'
#model_path = 'save/ERA5/OFormer/task3_mr=25/best-v5.ckpt'
model_path = 'save/ERA5/OFormer/task3_mr=50/best-v3.ckpt'

model_dict = torch.load(model_path, map_location=device)['state_dict']
model_dict = {k.replace('model.', ''): v for k, v in model_dict.items()}
print(model.load_state_dict(model_dict, strict=False))

model.eval()
test_aPos_had = torch.rand(1, 16200, 9).to(device)
test_pos_had  = torch.rand(1, 16200,  2).to(device)
test_pos_pred = torch.rand(1, 16200,  2).to(device)
test_To       = 7
test_pred = model(test_aPos_had, test_pos_had, test_pos_pred, test_To)


# load data
data_path = '/data/jingren/repository/dataset/WeatherBench/era5_temperature.grib'
H, W = 720, 1440
downsample = 8
num_train = 300
num_test  = 50
ds = xr.open_dataset(data_path)
data = torch.tensor(np.array(ds["t"]))
h = int((H/downsample))
w = int((W/downsample))
Tn = 7 * int(data.shape[0] / 7)
data = data[:, :720, :]
data = data[:, ::downsample, ::downsample]

data_list = []
for i in range(0, data.shape[0]-14, 7):
    data_list.append(data[i:i+14, ...])
data = torch.stack(data_list, dim=0).permute(0, 2, 3, 1)

# train & test
test_xy  = data[-num_test:, ...].reshape(num_test,  -1, 14)
test_xy_withMissing, test_mask = add_point_missing(test_xy, int(np.round(h*w*missing_rate)))
#test_au_withMissing, test_mask = add_patch_missing(test_xy, missing_rate, [h, w], patch_size=3)
test_x = test_xy_withMissing[..., :7]
test_y = test_xy            [..., 7:]

# pos
pos = torch2dgrid(h, w, bot=(-0.5, 0), top=(0.5, 2)).unsqueeze(0).contiguous()
test_pos = pos.repeat(num_test, 1, 1, 1).reshape(num_test, -1, 2)

# combine data
batch_size = 10
test_data = [test_mask, test_pos, test_x, test_y]
test_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*test_data), batch_size=batch_size, shuffle=False)

# eval
loss_func = LpLoss(size_average=False)
loss = 0.
plot_flag = True
with torch.no_grad():
    for batch_idx, batch in enumerate(test_dataloader):
        mask, pos, x, y = [x.to(device) for x in batch]
        B, _, Ti = x.shape
        _, _, To = y.shape
        x_had    = x  [mask[..., :Ti].bool()].reshape(B, -1, Ti)
        pos_had  = pos[mask[..., : 2].bool()].reshape(B, -1,  2)
        pos_pred = pos
        xPos_had = torch.concat([x_had, pos_had], dim=-1)
        pred     = model(xPos_had, pos_had, pos_pred, To)
        loss    += loss_func(pred.reshape(B, -1), y.reshape(B, -1))
        if plot_flag:
            # Plot the predictions
            fig, axes = plt.subplots(3, 7, figsize=(56, 12))
            yy = y.reshape(batch_size, h, w, 7)
            pred = pred.reshape(batch_size, h, w, 7)
            error = torch.abs(pred - yy)
            # ground truth
            for i in range(7):
                axes[0, i].imshow(yy[0, ..., i].cpu().numpy(), cmap='viridis')
                axes[0, i].set_title(f'Time {i+1}_gt')
                axes[0, i].axis('off')
            # pred
            for i in range(7):
                axes[1, i].imshow(pred[0, ..., i].cpu().numpy(), cmap='viridis')
                axes[1, i].set_title(f'Time {i+1}_pred')
                axes[1, i].axis('off')
            # error
            for i in range(7):
                axes[2, i].imshow(error[0, ..., i].cpu().numpy(), cmap='viridis')
                axes[2, i].set_title(f'Time {i+1}_error')
                axes[2, i].axis('off')
            plt.show()
            plt.savefig(f'imgs/OFormer_{batch_idx}.png')
            plt.clf()
            #plot_flag = False
print('ok!')
print(f"full loss = {loss/num_test}")
