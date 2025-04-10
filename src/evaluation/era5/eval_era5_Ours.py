import torch
from model_interface import get_model
from tools import get_pos_ref
from dataloaders.era5 import add_point_missing, add_patch_missing
import numpy as np
import matplotlib.pyplot as plt
from tools import LpLoss
import math
from einops import rearrange
import xarray as xr


# env
missing_rate = 0.5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
alpha = 1e-5


def loss_surrogate(psi1, psi2):
        psi1 = rearrange(psi1, 'b h f c -> b c (h f)')
        psi2 = rearrange(psi2, 'b h f c -> b c (h f)')
        psi1 = psi1.div(psi1.norm(dim=0).clamp(min=1e-6)) * math.sqrt(10)
        psi2 = psi2.div(psi2.norm(dim=0).clamp(min=1e-6)) * math.sqrt(10)
        psi_K_psi_diag = (psi1 * psi2).sum(0)
        psi2_d_K_psi1 = torch.einsum('bci, bcj -> cij', psi2, psi1)
        psi1_d_K_psi2 = torch.einsum('bci, bcj -> cij', psi1, psi2)
        loss = - psi_K_psi_diag.sum() * 2
        reg  = (psi2_d_K_psi1 ** 2).triu(1).sum() + \
               (psi1_d_K_psi2 ** 2).triu(1).sum()
        loss /= psi_K_psi_diag.numel()
        reg  /= psi_K_psi_diag.numel()
        loss = loss + alpha*reg
        return loss


class ModelParams():
    def __init__(self):
        self.name = 'Ours'
        self.input_size = 7
        self.space_size = [720, 1440]
        self.output_size = 1
        self.ref = 8
        self.feature_basis_num = 32
        self.downsample = 1
        self.unified_pos = True
        self.Time_Input = False
        self.slice_num = 32
        self.kernel_layers = 8
        self.heads_num = 8
        self.head_size = 32
        self.hidden_size = 256
        self.mlp_ratio = 1
        self.downsample = 8

model = get_model(ModelParams()).to(device)
#model_path = 'save/NS/Ours/task3_mr=50/best-v4.ckpt'
#model_path = 'save/NS/Ours/task3_mr=25/best-v4.ckpt'
#model_path = 'save/NS/Ours/task3_mr=5/best-v5.ckpt'
#model_path = 'save/NS/Ours/task4_mr=5/best-v2.ckpt'
#model_path = 'save/NS/Ours/task4_mr=25/best-v4.ckpt'
#model_path = 'save/NS/Ours/task4_mr=50/best-v3.ckpt'
#model_path = 'save/ERA5/Ours/task3_mr=5/best-v8.ckpt'
#model_path = 'save/ERA5/Ours/task3_mr=25/best-v8.ckpt'
#model_path = 'save/ERA5/Ours/task3_mr=50/best-v5.ckpt'
#model_path = 'save/ERA5/Ours/task4_mr=5/best-v4.ckpt'
model_path = 'save/ERA5/Ours/task4_mr=25/best-v7.ckpt'
model_dict = torch.load(model_path, map_location=device)['state_dict']
model_dict = {k.replace('model.', ''): v for k, v in model_dict.items()}
print(model.load_state_dict(model_dict, strict=False))

model.eval()
test_x = torch.rand(1, 16200, 7).to(device)
test_pos_in = torch.rand(1, 16200, 64).to(device)
test_mask = torch.rand(1, 16200, 1).to(device)
test_t = 7
test_pred = model(test_pos_in, test_x, test_mask)

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
test_xy = data[-num_test:, ...].reshape(num_test, -1, 14)
test_y  = test_xy[..., 7:]
#test_xy_withMissing, test_mask = add_point_missing(test_xy, int(np.round(h*w*missing_rate)))
test_xy_withMissing, test_mask = add_patch_missing(test_xy, missing_rate, [90, 180], patch_size=3)
test_x = test_xy_withMissing[..., :7]

pos = get_pos_ref(h, w, 8, batchsize=1).contiguous()
test_pos = pos.repeat(num_test, 1, 1, 1).reshape(num_test, -1, 8*8)

batch_size = 10
test_data = [test_mask, test_pos, test_x, test_y]
test_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*test_data), batch_size=batch_size, shuffle=False)

# eval
loss_func = LpLoss(size_average=False)
loss = 0.
plot_flag = True
with torch.no_grad():
    for batch_idx, batch in enumerate(test_dataloader):
        mask, pos, xx, yy = [x.to(device) for x in batch]

        pred_trajectory = []
        for t in range(7):
            y = yy[..., t:t+1]
            pred = model(pos, xx, mask[..., :1])
            pred_trajectory.append(pred)
            xx = torch.cat([xx[..., 1:], pred], dim=-1)
        pred = torch.cat(pred_trajectory, dim=-1)
        loss += loss_func(pred.view(batch_size, -1), yy.view(batch_size, -1))
        if plot_flag:
            # Plot the predictions
            fig, axes = plt.subplots(3, 7, figsize=(28, 12))
            yy = yy.reshape(batch_size, h, w, 7)
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
            plt.savefig(f'imgs/Ours_{batch_idx}.png')
            plt.clf()
            #plot_flag = False
print('ok!')
print(f"full loss = {loss/num_test}")
