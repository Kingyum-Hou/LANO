import torch
from scipy import io as scio
from model_interface import get_model
from tools import torch2dgrid, get_grid
from dataloaders.ns import add_point_missing
import numpy as np
import matplotlib.pyplot as plt
from tools import LpLoss
import math
from einops import rearrange

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
        self.input_size = 10
        self.space_dim = [64, 64]
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

model = get_model(ModelParams()).to(device)
model_path = 'logs/Ours_task3_0.5_Ours_reg9/checkpoints/best-v4.ckpt'
model_dict = torch.load(model_path, map_location=device)['state_dict']
model_dict = {k.replace('model.', ''): v for k, v in model_dict.items()}
print(model.load_state_dict(model_dict, strict=False))

model.eval()
test_x = torch.rand(1, 4096, 10).to(device)
test_pos_in = torch.rand(1, 4096, 64).to(device)
test_mask = torch.rand(1, 4096, 1).to(device)
test_t = 10
test_pred = model(test_pos_in, test_x, test_mask)

# load data
data_path = '/root/autodl-tmp/NavierStokes_V1e-5_N1200_T20.mat'
data = scio.loadmat(data_path)['u']
test_au = torch.tensor(data[-200:, ::1, ::1, :20], dtype=torch.float).reshape(200, -1, 20)
test_u = test_au[..., 10:]
test_au_withMissing, test_mask = add_point_missing(test_au, int(np.round(4096*missing_rate)))
test_a = test_au_withMissing[..., :10]
pos = get_grid(64, 64, 8, batchsize=1).contiguous()
test_pos = pos.repeat(200, 1, 1, 1).reshape(200, -1, 8*8)

test_data = [test_mask, test_pos, test_a, test_u]
test_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*test_data), batch_size=10, shuffle=False)

# eval
loss_func = LpLoss(size_average=False)
loss = 0.
plot_flag = True
with torch.no_grad():
    for batch_idx, batch in enumerate(test_dataloader):
        mask, pos, xx, yy = [x.to(device) for x in batch]

        pred_trajectory = []
        for t in range(10):
            y = yy[..., t:t+1]
            pred = model(pos, xx, mask[..., :1])
            pred_trajectory.append(pred)
            xx = torch.cat([xx[..., 1:], pred], dim=-1)
        pred = torch.cat(pred_trajectory, dim=-1)
        loss += loss_func(pred.view(10, -1), yy.view(10, -1))
        if plot_flag:
            # Plot the predictions
            fig, axes = plt.subplots(3, 10, figsize=(40, 12))
            yy = yy.reshape(10, 64, 64, 10)
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
            plt.savefig(f'imgs/TransolverPro_{batch_idx}.png')
            plt.clf()
            #plot_flag = False
print('ok!')
print(f"full loss = {loss/200}")
