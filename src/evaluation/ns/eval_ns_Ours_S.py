import torch
from scipy import io as scio
from model_interface import get_model
from tools import get_pos_ref
from tools import add_point_missing, add_patch_missing
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
        self.name = 'Ours_S_P'
        self.input_size = 10
        self.space_size = [64, 64]
        self.output_size = 1
        self.ref = 8
        self.scale_factor = 2.
        self.latent_num = 256
        self.downsample = 1
        self.unified_pos = True
        self.Time_Input = False
        self.kernel_layers = 12
        self.heads_num = 8
        self.head_size = 32
        self.hidden_size = 256
        self.token_Mixer="Attention"

model = get_model(ModelParams()).to(device)
model_path = '/root/ANOT/logs/NS_v-5/Ours_lno_task4_0.25_exp_ns_ours_lno_t4_mr25_layer12_bigPO_P8/checkpoints/best.ckpt'
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
#test_au_withMissing, test_mask = add_point_missing(test_au, int(np.round(4096*missing_rate)))
test_au_withMissing, test_mask = add_patch_missing(test_au, missing_rate, [64, 64], patch_size=8)
test_a = test_au_withMissing[..., :10]
pos = get_pos_ref(64, 64, 8).contiguous()
test_pos = pos.repeat(200, 1, 1, 1).reshape(200, -1, 8*8)

test_data = [test_mask, test_pos, test_a, test_u]
test_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*test_data), batch_size=10, shuffle=False)

# eval print latent score
data_batch = next(iter(test_dataloader))
data_batch = next(iter(test_dataloader))
with torch.no_grad():
    mask, pos, xx, yy = [x.to(device) for x in data_batch]
    pred_trajectory = []
    y = yy[..., 0:1]
    pred, score_list, mask_list, y_list = model(pos, xx, mask[..., :1])
    pred_trajectory.append(pred)
    xx = torch.cat([xx[..., 1:], pred], dim=-1)
    pred = torch.cat(pred_trajectory, dim=-1)
    print(score_list[0].shape)

fig, axes = plt.subplots(1, 12, figsize=(60, 5), dpi=318)
d_arrange = 23
#j = 5
j = 0
for i in range(6):
    axes[i].imshow(y_list[i][j, ..., d_arrange].cpu().numpy().reshape(64,64), cmap='viridis')
    axes[i].imshow(np.where(mask_list[i][j, ..., 0].cpu().numpy().reshape(64,64) == 0, 0, np.inf), cmap='binary', alpha=1, vmin=0, vmax=1)
    axes[i].axis('off')
for i in range(6):
    axes[i+6].imshow(y_list[i+6][j, ..., d_arrange].cpu().numpy().reshape(64,64), cmap='viridis')
    axes[i+6].imshow(np.where(mask_list[i+6][j, ..., 0].cpu().numpy().reshape(64,64) == 0, 0, np.inf), cmap='binary', alpha=1, vmin=0, vmax=1)
    axes[i+6].axis('off')
#plt.savefig("1.png", bbox_inches="tight", pad_inches=0)
plt.savefig("1.png")

d_arrange = 200
#j = 5
j = 0
for i in range(6):
    axes[i].imshow(score_list[i][j, ..., d_arrange].cpu().numpy().reshape(64,64), cmap='viridis')
    axes[i].imshow(np.where(mask_list[i][j, ..., 0].cpu().numpy().reshape(64,64) == 0, 0, np.inf), cmap='binary', alpha=1, vmin=0, vmax=1)
    axes[i].axis('off')
for i in range(6):
    axes[i+6].imshow(score_list[i+6][j, ..., d_arrange].cpu().numpy().reshape(64,64), cmap='viridis')
    axes[i+6].imshow(np.where(mask_list[i+6][j, ..., 0].cpu().numpy().reshape(64,64) == 0, 0, np.inf), cmap='binary', alpha=1, vmin=0, vmax=1)
    axes[i+6].axis('off')
#plt.savefig("1.png", bbox_inches="tight", pad_inches=0)
plt.savefig("2.png")

d_arrange = 20
#j = 5
j = 9
for i in range(6):
    axes[i].imshow(score_list[i][j, ..., d_arrange].cpu().numpy().reshape(64,64), cmap='viridis')
    axes[i].imshow(np.where(mask_list[i][j, ..., 0].cpu().numpy().reshape(64,64) == 0, 0, np.inf), cmap='binary', alpha=1, vmin=0, vmax=1)
    axes[i].axis('off')
for i in range(6):
    axes[i+6].imshow(score_list[i+6][j, ..., d_arrange].cpu().numpy().reshape(64,64), cmap='viridis')
    axes[i+6].imshow(np.where(mask_list[i+6][j, ..., 0].cpu().numpy().reshape(64,64) == 0, 0, np.inf), cmap='binary', alpha=1, vmin=0, vmax=1)
    axes[i+6].axis('off')
#plt.savefig("1.png", bbox_inches="tight", pad_inches=0)
plt.savefig("3.png")

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
            plt.savefig(f'imgs/Ours_{batch_idx}.png')
            plt.clf()
            #plot_flag = False
print('ok!')
print(f"full loss = {loss/200}")
