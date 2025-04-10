import torch
from scipy import io as scio
from model_interface import get_model
from tools import get_pos
from dataloaders.ns import add_point_missing, add_patch_missing
import numpy as np
import matplotlib.pyplot as plt
from tools import LpLoss
import math
from einops import rearrange

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
        self.ntrain = 1000
        self.ntest = 200
        self.r = 64

model = get_model(ModelParams()).to(device)
#model_path = 'save/NS/OFormer/task3_mr=5/best-v1.ckpt'
#model_path = 'save/NS/OFormer/task3_mr=25/best-v1.ckpt'
#model_path = 'save/NS/OFormer/task3_mr=50/best.ckpt'
#model_path = 'save/NS/OFormer/task4_mr=5/best-v2.ckpt'
#model_path = 'save/NS/OFormer/task4_mr=25/best-v2.ckpt'
model_path = 'save/NS/OFormer/task4_mr=50/best-v1.ckpt'
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
data_path = '/data/jingren/repository/dataset/Physics-informed-neural-network/PDE_datasets/NavierStokes_V1e-5_N1200_T20.mat'
data = scio.loadmat(data_path)['u']
test_au = torch.tensor(data[-200:, ::1, ::1, :20], dtype=torch.float).reshape(200, -1, 20)
test_u = test_au[..., 10:]
#test_au_withMissing, test_mask = add_point_missing(test_au, int(np.round(4096*missing_rate)))
test_au_withMissing, test_mask = add_patch_missing(test_au, missing_rate, [64, 64], patch_size=4)
test_a = test_au_withMissing[..., :10]

pos = get_pos(64, 64).unsqueeze(0).contiguous()
test_pos = pos.repeat(200, 1, 1, 1).reshape(200, -1, 2)

test_data = [test_mask, test_pos, test_a, test_u]
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
            plt.savefig(f'imgs/Ours_{batch_idx}.png')
            plt.clf()
            #plot_flag = False
print('ok!')
print(f"full loss = {loss/200}")
