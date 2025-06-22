from omegaconf import OmegaConf
import torch
import os
import numpy as np
import random
from einops import rearrange
import torchprofile
import warnings
import h5py
import scipy
from pytorch_lightning.callbacks import Callback
import torch.nn.functional as F


def cli(args):
    # load config data 
    if args.config_path != None:
        print('loading config from config file')
        base_conf = OmegaConf.load(args.config_path)
    else: 
        print('No base config')
        return None
    OmegaConf.resolve(base_conf)
    # load config from command lines
    args_conf = OmegaConf.create(vars(args))
    for key in args_conf:
        if args_conf[key] is None and key in base_conf:
            args_conf[key] = base_conf[key]
    conf = OmegaConf.merge(base_conf, args_conf)
    return conf


def seed_everything(seed) -> int:
    if not isinstance(seed, int):
        seed = int(seed)
    print(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return seed


"""
def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    total_bytes = total_params * 4  # float32
    total_megabytes = total_bytes / (1024**2)
    print(f"Total Trainable Params: {total_params}/{(total_params/1e6):.3f}M")
    print(f"memory is approximately: {total_megabytes:.3f}Mb")
    return total_params, total_megabytes
"""


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params:,} ({total_params / 1e6:.2f}M)")
    return total_params


def central_diff(x: torch.Tensor, h=64, w=64):
    # assuming PBC
    # x: (batch, n, seq_len), h is the step size, assuming n = h*w
    x = rearrange(x, 'b (h w) t -> b t h w', h=h, w=w)
    h = 1./64.
    x = F.pad(x,
              (1, 1, 1, 1), mode='circular')  # [b t h+2 w+2]
    grad_x = (x[..., 1:-1, 2:] - x[..., 1:-1, :-2]) / (2*h)  # f(x+h) - f(x-h) / 2h
    grad_y = (x[..., 2:, 1:-1] - x[..., :-2, 1:-1]) / (2*h)  # f(x+h) - f(x-h) / 2h
    return grad_x, grad_y


def rel_l2norm_loss(x, y):
    #   x, y [b, c, t, n]
    eps = 1e-6
    y_norm = (y**2).mean(dim=-1) + eps
    diff = ((x-y)**2).mean(dim=-1)
    diff = diff / y_norm   # [b, c, t]
    diff = diff.sqrt().mean()
    return diff


# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1),
                                                          self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def reshape2blocks(x, patch_size, patch_num):
    x = rearrange(x, 
                'B (PN1 H) (PN2 W) T -> B (PN1 PN2) H W T',
                B=x.shape[0],
                T=x.shape[-1],
                H=patch_size[0],
                W=patch_size[1],
                PN1=patch_num[0],
                PN2=patch_num[1],
            )
    return x


def reshape2data(x, patch_size, patch_num):
    x = rearrange(x,
                'B (PN1 PN2) H W T -> B (PN1 H) (PN2 W) T',
                B=x.shape[0],
                T=x.shape[-1],
                H=patch_size[0],
                W=patch_size[1],
                PN1=patch_num[0],
                PN2=patch_num[1],
            )
    return x


def add_patch_holes(data, patch_size, space_size, missing_rate):
        patch_num  = [space_size[0]//patch_size[0], space_size[1]//patch_size[1]]
        total_patches   = np.prod(patch_num) 
        num_patch_holes = int(total_patches * missing_rate)
        B = data.shape[0]
        
        valid_data = reshape2blocks(data, patch_size, patch_num)
        valid_mask = torch.ones_like(valid_data)
        for i in range(B):
            indices_addHoles = torch.randperm(total_patches)[:num_patch_holes]
            valid_data[i, indices_addHoles, ...] = 0.
            valid_mask[i, indices_addHoles, ...] = 0.
        
        valid_data = reshape2data(valid_data, patch_size, patch_num)
        valid_mask = reshape2data(valid_mask, patch_size, patch_num)
        return valid_data, valid_mask


def smoothness_penalty(tensor):
    dy = torch.diff(tensor, dim=1)
    dx = torch.diff(tensor, dim=2)
    ddy = torch.diff(dy, dim=1)
    ddx = torch.diff(dx, dim=2)
    penalty_y = torch.mean(ddy ** 2)
    penalty_x = torch.mean(ddx ** 2)
    smoothness_penalty = penalty_y + penalty_x
    return smoothness_penalty


# jerk loss
class L2Loss(torch.nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
    
    def forward(self, input):
        l2_norm_squared = torch.sum(input ** 2)
        return l2_norm_squared
jerk_penalty_func = L2Loss()
def jerk_penalty(tensor):
    num_latents = tensor.shape[-3]
    embedding_dim = tensor.shape[-2]
    jerk_penalty = jerk_penalty_func(tensor[..., 3] - 3*tensor[..., 2] + 3*tensor[..., 1] - tensor[..., 0])/embedding_dim/num_latents
    return jerk_penalty


def save_model(model_savePath, model_saveName, model_name, model, epoch_iter, dummy_input=None):
    if not os.path.exists(os.path.join(model_savePath, model_saveName)):
        os.makedirs(os.path.join(model_savePath, model_saveName))
    torch.save(model.state_dict(), os.path.join(model_savePath, model_saveName, f"{epoch_iter}_{model_name}.pt"))
    return


def load_model(model_savePath, model_saveName, model):
    model.load_state_dict(torch.load(os.path.join(model_savePath, model_saveName)))
    return model


def get_pos(num_x, num_y, bot=(0, 0), top=(1, 1), type='ij'):
        x_bot, y_bot = bot
        x_top, y_top = top
        x_arr = torch.linspace(x_bot, x_top, steps=num_x)
        y_arr = torch.linspace(y_bot, y_top, steps=num_y)
        xx, yy = torch.meshgrid(x_arr, y_arr, indexing=type)
        mesh = torch.stack([xx, yy], dim=2)
        return mesh


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


def get_pos_ref(H, W, ref, bot=(0, 0), top=(1, 1), type='ij'):
        """
        Generates a grid of positions and computes the Euclidean distance between 
        each point in the grid and a reference grid.
        Args:
            H (int): Height of the grid.
            W (int): Width of the grid.
            ref (int): Size of the reference grid.
            batchsize (int, optional): Batch size for the grid. Defaults to 1.
        Returns:
            torch.Tensor: A tensor containing the Euclidean distances between 
                          each point in the grid and the reference grid. The shape 
                          of the tensor is (batchsize, H, W, ref * ref).
        ref to:
        https://github.com/thuml/Transolver/blob/main/PDE-Solving-StandardBenchmark/model/Transolver_Structured_Mesh_2D.py#L138
        """
        x_bot, y_bot = bot
        x_top, y_top = top
        size_x, size_y = H, W
        gridx = torch.tensor(np.linspace(x_bot, x_top, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([1, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(y_bot, y_top, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([1, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1)  # B H W 2

        gridx = torch.tensor(np.linspace(x_bot, x_top, ref), dtype=torch.float)
        gridx = gridx.reshape(1, ref, 1, 1).repeat([1, 1, ref, 1])
        gridy = torch.tensor(np.linspace(y_bot, y_top, ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, ref, 1).repeat([1, ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1)  # B H W 8 8 2

        pos = torch.sqrt(
                    torch.sum(
                        (grid[:, :, :, None, None, :] - \
                        grid_ref[:, None, None, :, :, :]) ** 2, 
                        dim=-1
                    )
                ).reshape(1, size_x, size_y, ref * ref).contiguous()
        return pos


def check_model_parameters_isnan(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"param:{name}'s grad is NaN")
                

def reshape2blocks(x, patch_size, patch_num):
    x = rearrange(
                x, 
                'B (PN1 H PN2 W) T -> B (PN1 PN2) H W T',
                B=x.shape[0],
                T=x.shape[-1],
                H=patch_size,
                W=patch_size,
                PN1=patch_num[0],
                PN2=patch_num[1],
            )
    return x


def reshape2data(x, patch_size, patch_num):
    x = rearrange(
                x,
                'B (PN1 PN2) H W T -> B (PN1 H PN2 W) T',
                B=x.shape[0],
                T=x.shape[-1],
                H=patch_size,
                W=patch_size,
                PN1=patch_num[0],
                PN2=patch_num[1],
            )
    return x


class UnitTransformer():
    def __init__(self, X):
        self.mean = X.mean(dim=(0, 1), keepdim=True)
        self.std = X.std(dim=(0, 1), keepdim=True) + 1e-8

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

    def encode(self, x):
        x = (x - self.mean) / (self.std)
        return x

    def decode(self, x):
        return x * self.std + self.mean

    def transform(self, X, inverse=True, component='all'):
        if component == 'all' or 'all-reduce':
            if inverse:
                orig_shape = X.shape
                return (X * (self.std - 1e-8) + self.mean).view(orig_shape)
            else:
                return (X - self.mean) / self.std
        else:
            if inverse:
                orig_shape = X.shape
                return (X * (self.std[:, component] - 1e-8) + self.mean[:, component]).view(orig_shape)
            else:
                return (X - self.mean[:, component]) / self.std[:, component]


def masked_loss_average(pred, target, mask, criterion):
    B = pred.size(0)
    loss = 0.0
    valid_count = 0

    for i in range(B):
        pi = pred[i][mask[i]]
        yi = target[i][mask[i]]

        if pi.numel() > 0:
            loss += criterion(pi, yi)
            valid_count += 1

    if valid_count == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    return loss / valid_count


def add_point_missing(data, num_sampling:int)-> tuple:
    """
    Randomly masks a specified number of points in the input data tensor by setting their values to zero.

    Args:
        data (torch.Tensor): Input tensor of shape (B, HW, T), where B is the batch size, HW is the number of points, and T is the number of features per point.
        num_sampling (int): Number of points to randomly mask (set to zero) for each sample in the batch.

    Returns:
        tuple: A tuple containing:
            - valid_data (torch.Tensor): Tensor with the same shape as input, where randomly selected points are set to zero.
            - valid_mask (torch.Tensor): Binary mask tensor of the same shape as input, where masked points are set to zero and others are one.

    Note:
        The masking is performed independently for each sample in the batch.
    """
    B, HW, T   = data.shape

    valid_data = data.clone()
    valid_mask = torch.ones_like(valid_data)
    for i in range(B):
        indices_addMissing = torch.randperm(HW)[:num_sampling]
        valid_data[i, indices_addMissing, ...] = 0.
        valid_mask[i, indices_addMissing, ...] = 0.
    return valid_data, valid_mask


def add_patch_missing(data:torch.Tensor, missing_rate:float, space_size:list, patch_size:int=4)-> tuple:
    """
    Randomly masks patches in the input tensor to simulate missing data.
    
    Args:
        data (torch.Tensor): Input tensor of shape (B, HW, T) where B is batch size.
        missing_rate (float): Fraction of patches to be masked (set to zero).
        space_size (list): Spatial dimensions of the input data, typically [H, W].
        patch_size (int, optional): Size of each square patch. Default is 4.

    Returns:
        tuple:
            - valid_data (torch.Tensor): Tensor with randomly masked patches set to zero, same shape as input.
            - valid_mask (torch.Tensor): Binary mask tensor of the same shape as input, where 1 indicates valid (unmasked) data and 0 indicates masked (missing) patches.
    
    Note:
        The masking is performed independently for each sample in the batch.
    """
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
