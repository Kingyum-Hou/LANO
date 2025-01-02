import torch
import numpy as np
import torch.nn as nn
from timm.layers import trunc_normal_
from einops import rearrange
import math


ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


def weights_init(init_type='gaussian'):
        def init_fun(m):
            classname = m.__class__.__name__
            if (classname.find('Conv') == 0 or classname.find(
                    'Linear') == 0) and hasattr(m, 'weight'):
                if init_type == 'gaussian':
                    nn.init.normal_(m.weight, 0.0, 0.02)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                elif init_type == 'default':
                    pass
                else:
                    assert 0, "Unsupported initialization: {}".format(init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        return init_fun


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask
    

class Physics_Attention_Irregular_Mesh(nn.Module):
    ## for irregular meshes in 1D, 2D or 3D space
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x  = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.conv = PartialConv(slice_num*heads, slice_num*heads, 3, padding=1)

    def forward(self, x, mask):
        # B N C
        B, N, C = x.shape

        ### (1) Slice
        fx_mid_ = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid_ = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        mask_  = adjust_mask(rearrange(mask[..., 0:1], 'b n (1 1) -> b 1 n 1'))
        slice_weights_ = self.softmax(self.in_project_slice(x_mid_) / self.temperature)  # B H N G

        slice_weights = torch.masked_select(slice_weights_, mask_.bool()).reshape(B, 8, -1, 32)
        fx_mid        = torch.masked_select(fx_mid_,        mask_.bool()).reshape(B, 8, -1, 32)
        slice_norm  = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        # fillGap
        slice_weights_new = rearrange(slice_weights_, 'b h (H W) g -> b (h g) H W', H=64, W=64)
        mask_new = rearrange(mask[..., 0:1], 'b (H W) 1 -> b 1 H W', H=64, W=64)
        mask_new = mask_new.repeat(1, slice_weights_new.shape[1], 1, 1)
        slice_weights_new, mask_new = self.conv(slice_weights_new, mask_new)
        slice_weights_new = slice_weights_new.reshape(B, 256, -1)
        slice_weights_new = rearrange(slice_weights_new, 'b (h g) N -> b h N g', h=8, g=32)
        mask_new = rearrange(mask_new, 'b C H W -> b (H W) C')[..., :1]
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights_new)
        out_x = rearrange(out_x, 'b h n c -> b n (h c)')
        return self.to_out(out_x), mask_new


class Transolver_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
            H=85,
            W=85
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp  = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)
            
    def forward(self, fx, mask):
        no_valid = mask == 0
        fx = fx.masked_fill_(no_valid[..., 0:1], 0.0) 
        fx_attn, new_mask = self.Attn(self.ln_1(fx), mask) 
        fx_attn = fx_attn + fx
        fx_mlp = self.mlp(self.ln_2(fx_attn)) + fx_attn

        if self.last_layer:
            return self.mlp2(self.ln_3(fx_mlp)), new_mask
        else:
            return fx_mlp, new_mask


def adjust_mask(mask):
    B, _, N, _ = mask.shape
    min_mask_count = mask.sum(dim=2).min().item()
    adjusted_mask  = mask.clone()
    
    for b in range(B):
        mask_indices = torch.nonzero(mask[b, 0, :, 0], as_tuple=False).squeeze()
        if len(mask_indices) > min_mask_count:
            disable_num = int(len(mask_indices) - min_mask_count)
            disable_indices = torch.randperm(len(mask_indices))[:disable_num]
            disable_indices = mask_indices[disable_indices]
            adjusted_mask[b, 0, disable_indices, 0] = 0
    
    return adjusted_mask


class TransovlerPro(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.__name__ = 'Transolver_2D'
        self.H    = int(((args.space_dim[0]-1) / args.downsample) + 1)
        self.W    = int(((args.space_dim[1]-1) / args.downsample) + 1)
        self.S    = 2
        self.ref  = args.ref
        self.unified_pos = args.unified_pos
        n_layers  = args.num_layers
        n_hidden  = args.hidden_size
        n_head    = args.num_heads
        dropout   = 0.
        Time_Input = args.Time_Input
        act       = 'gelu'
        fun_dim   = args.input_channel
        out_dim   = args.output_channel

        if self.unified_pos:
            self.preprocess = MLP(fun_dim + self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
            self.pos = self.get_grid()
        else:
            self.preprocess = MLP(fun_dim + self.S, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)


        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        self.blocks = []
        for i in range(n_layers):
            self.blocks.append(
                Transolver_block(
                    num_heads=n_head, 
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=args.mlp_ratio,
                    out_dim=out_dim,
                    slice_num=args.slice_num,
                    H=self.H,
                    W=self.W,
                    last_layer=(i == n_layers - 1)
                )
            )
        self.blocks = nn.ModuleList(self.blocks)
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))
    
    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, batchsize=1):
        size_x, size_y = self.H, self.W
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).cuda()  # B H W 2

        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).cuda()  # B 8 8 2

        pos = torch.sqrt(
                torch.sum(
                    (grid[:, :, :, None, None, :] - \
                    grid_ref[:, None, None, :, :, :]) ** 2, dim=-1
                )
            ). reshape(batchsize, size_x, size_y, self.ref * self.ref).contiguous()  # B H W 8 8 2
        return pos

    def forward(self, x, fx, mask=None):
        if self.unified_pos:
            x = self.pos.repeat(x.shape[0], 1, 1, 1).reshape(x.shape[0], self.H * self.W, self.ref * self.ref)
            x = x.to(fx.device)
        if fx is not None:
            fx = rearrange(fx, 'b ... c -> b (...) c')
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx, mask  = block(fx, mask)
        
        return fx
