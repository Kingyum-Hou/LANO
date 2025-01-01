from __future__ import annotations
import torch
import torch.nn as nn
from einops import rearrange, repeat, einsum
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Dict
from torch import Tensor
from timm.models.layers import trunc_normal_
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_, orthogonal_
import math


ACTIVATION = {'gelu':nn.GELU(),'tanh':nn.Tanh(),'sigmoid':nn.Sigmoid(),'relu':nn.ReLU(),'leaky_relu':nn.LeakyReLU(0.1),'softplus':nn.Softplus(),'ELU':nn.ELU()}


class MLP(nn.Module):
    '''
    A simple MLP class, includes at least 2 layers and n hidden layers
    implementation based on:
    "https://github.com/thuml/Transolver/blob/main/PDE-Solving-StandardBenchmark/model/Transolver_Irregular_Mesh.py#L12"
    '''
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input  = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res      = res
        self.linear_pre  = nn.Sequential(nn.Linear(n_input, n_hidden), act)
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears     = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


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
    

class RotaryEmbedding(nn.Module):
    """
    New position encoding module
    modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
    """
    def __init__(self, dim, min_freq=1/64, scale=1.):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coordinates, device):
        # coordinates [b, n]
        t = coordinates.to(device).type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)  # [b, n, d//2]
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]


class ApplyRotaryEmbedding():
    """
    A class to apply rotary positional embeddings to input tensors.
    Attributes: 
        space_dim : int : The dimensionality of the space (1D or 2D) for the rotary embedding.
    refer to:
    https://github.com/BaratiLab/OFormer/blob/main/uniform_grids/nn_module/attention_module.py#L96
    """
    def __init__(self, space_dim):
        self.name = 'rotaryEmbedding'
        self.space_dim = space_dim

    def rotate_half(x):
        x = rearrange(x, '... (j d) -> ... j d', j = 2)
        x1, x2 = x.unbind(dim = -2)
        return torch.cat((-x2, x1), dim = -1)

    def apply_rotary_pos_emb(self, t, freqs):
        return (t * freqs.cos()) + (self.rotate_half(t) * freqs.sin())


    def apply_2d_rotary_pos_emb(self, t, freqs_x, freqs_y):
        # split t into first half and second half
        # t: [b, h, n, d]
        # freq_x/y: [b, n, d]
        d = t.shape[-1]
        t_x, t_y = t[..., :d//2], t[..., d//2:]

        return torch.cat(
            (
                self.apply_rotary_pos_emb(t_x, freqs_x),
                self.apply_rotary_pos_emb(t_y, freqs_y)
            ), dim=-1
        )

    def __call__(self, t, **freqs):
        if self.space_dim == 1:
            return self.apply_rotary_pos_emb(t, freqs)
        elif self.space_dim == 2:
            return self.apply_2d_rotary_pos_emb(t, freqs['freqs_x'], freqs['freqs_y'])
        else:
            raise Exception('Currently doesnt support relative embedding > 2 dimensions')


class Temperature(nn.Module):
    def __init__(self, heads_num, temperature=0.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones([1, heads_num, 1, 1]) * temperature)

    def forward(self, x):
        return x / self.temperature


class KernelIntegrator(nn.Module):
    def __init__(self, hidden_size, feature_basis_num, heads_num):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.feature_basis_projector = nn.Sequential(
            MLP(hidden_size, hidden_size, feature_basis_num, n_layers=0, act='gelu', res=False),
            nn.Softmax(dim=-1),
        )
        self.attn = nn.MultiheadAttention(embed_dim=feature_basis_num, num_heads=heads_num)
        self.conv = PartialConv(feature_basis_num, feature_basis_num, 3, padding=1)
        self.last_mlp  = MLP(
            hidden_size, hidden_size, hidden_size, n_layers=0, act='gelu', res=False
        )
        self.feature_basis_num = feature_basis_num

    def compute_feature_map(self, x, no_valid):
        B, N, C = x.shape
        feature_basis = self.feature_basis_projector(x)
        psi = torch.einsum("b n c, b n f -> b c f", x, feature_basis)
        feature_basis = feature_basis.masked_fill(no_valid, 0.)
        feature_basis_norm = feature_basis.sum(dim=1)
        psi = psi / (feature_basis_norm + 1e-6)[:, None, :].repeat(1, C, 1)
        return psi, feature_basis

    def map_back_to_originalSpace(self, feature_basis, mask, psi):
        feature_basis_new = rearrange(feature_basis, 'b (H W) F -> b F H W', H=64, W=64)
        mask_new = rearrange(mask, 'b (H W) 1 -> b 1 H W', H=64, W=64)
        mask_new = mask_new.repeat(1, self.feature_basis_num, 1, 1)
        feature_basis_new, mask_new = self.conv(feature_basis_new, mask_new)
        feature_basis_new = rearrange(feature_basis_new, 'b F H W -> b (H W) F')
        mask_new          = rearrange(mask_new, 'b F H W -> b (H W) F')[..., :1]
        x        = torch.einsum("b c f, b n f -> b n c", psi, feature_basis_new)
        return x, mask_new

    def get_psi(self, x, mask):
        no_valid = mask == 0
        x = x.masked_fill(no_valid, 0.)
        
        # compute feature map
        x_ = self.ln_1(x)
        psi, _ = self.compute_feature_map(x_, no_valid)
        return psi

    def forward(self, x, mask):
        no_valid = mask == 0
        x = x.masked_fill(no_valid, 0.)
        
        # compute feature map
        x_ = self.ln_1(x)
        psi, feature_basis = self.compute_feature_map(x_, no_valid)

        # conjugate operator
        psi_trans = psi.permute(2, 0, 1)
        psi_trans, _ = self.attn(psi_trans, psi_trans, psi_trans)
        psi = psi_trans.permute(1, 2, 0)

        # map back to original space
        x_, mask_new = self.map_back_to_originalSpace(feature_basis, mask, psi)

        # mlp
        x = x + x_
        x = self.last_mlp(self.ln_2(x)) + x
        return x, mask_new


class OursModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.featureExpander = MLP(
            args.input_size + args.ref*args.ref, 
            args.hidden_size * 2, args.hidden_size, 
            n_layers=0, act='gelu', res=False
        )
        self.kernelProcessor = []
        for _ in range(args.kernel_layers):
            self.kernelProcessor.append(
                KernelIntegrator(
                    args.hidden_size, 
                    args.feature_basis_num, 
                    args.heads_num
                )
            )
        self.kernelProcessor = nn.ModuleList(self.kernelProcessor)
        self.projector = nn.Sequential(
            nn.LayerNorm(args.hidden_size), 
            nn.Linear(args.hidden_size, args.output_size)
        )
        self.initialize_weights()

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

    def get_psi(self, pos, x, mask1, mask2):
        if pos.dim()>3:
            pos = rearrange(pos, 'b ... c -> b (...) c')
        z = torch.concat([pos, x], dim=-1)
        z = self.featureExpander(z)
        psi1 = self.kernelProcessor[0].get_psi(z, mask1)
        psi2 = self.kernelProcessor[0].get_psi(z, mask2)
        return psi1, psi2

    def forward(self, pos, x, mask):
        if pos.dim()>3:
            pos = rearrange(pos, 'b ... c -> b (...) c')
        z = torch.concat([pos, x], dim=-1)
        z = self.featureExpander(z)
        for _, block in enumerate(self.kernelProcessor):
            z, mask = block(z, mask)
        y  = self.projector(z)
        return y

