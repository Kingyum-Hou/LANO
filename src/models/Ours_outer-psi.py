from __future__ import annotations
import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.layers import trunc_normal_
import math
from xformers.ops import memory_efficient_attention


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

    def rotate_half(self, x):
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
    

class Empty(nn.Module):
    def __init__(self):
        super().__init__()
    
    def build(self, mask):
        self.mask = mask

    def forward(self, x):
        x.masked_fill(self.mask, 0.)
        return x


class KernelIntegrator(nn.Module):
    def __init__(self, hidden_size, feature_basis_num, heads_num, space_size):
        super().__init__()
        head_size = hidden_size // heads_num
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        
        self.to_qkv = nn.Linear(hidden_size, hidden_size*3, bias=False)
        #self.conjugate = MLP(feature_basis_num, feature_basis_num*2, feature_basis_num, n_layers=0, act='gelu')
        
        self.fnn  = MLP(hidden_size, hidden_size, hidden_size, n_layers=0, act='gelu', res=False)
        self.feature_basis_num = feature_basis_num
        self.heads_num = heads_num
        self.head_size = head_size
        self.space_size = space_size

    def forward(self, x):
        """
            conjugate operator
        """
        # multihead attention
        query, key, value = self.to_qkv(self.ln_1(x)).chunk(3, dim = -1)
        query = rearrange(query, 'b f (h c) -> b f h c', h=self.heads_num, c=self.head_size)
        key   = rearrange(key,   'b f (h c) -> b f h c', h=self.heads_num, c=self.head_size)
        value = rearrange(value, 'b f (h c) -> b f h c', h=self.heads_num, c=self.head_size)
        attn_out = memory_efficient_attention(query, key, value)
        attn_out = rearrange(attn_out, 'b f h c -> b f (h c)')
        x     = x + attn_out
        """
        psi = psi.permute(0, 1, 3, 2)
        psi = self.conjugate(psi)
        psi = psi.permute(0, 1, 3, 2)
        """
        # fnn
        x = x + self.fnn(self.ln_2(x))
        return x


class Interp(nn.Module):
    def __init__(self, feature_basis_num, layers_num):
        super().__init__()
        self.convs = []
        for _ in range(layers_num):
            self.convs.append(
                PartialConv(feature_basis_num, feature_basis_num, 3, padding=1)
            )
        self.convs = nn.ModuleList(self.convs)

    def forward(self, x, mask):
        for _, conv in enumerate(self.convs):
            x, mask = conv(x, mask)
        return x, mask


class OursModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.featureExpander = MLP(
            args.input_size + args.ref*args.ref, 
            args.hidden_size * 2, args.hidden_size, 
            n_layers=0, act='gelu', res=False
        )
        self.head_size  = args.hidden_size // args.heads_num
        self.heads_num  = args.heads_num
        self.space_size = args.space_size
        self.downsample = args.downsample
        self.hidden_size = args.hidden_size
        self.feature_basis_num = args.feature_basis_num
        self.ln   = nn.LayerNorm(args.hidden_size)
        self.feature_basis_projector = nn.Sequential(
            MLP(args.head_size, args.head_size, args.feature_basis_num, n_layers=0, act='gelu', res=False),
            Temperature(args.heads_num, temperature=0.5),
            Empty(),
            nn.Softmax(dim=-1),
        )
        self.kernelProcessor = []
        h = int((args.space_size[0] / args.downsample))
        w = int((args.space_size[1] / args.downsample))
        for _ in range(args.kernel_layers):
            self.kernelProcessor.append(
                KernelIntegrator(
                    args.hidden_size, 
                    args.feature_basis_num, 
                    args.heads_num,
                    space_size=[h, w],
                )
            )
        self.kernelProcessor = nn.ModuleList(self.kernelProcessor)
        self.interp = Interp(args.feature_basis_num*args.heads_num, 4)
        self.to_out = nn.Linear(args.hidden_size, args.hidden_size)
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

        # dynamic-static split for mask1 & mask2
        no_valid1 = mask1 == 0
        no_valid2 = mask2 == 0
        z1 = z.masked_fill(no_valid1, 0.)
        z2 = z.masked_fill(no_valid2, 0.)
        psi1, _ = self.compute_feature_map(z1, no_valid1)
        psi2, _ = self.compute_feature_map(z2, no_valid2)
        return psi1, psi2
    
    def compute_feature_map(self, x, no_valid):
        no_valid = rearrange(no_valid, 'b n (1 1) -> b 1 n 1')
        self.feature_basis_projector[2].build(no_valid)
        x = rearrange(x, 'b n (h c) -> b h n c', h=self.heads_num, c=self.head_size)
        feature_basis = self.feature_basis_projector(x)
        psi = torch.einsum("b h n c, b h n f -> b h f c", x, feature_basis).contiguous()
        feature_basis = feature_basis.masked_fill(no_valid, 0.)
        feature_basis_norm = feature_basis.sum(dim=2)
        psi = psi / (feature_basis_norm + 1e-6)[:, :, :, None].repeat(1, 1, 1, self.head_size)
        return psi, feature_basis

    def map_back_to_originalSpace(self, feature_basis, mask, psi):
        H = self.space_size[0] // self.downsample
        W = self.space_size[1] // self.downsample
        feature_basis_new = rearrange(feature_basis, 'b h (H W) f -> b (h f) H W', H=H, W=W)
        mask              = rearrange(mask,          'b (H W) 1 -> b 1 H W', H=H, W=W) 
        mask              = mask.repeat(1, self.heads_num*self.feature_basis_num, 1, 1)
        feature_basis_new, _ = self.interp(feature_basis_new, mask)
        feature_basis_new = rearrange(feature_basis_new, 'b (h f) H W -> b h (H W) f', h=self.heads_num, f=self.feature_basis_num)
        x = torch.einsum("b h f c, b h n f -> b h n c", psi, feature_basis_new)
        x = rearrange(x, 'b h n c -> b n (h c)')
        return x

    def forward(self, pos, x, mask):
        if pos.dim()>3:
            pos = rearrange(pos, 'b ... c -> b (...) c')
        z = torch.concat([pos, x], dim=-1)
        z = self.featureExpander(z)

        # dynamic-static split
        no_valid = mask == 0
        z = z.masked_fill(no_valid, 0.)
        #psi, feature_basis = self.compute_feature_map(z, no_valid)
        psi, feature_basis = self.compute_feature_map(self.ln(z), no_valid)

        # token mixing
        psi = rearrange(psi, 'b h n c -> b n (h c)', h=self.heads_num, c=self.head_size)
        for _, block in enumerate(self.kernelProcessor):
            psi = block(psi)

        # dynamic-static recovery
        psi = rearrange(psi, 'b n (h c) -> b h n c', h=self.heads_num, c=self.head_size)
        z_new = self.map_back_to_originalSpace(feature_basis, mask, psi)
        z_new = self.to_out(z_new)
        z     = z + z_new 

        y  = self.projector(z)
        return y
