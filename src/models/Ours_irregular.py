from __future__ import annotations
import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.layers import trunc_normal_
import math
from xformers.ops import memory_efficient_attention
from torch_scatter import scatter_add
from torch_cluster import knn_graph


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


class PartialPointConv2D(nn.Module):
    """
        core layer
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.kaiming_normal_(self.lin.weight, nonlinearity='relu')

    def forward(self, x, edge_index, mask):
        N = x.size(0)
        x_masked = x * mask                    # (N, C_in)
        feat = self.lin(x_masked)              # (N, C_out)

        row, col = edge_index
        out      = scatter_add(feat[col], row, dim=0, dim_size=N)
        mask_sum = scatter_add(mask[col], row, dim=0, dim_size=N)

        bias      = (self.lin.bias if self.lin.bias is not None
                     else torch.zeros(out.size(1), device=x.device)).view(1, -1)
        no_update = mask_sum == 0
        mask_sum  = mask_sum.masked_fill_(no_update, 1.)

        out = (out - bias) / mask_sum + bias
        out = out.masked_fill_(no_update, 0.)

        new_mask = (~no_update).float()
        return out, new_mask


class BatchedDistanceMaskedPointConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, k=9, radius=0.05, bias=True):
        super().__init__()
        self.k = k
        self.radius = radius
        self.core = PartialPointConv2D(in_channels, out_channels, bias=bias)

    @torch.no_grad()
    def _build_edge_index(self, pos_flat, batch_vec):
        ei = knn_graph(pos_flat, k=self.k, batch=batch_vec, loop=True)

        row, col = ei
        dist2 = ((pos_flat[row] - pos_flat[col]).pow(2)).sum(-1)
        mask_edges = dist2 < (self.radius ** 2)
        return ei[:, mask_edges]

    def forward(self, x, pos, mask):
        B, N, _ = x.shape
        x_f    = x.reshape(-1, x.size(-1))       # (B*N, C_in)
        pos_f  = pos.reshape(-1, 2)              # (B*N, 2)
        mask_f = mask.reshape(-1, 1)             # (B*N, 1)
        batch_vec = torch.arange(B, device=x.device).repeat_interleave(N)

        ei = self._build_edge_index(pos_f, batch_vec)
        out_f, new_mask_f = self.core(x_f, ei, mask_f)
        out      = out_f.reshape(B, N, -1)
        new_mask = new_mask_f.reshape(B, N, 1)
        return out, new_mask


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
    def __init__(self, feature_basis_num, temperature=0.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones([1, 1, feature_basis_num]) * temperature)

    def forward(self, x):
        return x / self.temperature


class KernelIntegrator(nn.Module):
    def __init__(self, hidden_size, feature_basis_num, heads_num):
        super().__init__()
        head_size = hidden_size // heads_num
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.ln_4 = nn.LayerNorm(hidden_size)
        self.feature_basis_projector = nn.Sequential(
            MLP(hidden_size, hidden_size, feature_basis_num, n_layers=0, act='gelu', res=False),
            Temperature(feature_basis_num, temperature=0.5),
            nn.Softmax(dim=-1),
        )
        self.to_qkv = nn.Linear(hidden_size, hidden_size*3, bias=False)
        self.attn_mlp = MLP(hidden_size, hidden_size, hidden_size, n_layers=0, act='gelu', res=False)
        #self.conjugate = MLP(head_size, hidden_size, head_size, n_layers=0, act='gelu')
        self.to_out = nn.Linear(hidden_size, hidden_size)
        self.conv = BatchedDistanceMaskedPointConv2D(feature_basis_num, feature_basis_num, k=9, radius=0.04)
        self.feature_basis_num = feature_basis_num
        self.heads_num = heads_num
        self.head_size = head_size
        self.hidden_size = hidden_size

    def compute_feature_map(self, x, no_valid):
        feature_basis = self.feature_basis_projector(x)
        psi = torch.einsum("b n c, b n f -> b f c", x, feature_basis).contiguous()
        feature_basis = feature_basis.masked_fill(no_valid, 0.)
        feature_basis_norm = feature_basis.sum(dim=1)
        psi = psi / (feature_basis_norm + 1e-6)[:, :, None].repeat(1, 1, self.hidden_size)
        return psi, feature_basis

    def map_back_to_originalSpace(self, feature_basis, pos, mask, psi):
        mask_new          = mask [:, :, None]
        feature_basis_new, mask_new = self.conv(feature_basis, pos, mask_new)
        x = torch.einsum("b f c, b n f -> b n c", psi, feature_basis_new)
        mask_new = mask_new.squeeze(-1)
        return x, mask_new

    def get_psi(self, x, mask):
        no_valid = (mask == 0)[:, :, None]
        x = x.masked_fill(no_valid, 0.)
        
        # compute feature map
        psi, _ = self.compute_feature_map(self.ln_1(x), no_valid)
        return psi

    def forward(self, x, pos, mask):
        no_valid = (mask == 0)[:, :, None]
        x = x.masked_fill(no_valid, 0.)
        
        # compute feature map
        psi, feature_basis = self.compute_feature_map(self.ln_1(x), no_valid)

        # conjugate operator
        query, key, value = self.to_qkv(psi).chunk(3, dim = -1)
        query = rearrange(query, 'b f (h c) -> b f h c', h=self.heads_num, c=self.head_size)
        key   = rearrange(key,   'b f (h c) -> b f h c', h=self.heads_num, c=self.head_size)
        value = rearrange(value, 'b f (h c) -> b f h c', h=self.heads_num, c=self.head_size)
        attn_out = memory_efficient_attention(query, key, value)
        attn_out = rearrange(attn_out, 'b f h c -> b f (h c)')
        psi = psi + attn_out
        psi = psi + self.attn_mlp(self.ln_3(psi))

        # map back to original space
        x_new, mask_new = self.map_back_to_originalSpace(feature_basis, pos, mask, psi)
        x = x + self.to_out(x_new)
        return x, mask_new


class OursIrregularModel(nn.Module):
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
                    args.heads_num,
                )
            )
        self.kernelProcessor = nn.ModuleList(self.kernelProcessor)
        self.projector = nn.Sequential(
            nn.LayerNorm(args.hidden_size), 
            nn.Linear(args.hidden_size, args.output_size)
        )
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (args.hidden_size)) * torch.rand(args.hidden_size, dtype=torch.float))

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
        if x is not None:
            z = torch.concat([pos, x], dim=-1)
        else:
            z = pos
        z = self.featureExpander(z)
        z = z + self.placeholder[None, None, :]

        for _, block in enumerate(self.kernelProcessor):
            z, mask = block(z, pos, mask)
    
        y  = self.projector(z)
        return y
