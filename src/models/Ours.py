from __future__ import annotations
import torch
import torch.nn as nn
from einops import rearrange, repeat, einsum
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Dict
from torch import Tensor
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_, orthogonal_


ACTIVATION = {'gelu':nn.GELU,'tanh':nn.Tanh,'sigmoid':nn.Sigmoid,'relu':nn.ReLU,'leaky_relu':nn.LeakyReLU(0.1),'softplus':nn.Softplus,'ELU':nn.ELU,'silu':nn.SiLU}


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


def gram_schmidt(input):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u
    output = []
    for x in input:
        for y in output:
            x = x - projection(y, x)
        x = x/x.norm(p=2)
        output.append(x)
    return torch.stack(output)

class GMEncoding(nn.Module):
    def __init__(self, is_logicalNot=False, is_filter=False):
        super().__init__()
        self.is_logicalNot = is_logicalNot
        self.is_filter     = is_filter

    def forward(self, x, m):
        if self.is_filter:
            B = m.shape[0]
            D = x.shape[0]
            x = repeat(x, 'd ... -> b (...) d', b=B)
        else:
            B, _ , D = x.shape
        m = repeat(m, 'b ... -> b (...) d', d=D).bool()
        if self.is_logicalNot:
            m = ~m
        x_sequence = []
        for i in range(B):
            x_i = x[i]
            m_i = m[i]
            x_i = torch.masked_select(x_i, m_i).reshape(-1, D)
            x_sequence.append(x_i)
        x = pad_sequence(x_sequence).permute(1, 0, 2)
        return x
    

def initialize_orthogonal_filters(c, h, w):

    if h*w < c:
        n = c//(h*w)
        gram = []
        for i in range(n):
            gram.append(gram_schmidt(torch.rand([h * w, 1, h, w])))
        return torch.cat(gram, dim=0)
    else:
        return gram_schmidt(torch.rand([c, 1, h, w]))
    

class GramSchmidtTransform(nn.Module):
    instance: Dict[int, Optional[GramSchmidtTransform]] = {}
    constant_filter: Tensor

    @staticmethod
    def build(c: int, h: int):
        if c not in GramSchmidtTransform.instance:
            GramSchmidtTransform.instance[(c, h)] = GramSchmidtTransform(c, h)
        return GramSchmidtTransform.instance[(c, h)]

    def __init__(self, c: int, h: int):
        super().__init__()
        with torch.no_grad():
            rand_ortho_filters = initialize_orthogonal_filters(c, h, h).view(c, h, h)
        self.register_buffer("constant_filter", rand_ortho_filters.detach())
        self.gme4o = GMEncoding(is_filter=True)
        
    def forward(self, x, m):
        selected_constant_filter_ = self.gme4o(self.constant_filter, m)
        result = selected_constant_filter_ * x
        #result = rearrange(self.constant_filter, 'C H W -> (H W) C') * x
        return result.sum(dim=(-2), keepdim=True).permute(0, 2, 1)
    

class OrthoAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, FWT: GramSchmidtTransform, input: Tensor, m: Tensor):
        #happens once in case of BigFilter
        while input[0].size(-1) > 1:
            input = FWT(input, m)
        b = input.size(0)
        return input.view(b, -1)
    

class MLP(nn.Module):
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


# New position encoding module
# modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
class RotaryEmbedding(nn.Module):
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


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(t, freqs):
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
    # split t into first half and second half
    # t: [b, h, n, d]
    # freq_x/y: [b, n, d]
    d = t.shape[-1]
    t_x, t_y = t[..., :d//2], t[..., d//2:]

    return torch.cat((apply_rotary_pos_emb(t_x, freqs_x),
                      apply_rotary_pos_emb(t_y, freqs_y)), dim=-1)


class CrossLinearAttention(nn.Module):
    def __init__(self,
                 dim,
                 attn_type,  # ['fourier', 'galerkin']
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 init_params=True,
                 relative_emb=False,
                 scale=1.,
                 init_method='orthogonal',  # ['xavier', 'orthogonal']
                 init_gain=None,
                 relative_emb_dim=2,
                 min_freq=1 / 64,  # 1/64 is for 64 x 64 ns2d,
                 cat_pos=False,
                 pos_dim=2,
                 ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.attn_type = attn_type

        self.heads = heads
        self.dim_head = dim_head

        # query is the classification token
        self.to_q  = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        if attn_type == 'galerkin':
            self.k_norm = nn.InstanceNorm1d(dim_head)
            self.v_norm = nn.InstanceNorm1d(dim_head)
        elif attn_type == 'fourier':
            self.q_norm = nn.InstanceNorm1d(dim_head)
            self.k_norm = nn.InstanceNorm1d(dim_head)
        else:
            raise Exception(f'Unknown attention type {attn_type}')

        if not cat_pos:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim + pos_dim*heads, dim),
                nn.Dropout(dropout)
            )

        if init_gain is None:
            self.init_gain = 1. / dim_head
            self.diagonal_weight = 1. / dim_head
        else:
            self.init_gain = init_gain
            self.diagonal_weight = init_gain
        self.init_method = init_method
        if init_params:
            self._init_params()

        self.cat_pos = cat_pos
        self.pos_dim = pos_dim

        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        if relative_emb:
            self.emb_module = RotaryEmbedding(dim_head // self.relative_emb_dim, min_freq=min_freq, scale=scale)

    def _init_params(self):
        if self.init_method == 'xavier':
            init_fn = xavier_uniform_
        elif self.init_method == 'orthogonal':
            init_fn = orthogonal_
        else:
            raise Exception('Unknown initialization')

        for param in self.to_kv.parameters():
            if param.ndim > 1:
                for h in range(self.heads):
                    # for k
                    init_fn(param[h*self.dim_head:(h+1)*self.dim_head, :], gain=self.init_gain)
                    param.data[h*self.dim_head:(h+1)*self.dim_head, :] += self.diagonal_weight * \
                                                                          torch.diag(torch.ones(
                                                                              param.size(-1), dtype=torch.float32))

                    # for v
                    init_fn(param[(self.heads + h) * self.dim_head:(self.heads + h + 1) * self.dim_head, :], gain=self.init_gain)
                    param.data[(self.heads + h) * self.dim_head:(self.heads + h + 1) * self.dim_head, :] += self.diagonal_weight * \
                                                                           torch.diag(torch.ones(
                                                                               param.size(-1), dtype=torch.float32))
                                                                               
        for param in self.to_q.parameters():
            if param.ndim > 1:
                for h in range(self.heads):
                    # for q
                    init_fn(param[h * self.dim_head:(h + 1) * self.dim_head, :], gain=self.init_gain)
                    param.data[h * self.dim_head:(h + 1) * self.dim_head, :] += self.diagonal_weight * \
                                                                                torch.diag(torch.ones(
                                                                                    param.size(-1), dtype=torch.float32))

    def norm_wrt_domain(self, x, norm_fn):
        b = x.shape[0]
        x_ = rearrange(x, 'b h n d -> (b h) d n')
        x_ = norm_fn(x_)
        x = rearrange(x_, '(b h) d n -> b h n d', b=b)
        return x
    

    def forward(self, x, z, x_pos=None, z_pos=None):
        n2 = z.shape[1]   # z [b, n2, d]

        q = self.to_q(x)

        kv = self.to_kv(z).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)

        if (x_pos is None or z_pos is None) and self.relative_emb:
            raise Exception('Must pass in coordinates when under relative position embedding mode')
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        if self.attn_type == 'galerkin':
            k = self.norm_wrt_domain(k, self.k_norm)
            v = self.norm_wrt_domain(v, self.v_norm)
        else:  # fourier
            q = self.norm_wrt_domain(q, self.q_norm)
            k = self.norm_wrt_domain(k, self.k_norm)

        if self.relative_emb:
            if self.relative_emb_dim == 2:

                x_freqs_x = self.emb_module.forward(x_pos[..., 0], x.device)
                x_freqs_y = self.emb_module.forward(x_pos[..., 1], x.device)
                x_freqs_x = repeat(x_freqs_x, 'b n d -> b h n d', h=q.shape[1])
                x_freqs_y = repeat(x_freqs_y, 'b n d -> b h n d', h=q.shape[1])

                z_freqs_x = self.emb_module.forward(z_pos[..., 0], z.device)
                z_freqs_y = self.emb_module.forward(z_pos[..., 1], z.device)
                z_freqs_x = repeat(z_freqs_x, 'b n d -> b h n d', h=q.shape[1])
                z_freqs_y = repeat(z_freqs_y, 'b n d -> b h n d', h=q.shape[1])

                q = apply_2d_rotary_pos_emb(q, x_freqs_x, x_freqs_y)
                k = apply_2d_rotary_pos_emb(k, z_freqs_x, z_freqs_y)

            elif self.relative_emb_dim == 1:
                assert x_pos.shape[-1] == 1 and z_pos.shape[-1] == 1
                x_freqs = self.emb_module.forward(x_pos[..., 0], x.device)
                x_freqs = repeat(x_freqs, 'b n d -> b h n d', h=q.shape[1])

                z_freqs = self.emb_module.forward(z_pos[..., 0], x.device)
                z_freqs = repeat(z_freqs, 'b n d -> b h n d', h=q.shape[1])

                q = apply_rotary_pos_emb(q, x_freqs)  # query from x domain
                k = apply_rotary_pos_emb(k, z_freqs)  # key from z domain
            else:
                raise Exception('Currently doesnt support relative embedding > 2 dimensions')
        elif self.cat_pos:
            assert x_pos.size(-1) == self.pos_dim and z_pos.size(-1) == self.pos_dim
            x_pos = x_pos.unsqueeze(1)
            x_pos = x_pos.repeat([1, self.heads, 1, 1])
            q = torch.cat([x_pos, q], dim=-1)

            z_pos = z_pos.unsqueeze(1)
            z_pos = z_pos.repeat([1, self.heads, 1, 1])
            k = torch.cat([z_pos, k], dim=-1)
            v = torch.cat([z_pos, v], dim=-1)

        dots = torch.matmul(k.transpose(-1, -2), v)

        out = torch.matmul(q, dots) * (1./n2)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)
    

def my_pad_sequence(sequences):
    max_len = max(seq.size(0) for seq in sequences)  # 找到最长的序列长度
    padded_seqs = []

    for seq in sequences:
        pad_size = max_len - seq.size(0)
        padded_seq = torch.cat([seq, torch.zeros(pad_size, seq.size(1)).to(seq.device)])
        padded_seqs.append(padded_seq)
    padded_seqs = torch.stack(padded_seqs, dim=1)
    return padded_seqs


class SelfLinearAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return


class Attention(nn.Module):
    def __init__(
            self, query_channel, context_channel=None, output_channel=None,
            heads_num=8, heads_channel=64, dropout=0.):
        super(Attention, self).__init__()
        inner_channel = heads_channel * heads_num
        context_dim = default(context_channel, query_channel)
        output_dim = default(output_channel, query_channel)

        self.scale = heads_channel ** -0.5
        self.heads = heads_num

        self.to_q = nn.Linear(query_channel, inner_channel, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_channel * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_channel, output_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
    

# code copied from: https://github.com/ndahlquist/pytorch-fourier-feature-networks
# author: Nic Dahlquist
class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, n, num_input_channels],
     returns a tensor of size [batches, n, mapping_size*2].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x):

        batches, num_of_points, channels = x.shape

        # Make shape compatible for matmul with _B.
        # From [B, N, C] to [(B*N), C].
        x = rearrange(x, 'b n c -> (b n) c')

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = rearrange(x, '(b n) c -> b n c', b=batches)

        x = 2 * torch.math.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

def gaussian_weighted_interpolation(dist, values, sigma=0.1): 
    weights = torch.exp(- (dist ** 2) / (2 * sigma ** 2))
    weights /= weights.sum(dim=-1, keepdim=True)
    interpolated_values = torch.einsum('bhnc, bmn -> bhmc', values, weights)
    return interpolated_values


def fill_gap_mut(k_o, v_o, dist):
    dx = dy = 1.0 / 64
    k_grid = gaussian_weighted_interpolation(dist, k_o)
    v_grid = gaussian_weighted_interpolation(dist, v_o)

    dots = torch.einsum('bhnc, bhnd -> bhcnd', k_grid, v_grid)
    dots = rearrange(dots, 'b h c (H W) d -> b h c H W d', H=64, W=64)
    integral_x  = torch.trapz(dots,       dx=dx, dim=-2)
    integral_xy = torch.trapz(integral_x, dx=dy, dim=-2)
    return integral_xy


class FGBlock(nn.Module):
    def __init__(self, embedding_channels, attn_channels, num_heads, fourier_frequency=10, min_freq=1/64, act='gelu', relative_emb_dim=2, is_fillGap=False, project_out=False):
        super().__init__()
        self.to_qkv = nn.Linear(embedding_channels, attn_channels*3, bias=False)
        #self.gme4m = GMEncoding(is_logicalNot=True)
        """
        self.coordinate_embedding = nn.Sequential(
            GaussianFourierFeatureTransform(2, embedding_channels//2, scale=fourier_frequency),
            nn.Linear(embedding_channels, embedding_channels, bias=False),
            ACTIVATION[act](),
            nn.Linear(embedding_channels, embedding_channels, bias=False),
        )
        """
        head_channel = attn_channels//num_heads
        self.k_norm = nn.InstanceNorm1d(head_channel)
        self.v_norm = nn.InstanceNorm1d(head_channel)
        self.coordinate_embedding = RotaryEmbedding(head_channel//relative_emb_dim, min_freq=min_freq, scale=fourier_frequency)
        self.to_out = nn.Linear(attn_channels, embedding_channels) if project_out else nn.Identity()    
        self.num_heads = num_heads
        self.is_fillGap = is_fillGap
    
    def forward(self, z_o, p_o, dist):
        # observe p/q/k/v
        qkv_o = self.to_qkv(z_o).chunk(3, dim=-1)
        q_o, k_o, v_o = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv_o)

        k_o = self.norm_wrt_domain(k_o, self.k_norm)
        v_o = self.norm_wrt_domain(v_o, self.v_norm)
        freqs_x = self.coordinate_embedding.forward(p_o[..., 0], z_o.device)
        freqs_y = self.coordinate_embedding.forward(p_o[..., 1], z_o.device)
        freqs_x = repeat(freqs_x, 'b n d -> b h n d', h=q_o.shape[1])
        freqs_y = repeat(freqs_y, 'b n d -> b h n d', h=q_o.shape[1])

        q_o = apply_2d_rotary_pos_emb(q_o, freqs_x, freqs_y)
        k_o = apply_2d_rotary_pos_emb(k_o, freqs_x, freqs_y)
        """
        # unobserve p/q
        p_m = self.gme4m(p, m)
        q_m = self.coordinate_embedding(p_m)
        q_m = rearrange(q_m, 'b n (h d) -> b h n d', h=self.num_heads)
        """

        # fill-gap kv
        if self.is_fillGap:
            dots = fill_gap_mut(k_o, v_o, dist)
        else:
            dots = torch.matmul(k_o.transpose(-1, -2), v_o)
        out = torch.matmul(q_o, dots) * (1./q_o.shape[2])
        out = rearrange(out, 'b h n d -> b n (h d)')
        z_o = self.to_out(out)
        return z_o
    
    def norm_wrt_domain(self, x, norm_fn):
        b = x.shape[0]
        x_ = rearrange(x, 'b h n d -> (b h) d n')
        x_ = norm_fn(x_)
        x = rearrange(x_, '(b h) d n -> b h n d', b=b)
        return x


class SpatialFeatExtraction(nn.Module):
    def __init__(self, num_layers, scale, embedding_channels, is_fillGap):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.gme4o = GMEncoding()
        
        for layer_index in range(num_layers):
            fg_block = FGBlock(embedding_channels, embedding_channels, 1, is_fillGap=is_fillGap, fourier_frequency=scale[layer_index])
            block = nn.ModuleList([
                nn.LayerNorm(embedding_channels), 
                fg_block, 
                nn.LayerNorm(embedding_channels), 
                MLP(embedding_channels, 2*embedding_channels, embedding_channels)
            ])
            self.blocks.append(block)

    def forward(self, z_o, p_o, complete_pos, dist=None):
        if dist is None:
            dist = torch.cdist(complete_pos, p_o)
        for _, block in enumerate(self.blocks):
            ln1, fg_block, ln2, project = block
            z_o = ln1(z_o)
            z_o = fg_block(z_o, p_o, dist) + z_o
            z_o = ln2(z_o)
            z_o = project(z_o) + z_o
        return z_o


class PreNorm(nn.Module):
    def __init__(self, channel, fn, context_channel=None):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(channel)
        self.norm_context = nn.LayerNorm(context_channel) if exists(context_channel) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)
    

class FGEncoder(nn.Module):
    def __init__(
        self, 
        input_channels, 
        embedding_channels, 
        encoding_channels, 
        num_layers, 
        is_fillGap, 
        scale
    ):
        super().__init__()
        self.temporalFeat_extract = nn.Linear(input_channels, embedding_channels, bias=False)
        self.spatialFeat_extract  = SpatialFeatExtraction(
                                            num_layers, 
                                            scale, 
                                            embedding_channels, 
                                            is_fillGap
                                        )
        self.project              = nn.Linear(embedding_channels, encoding_channels, bias=False)
        
    def forward(self, x, input_pos, complete_pos, dist=None):
        z = self.temporalFeat_extract(x)
        z = self.spatialFeat_extract(z, input_pos, complete_pos, dist=dist)
        z = self.project(z)
        return z
    

class TEProcessor(nn.Module):
    def __init__(self, encoding_channels, num_layers, act='gelu', is_OrthoAttention=True):
        super().__init__()
        self.temporal_evolution_layers = nn.ModuleList([
                nn.ModuleList([
                    nn.LayerNorm(encoding_channels),
                    nn.Sequential(
                        nn.Linear(encoding_channels+2, encoding_channels, bias=False),
                        ACTIVATION[act](),
                        nn.Linear(encoding_channels, encoding_channels, bias=False),
                        ACTIVATION[act](),
                        nn.Linear(encoding_channels, encoding_channels, bias=False)
                    )
                ])
            for _ in range(num_layers)
        ])
        self._excitation = nn.Sequential(
            nn.Linear(in_features=encoding_channels, out_features=round(encoding_channels/ 16), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=round(encoding_channels/ 16), out_features=encoding_channels, bias=False),
            nn.Sigmoid(),
        )
        self.OrthoAttention = OrthoAttention()
        self.weights = GramSchmidtTransform.build(encoding_channels, 64)
        self.is_OrthoAttention = is_OrthoAttention

    def forward(self, z, p, m2):
        for layer in self.temporal_evolution_layers:
            norm_fn, ffn = layer
            residual = z
            z = ffn(torch.cat([norm_fn(z), p], dim=-1))
            if self.is_OrthoAttention:
                compressed = self.OrthoAttention(self.weights, z, m2)
                b, _, c = z.shape
                excitation = self._excitation(compressed).view(b, 1, c)
                z = excitation * z
            z += residual
        return z
    

class SpatialInteraction(nn.Module):
    def __init__(self, decoding_channels, num_heads, heads_channels, relative_emb_dim, scale, cat_pos=False, min_freq=1/64, relative_emb=True, dropout=0., attn_type='galerkin'):
        super().__init__()
        self.cross_attn_module = CrossLinearAttention(
            decoding_channels, attn_type, 
            heads=num_heads, dim_head=heads_channels, 
            dropout=dropout, relative_emb=relative_emb,
            scale=scale, relative_emb_dim=relative_emb_dim,
            min_freq=min_freq, init_method='orthogonal',
            cat_pos=cat_pos, pos_dim=relative_emb_dim,
        )
        self.project = MLP(decoding_channels, decoding_channels*2, decoding_channels)
    def forward(self, x, z, output_pos, p_o):
        x = self.cross_attn_module(x, z, output_pos, p_o) + x
        x = self.project(x) + x
        return x


class CrossAttn(nn.Module):
    def __init__(self, decoding_channels, num_heads, scale=16., act='gelu', relative_emb_dim=2, min_freq=1/64, fourier_frequency=10):
        super().__init__()
        heads_channels = decoding_channels//2
        self.coordinate_embedding = nn.Sequential(
            GaussianFourierFeatureTransform(2, decoding_channels//2, scale=fourier_frequency),
            nn.Linear(decoding_channels, decoding_channels, bias=False),
            ACTIVATION[act](),
            nn.Linear(decoding_channels, decoding_channels//2, bias=False),
        )
        self.spatial_interaction = SpatialInteraction(decoding_channels//2, num_heads, heads_channels, relative_emb_dim, scale, min_freq=min_freq)

    def forward(self, z, output_pos, p_o):
        x = self.coordinate_embedding(output_pos)
        y = self.spatial_interaction(x, z, output_pos, p_o)
        return y
    

class OursModel(nn.Module):
    def __init__(self, 
                 T_in=10, 
                 is_fillGap=True, 
                 is_OrthoAttention=True, 
                 outputs_timeStep=4
                 ):
        super().__init__()
        self.encoder       = FGEncoder(T_in+2, 64, 128, 5, is_fillGap=is_fillGap, scale=[32, 16, 8, 8, 1])
        self.expand_feat   = nn.Linear(128, 256, bias=True)
        self.processor     = TEProcessor(256, 1, is_OrthoAttention=is_OrthoAttention)
        #self.compress_feat = nn.Linear(256, 128, bias=True)
        #self.crossAttn     = CrossAttn(512, 4)
        self.crossAttn     = CrossAttn(256, 4)
        self.decoder       = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128, bias=False),
            nn.GELU(),
            nn.Linear(128, 128, bias=False),
            nn.GELU(),
            nn.Linear(128, outputs_timeStep, bias=True),
        )


    def forward(self, x, m, m2, input_pos, output_pos, complete_pos, forward_steps=None, dist=None):
        if x.dim()>3:
            x = rearrange(x, 'b ... c -> b (...) c')

        z_ = self.encoder(x, input_pos, complete_pos, dist=dist)
        z_ = self.crossAttn(z_, output_pos, input_pos)
        z  = self.expand_feat(z_)
        
        if forward_steps is not None:
            y_trajectory = []
            for _ in range(forward_steps):
                #z  = self.processor(z, input_pos, m2)
                z  = self.processor(z, output_pos, m2)
                #z_ = self.compress_feat(z)
                y  = self.decoder(z)
                z  = self.processor(z, input_pos, m2)
                #z_ = self.compress_feat(z)
                z_  = self.crossAttn(z, output_pos, input_pos)
            z  = self.processor(z, input_pos, m2)
            #z_ = self.compress_feat(z)
            y  = self.decoder(z_, output_pos, input_pos)
            return y
