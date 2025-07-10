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


class neighborConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        self.mask_conv.weight.requires_grad = False

    def forward(self, mask):
        with torch.no_grad():
            output_mask = self.mask_conv(mask)
        inner_holes = output_mask == 0
        new_mask = torch.ones_like(output_mask)
        new_mask = new_mask.masked_fill_(inner_holes, 0.0)
        return new_mask


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


class PhCA_Encoder(nn.Module):
    def __init__(self, hidden_size, heads_num, latent_num):
        super().__init__()
        self.head_size = hidden_size // heads_num
        self.heads_num = heads_num
        self.attention_encoder = MLP(self.head_size, latent_num, latent_num, n_layers=0, act='gelu', res=False)
        self.temperature = Temperature(heads_num, temperature=0.5)

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.0002)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(self, y, no_valid):
        y = rearrange(y, 'b n (h c) -> b h n c', h=self.heads_num, c=self.head_size)
        score_encode = self.attention_encoder(y)
        score_encode = self.temperature(score_encode)
        score_encode = torch.softmax(score_encode, dim=-1)

        z = torch.einsum("bhnl, bhnc -> bhlc", score_encode, y).contiguous()
        no_valid = rearrange(no_valid, 'b n (1 1) -> b 1 n 1')
        score_encode = score_encode.masked_fill(no_valid, 0.)
        
        score_encode_norm = score_encode.sum(dim=2)
        z = z / (score_encode_norm + 1e-6)[:, :, :, None]
        z = rearrange(z, 'b h l c -> b l (h c)')
        return z, score_encode
    

class PhCA_Decoder(nn.Module):
    def __init__(self, hidden_size, heads_num, latent_num):
        super().__init__()
        self.heads_num = heads_num
        self.head_size = hidden_size // heads_num
        #self.attention_decoder = MLP(hidden_size, hidden_size, latent_num, n_layers=0, act='gelu', res=False)

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.0002)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(self, z, score):
        z = rearrange(z, 'b l (h c) -> b h l c', h=self.heads_num, c=self.head_size)
        y = torch.einsum("bhnl, bhlc -> bhnc", score, z)
        y = rearrange(y, 'b h n c -> b n (h c)')
        return y
    

class PhLP(nn.Module):
    def __init__(self, hidden_size, latent_num, heads_num, token_Mixer, space_size):
        super().__init__()
        head_size = hidden_size // heads_num
        self.head_size   = head_size
        self.heads_num   = heads_num
        self.token_Mixer = token_Mixer
        self.space_size  = space_size
        self.latent_num  = latent_num

        #self.trunk_projector = MLP(hidden_size, hidden_size, hidden_size, n_layers=0, act='gelu', res=False)
        #self.branch_projector = MLP(hidden_size, hidden_size, hidden_size, n_layers=0, act='gelu', res=False)

        self.phca_encoder = PhCA_Encoder(hidden_size, heads_num, latent_num)
        self.phca_decoder = PhCA_Decoder(hidden_size, heads_num, latent_num)
        
        if token_Mixer == 'Attention':
            self.to_qkv = nn.Linear(head_size, head_size*3, bias=False)
        if token_Mixer == 'MLP':
            self.conjugate = MLP(latent_num, latent_num*2, latent_num, n_layers=0, act='gelu')
        #self.neighbor = neighborConv()
        self.conv = PartialConv(heads_num*latent_num, heads_num*latent_num, 3, padding=1)

    def forward(self, y, mask):
        # encoder
        no_valid = mask == 0
        y_in = y.masked_fill(no_valid, 0.)
        z, score = self.phca_encoder(y_in, no_valid)

        # conjugate operator
        if self.token_Mixer == 'Attention':
            z = rearrange(z, 'b f (h c) -> b f h c', h=self.heads_num, c=self.head_size)
            query, key, value = self.to_qkv(z).chunk(3, dim = -1)
            z = memory_efficient_attention(query, key, value)
            z = rearrange(z, 'b f h c -> b f (h c)')
        elif self.token_Mixer == 'MLP':
            z = z.permute(0, 2, 1)
            z = self.conjugate(z)
            z = z.permute(0, 2, 1)
        
        # calculate score
        next_score = rearrange(score, 'b h (H W) l -> b (h l) H W', H=self.space_size[0], W=self.space_size[1])
        next_mask  = rearrange(mask, 'b (H W) 1 -> b 1 H W', H=self.space_size[0], W=self.space_size[1])
        next_mask  = next_mask.repeat(1, self.heads_num*self.latent_num, 1, 1)
        next_score, next_mask = self.conv(next_score, next_mask)
        next_score = rearrange(next_score, 'b (h l) H W -> b h (H W) l', h=self.heads_num, l=self.latent_num)
        next_mask  = rearrange(next_mask, 'b c H W -> b (H W) c')[..., :1]
        # decoder
        y_out = self.phca_decoder(z, next_score)
        return y_out, next_mask
    

class KernelIntegrator(nn.Module):
    def __init__(self, hidden_size, latent_num, heads_num, token_Mixer='Attention', space_size=(64,64)):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.phlp = PhLP(hidden_size, latent_num, heads_num, token_Mixer=token_Mixer, space_size=space_size)
        self.mlp  = MLP(hidden_size, hidden_size, hidden_size, n_layers=0, act='gelu', res=False)
        
    def forward(self, y, mask):
        # PhLP
        y_, new_mask = self.phlp(self.ln_1(y), mask)
        y  = y_ + y
        # mlp
        y = self.mlp(self.ln_2(y)) + y
        return y, new_mask


class OursLNOModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.featureExpander = MLP(
            args.input_size + args.ref*args.ref, 
            args.hidden_size * 2, args.hidden_size, 
            n_layers=0, act='gelu', res=False
        )
        self.kernelProcessor = []
        h = int((args.space_size[0] / args.downsample))
        w = int((args.space_size[1] / args.downsample))
        for _ in range(args.kernel_layers):
            self.kernelProcessor.append(
                KernelIntegrator(
                    args.hidden_size, 
                    args.latent_num, 
                    args.heads_num,
                    token_Mixer=args.token_Mixer,
                    space_size=(h, w)
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

    def forward(self, pos, x, mask):
        if pos.dim()>3:
            pos = rearrange(pos, 'b ... c -> b (...) c')
        y = torch.concat([pos, x], dim=-1)
        y = self.featureExpander(y)

        for _, block in enumerate(self.kernelProcessor):
            y, mask = block(y, mask)
    
        x  = self.projector(y)
        return x
