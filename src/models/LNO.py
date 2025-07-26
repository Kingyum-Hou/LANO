import torch
import math
from einops import rearrange, reduce
import torch.nn.functional as F
from torch import nn, einsum


class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask = None, return_attn = False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value = 0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value = False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks = q_landmarks / divisor
        k_landmarks = k_landmarks / divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values

        if self.residual:
            out = out + self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out


def Attention_Vanilla(q, k, v):
    score = torch.softmax(torch.einsum("bhic,bhjc->bhij", q, k) / math.sqrt(k.shape[-1]), dim=-1)
    r = torch.einsum("bhij,bhjc->bhic", score, v)
    return r


def Attention_Linear_GNOT(q, k, v):
    q = q.softmax(dim=-1)
    k = k.softmax(dim=-1)
    k_sum = k.sum(dim=-2, keepdim=True)
    inv = 1. / (q * k_sum).sum(dim=-1, keepdim=True)
    r = q + (q @ (k.transpose(-2, -1) @ v)) * inv
    return r


class LinearAttention_Galerkin_and_Fourier(torch.nn.Module):
    def __init__(self, attn_type, n_dim, n_head, use_ln=False):
        super().__init__()
        self.attn_type = attn_type
        self.n_dim = n_dim
        self.n_head = n_head
        self.dim_head = self.n_dim // self.n_head
        self.use_ln = use_ln
        self.to_qkv = torch.nn.Linear(n_dim, n_dim*3, bias = False)
        self.project_out = (not self.n_head == 1)

        if attn_type == 'galerkin':
            if not self.use_ln:
                self.k_norm = torch.nn.InstanceNorm1d(self.dim_head)
                self.v_norm = torch.nn.InstanceNorm1d(self.dim_head)
            else:
                self.k_norm = torch.nn.LayerNorm(self.dim_head)
                self.v_norm = torch.nn.LayerNorm(self.dim_head)

        elif attn_type == 'fourier':
            if not self.use_ln:
                self.q_norm = torch.nn.InstanceNorm1d(self.dim_head)
                self.k_norm = torch.nn.InstanceNorm1d(self.dim_head)
            else:
                self.q_norm = torch.nn.LayerNorm(self.dim_head)
                self.k_norm = torch.nn.LayerNorm(self.dim_head)

        else:
            raise Exception(f'Unknown attention type {attn_type}')

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(self.n_dim, self.n_dim),
            torch.nn.Dropout(0.0)
        ) if self.project_out else torch.nn.Identity()

    def norm_wrt_domain(self, x, norm_fn):
        b = x.shape[0]
        return rearrange(
            norm_fn(rearrange(x, 'b h n d -> (b h) n d')),
            '(b h) n d -> b h n d', b=b)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_head), qkv)

        if self.attn_type == 'galerkin':
            k = self.norm_wrt_domain(k, self.k_norm)
            v = self.norm_wrt_domain(v, self.v_norm)
        elif self.attn_type == "fourier":
            q = self.norm_wrt_domain(q, self.q_norm)
            k = self.norm_wrt_domain(k, self.k_norm)
        else:
            raise NotImplementedError("Invalid Attention Type!")

        dots = torch.matmul(k.transpose(-1, -2), v)
        out = torch.matmul(q, dots) * (1./q.shape[2])
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


ACTIVATION = {"Sigmoid": torch.nn.Sigmoid(),
              "Tanh": torch.nn.Tanh(),
              "ReLU": torch.nn.ReLU(),
              "LeakyReLU": torch.nn.LeakyReLU(0.1),
              "ELU": torch.nn.ELU(),
              "GELU": torch.nn.GELU()
              }


ATTENTION = {"Attention_Vanilla": Attention_Vanilla,
             "Attention_Linear_GNOT": Attention_Linear_GNOT
            }


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer, act):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layer = n_layer
        self.act = act
        self.input = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden = torch.nn.ModuleList([torch.nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_layer)])
        self.output = torch.nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, x):
        r = self.act(self.input(x))
        for i in range(0, self.n_layer):
            r = r + self.act(self.hidden[i](r))
        r = self.output(r)
        return r


class LNO(torch.nn.Module):
    class SelfAttention(torch.nn.Module):
        def __init__(self, n_mode, n_dim, n_head, attn):
            super().__init__()
            self.n_mode = n_mode
            self.n_dim = n_dim
            self.n_head = n_head
            self.Wq = torch.nn.Linear(self.n_dim, self.n_dim)
            self.Wk = torch.nn.Linear(self.n_dim, self.n_dim)
            self.Wv = torch.nn.Linear(self.n_dim, self.n_dim)
            self.attn = attn
            self.proj = torch.nn.Linear(self.n_dim, self.n_dim)
        
        def forward(self, x):
            B, N, D = x.size()
            q = self.Wq(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            k = self.Wk(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            v = self.Wv(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            r = self.attn(q, k, v).permute(0, 2, 1, 3).contiguous().view(B, N, D)
            r = self.proj(r)
            return r
    
    class AttentionBlock(torch.nn.Module):
        def __init__(self, n_mode, n_dim, n_head, attn, act):
            super().__init__()
            self.n_mode = n_mode
            self.n_dim = n_dim
            self.n_head = n_head
            self.attn = attn
            self.act = act
            
            if self.attn == "Galerkin":
                self.self_attn = LinearAttention_Galerkin_and_Fourier('galerkin', self.n_dim, self.n_head, use_ln=True)
            elif self.attn == "Nystrom":
                self.self_attn = NystromAttention(self.n_dim, heads =self.n_head, dim_head=self.n_dim//self.n_head, dropout=0.0)
            else:
                self.self_attn = LNO.SelfAttention(self.n_mode, self.n_dim, self.n_head, ATTENTION[self.attn])
            
            self.ln1 = torch.nn.LayerNorm(self.n_dim)
            self.ln2 = torch.nn.LayerNorm(self.n_dim)
            self.drop = torch.nn.Dropout(0.0)
            
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.n_dim, self.n_dim*2),
                self.act,
                torch.nn.Linear(self.n_dim*2, self.n_dim),
            )

        def forward(self, y):   
            y = y + self.drop(self.self_attn(self.ln1(y)))
            y = y + self.mlp(self.ln2(y))
            return y

        
    def __init__(self, n_block, n_mode, n_dim, n_head, n_layer, x_dim, y1_dim, y2_dim, attn, act, model_attr):
        super().__init__()
        self.n_block = n_block
        self.n_mode = n_mode
        self.n_dim = n_dim
        self.n_head = n_head
        self.n_layer = n_layer
        self.act = ACTIVATION[act]
        
        self.x_dim = x_dim
        self.y1_dim = y1_dim
        if model_attr["time"]:
            self.y2_dim = 1
        else:
            self.y2_dim = y2_dim
        
        self.trunk_projector = MLP(self.x_dim, self.n_dim, self.n_dim, self.n_layer, self.act)
        self.branch_projector = MLP(self.y1_dim, self.n_dim, self.n_dim, self.n_layer, self.act)
        self.out_mlp = MLP(self.n_dim, self.n_dim, self.y2_dim, self.n_layer, self.act)
        self.attention_projector = MLP(self.n_dim, self.n_dim, self.n_mode, self.n_layer, self.act)
        self.attn_blocks = torch.nn.Sequential(*[LNO.AttentionBlock(self.n_mode, self.n_dim, self.n_head, attn, self.act) for _ in range(0, self.n_block)])
        
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.0002)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(self, x, y):
        x = self.trunk_projector(x)
        y = self.branch_projector(y)

        score = self.attention_projector(x)
        score_encode = torch.softmax(score, dim=1)
        score_decode = torch.softmax(score, dim=-1)
        
        z = torch.einsum("bij,bic->bjc", score_encode, y)
        
        for block in self.attn_blocks:
            z = block(z)
        
        r = torch.einsum("bij,bjc->bic", score_decode, z)
        r = self.out_mlp(r)
        return r


class LNO_single(torch.nn.Module):
    class SelfAttention(torch.nn.Module):
        def __init__(self, n_mode, n_dim, n_head, attn):
            super().__init__()
            self.n_mode = n_mode
            self.n_dim = n_dim
            self.n_head = n_head
            self.Wq = torch.nn.Linear(self.n_dim, self.n_dim, bias=False)
            self.Wk = torch.nn.Linear(self.n_dim, self.n_dim, bias=False)
            self.Wv = torch.nn.Linear(self.n_dim, self.n_dim, bias=False)
            self.attn = attn
            self.proj = torch.nn.Linear(self.n_dim, self.n_dim, bias=False)
        
        def forward(self, x):
            B, N, D = x.size()
            q = self.Wq(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            k = self.Wk(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            v = self.Wv(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            r = self.attn(q, k, v).permute(0, 2, 1, 3).contiguous().view(B, N, D)
            r = self.proj(r)
            return r
    
    class AttentionBlock(torch.nn.Module):
        def __init__(self, n_mode, n_dim, n_head, attn, act):
            super().__init__()
            self.n_mode = n_mode
            self.n_dim = n_dim
            self.n_head = n_head
            self.attn = attn
            self.act = act
            
            self.self_attn = LNO_single.SelfAttention(self.n_mode, self.n_dim, self.n_head, self.attn)
            self.ln1 = torch.nn.LayerNorm(self.n_dim)
            self.ln2 = torch.nn.LayerNorm(self.n_dim)
            self.drop = torch.nn.Dropout(0.0)
            
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.n_dim, self.n_dim*2),
                self.act,
                torch.nn.Linear(self.n_dim*2, self.n_dim),
            )

        def forward(self, y):   
            y = y + self.drop(self.self_attn(self.ln1(y)))
            y = y + self.mlp(self.ln2(y))
            return y


    def __init__(self, n_block, n_mode, n_dim, n_head, n_layer, x_dim, y1_dim, y2_dim, attn, act, model_attr):
        super().__init__()
        self.n_block = n_block
        self.n_mode = n_mode
        self.n_dim = n_dim
        self.n_head = n_head
        self.n_layer = n_layer

        self.attn = ATTENTION[attn]
        self.act = ACTIVATION[act]
        
        self.x_dim = x_dim
        self.y1_dim = y1_dim
        if model_attr["time"]:
            self.y2_dim = 1
        else:
            self.y2_dim = y2_dim
        
        self.in_mlp = MLP(self.y1_dim, self.n_dim, self.n_dim, self.n_layer, self.act)
        self.out_mlp = MLP(self.n_dim, self.n_dim, self.y2_dim, self.n_layer, self.act)
        self.Wm = torch.nn.Linear(self.n_dim, self.n_mode, bias=False)
        
        self.attn_blocks = torch.nn.Sequential(*[LNO_single.AttentionBlock(self.n_mode, self.n_dim, self.n_head, self.attn, self.act) for _ in range(0, self.n_block)])
        
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.0002)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(self, y):
        y = self.in_mlp(y)
        weight = torch.softmax(self.Wm(y), dim=-1)
        y = torch.einsum("bij,bic->bjc", weight, y) / torch.sum(weight, dim=-2, keepdim=True)
        
        for block in self.attn_blocks:
            y = block(y)

        r = torch.einsum("bij,bjc->bic", weight, y)
        r = self.out_mlp(r)
        return r
    

class LNO_triple(torch.nn.Module):
    class SelfAttention(torch.nn.Module):
        def __init__(self, n_mode, n_dim, n_head, attn):
            super().__init__()
            self.n_mode = n_mode
            self.n_dim = n_dim
            self.n_head = n_head
            self.Wq = torch.nn.Linear(self.n_dim, self.n_dim)
            self.Wk = torch.nn.Linear(self.n_dim, self.n_dim)
            self.Wv = torch.nn.Linear(self.n_dim, self.n_dim)
            self.attn = attn
            self.proj = torch.nn.Linear(self.n_dim, self.n_dim)
        
        def forward(self, x):
            B, N, D = x.size()
            q = self.Wq(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            k = self.Wk(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            v = self.Wv(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            r = self.attn(q, k, v).permute(0, 2, 1, 3).contiguous().view(B, N, D)
            r = self.proj(r)
            return r
    
    class CrossAttention(torch.nn.Module):
        def __init__(self, n_mode, n_dim, n_head, attn):
            super().__init__()
            self.n_mode = n_mode
            self.n_dim = n_dim
            self.n_head = n_head
            self.Wq = torch.nn.Linear(self.n_dim, self.n_dim)
            self.Wk = torch.nn.Linear(self.n_dim, self.n_dim)
            self.Wv = torch.nn.Linear(self.n_dim, self.n_dim)
            self.attn = attn
            self.proj = torch.nn.Linear(self.n_dim, self.n_dim)
        
        def forward(self, y, x):
            B, N, D = x.size()
            _, M, _ = y.size()
            q = self.Wq(y).view(B, M, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            k = self.Wk(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            v = self.Wv(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            r = self.attn(q, k, v).permute(0, 2, 1, 3).contiguous().view(B, M, D)
            r = self.proj(r)
            return r
    
    class AttentionBlock(torch.nn.Module):
        def __init__(self, n_mode, n_dim, n_head, attn, act):
            super().__init__()
            self.n_mode = n_mode
            self.n_dim = n_dim
            self.n_head = n_head
            self.attn = attn
            self.act = act
            
            if self.attn == "Galerkin":
                self.self_attn = LinearAttention_Galerkin_and_Fourier('galerkin', self.n_dim, self.n_head, use_ln=True)
            elif self.attn == "Nystrom":
                self.self_attn = NystromAttention(self.n_dim, heads =self.n_head, dim_head=self.n_dim//self.n_head, dropout=0.0)
            else:
                self.self_attn = LNO.SelfAttention(self.n_mode, self.n_dim, self.n_head, ATTENTION[self.attn])
            
            self.ln1 = torch.nn.LayerNorm(self.n_dim)
            self.ln2 = torch.nn.LayerNorm(self.n_dim)
            self.drop = torch.nn.Dropout(0.0)
            
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.n_dim, self.n_dim*2),
                self.act,
                torch.nn.Linear(self.n_dim*2, self.n_dim),
            )

        def forward(self, y):   
            y = y + self.drop(self.self_attn(self.ln1(y)))
            y = y + self.mlp(self.ln2(y))
            return y

        
    def __init__(self, n_block, n_mode, n_dim, n_head, n_layer, x_dim, y1_dim, y2_dim, attn, act, model_attr):
        super().__init__()
        self.n_block = n_block
        self.n_mode = n_mode
        self.n_dim = n_dim
        self.n_head = n_head
        self.n_layer = n_layer
        self.act = ACTIVATION[act]
        
        self.x_dim = x_dim
        self.y1_dim = y1_dim
        if model_attr["time"]:
            self.y2_dim = 1
        else:
            self.y2_dim = y2_dim
        
        self.trunk_projector = MLP(self.x_dim, self.n_dim, self.n_dim, self.n_layer, self.act)
        self.branch_projector = MLP(self.y1_dim, self.n_dim, self.n_dim, self.n_layer, self.act)
        self.out_mlp = MLP(self.n_dim, self.n_dim, self.y2_dim, self.n_layer, self.act)
        self.attention_encoder = MLP(self.n_dim, self.n_dim, self.n_mode, self.n_layer, self.act)
        self.attention_decoder = MLP(self.n_dim, self.n_dim, self.n_mode, self.n_layer, self.act)
        self.attn_blocks = torch.nn.Sequential(*[LNO.AttentionBlock(self.n_mode, self.n_dim, self.n_head, attn, self.act) for _ in range(0, self.n_block)])
        
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.0002)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(self, x1, x2, y):
        x1 = self.trunk_projector(x1)
        x2 = self.trunk_projector(x2)
        y = self.branch_projector(y)

        score_encode = self.attention_encoder(x1)
        score_encode = torch.softmax(score_encode, dim=1)
        
        score_decode = self.attention_decoder(x2)
        score_decode = torch.softmax(score_decode, dim=-1)
        
        z = torch.einsum("bij,bic->bjc", score_encode, y)
        
        for block in self.attn_blocks:
            z = block(z)
        
        r = torch.einsum("bij,bjc->bic", score_decode, z)
        r = self.out_mlp(r)
        return r
