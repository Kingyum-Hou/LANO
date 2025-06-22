import torch
import torch.nn as nn
from torch_scatter import scatter_add          # pip install torch-scatter
from torch_cluster import knn_graph            # pip install torch-cluster

class PartialPointConv2D(nn.Module):
    """内核层：假定输入已展平，批次信息靠 batch 向量。"""
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.kaiming_normal_(self.lin.weight, nonlinearity='relu')

    def forward(self, x, edge_index, mask):
        """
        x       : (N, C_in)     已乘 mask 前的原始特征
        edge_index : (2, E)     有向邻接表 (col → row)
        mask    : (N, 1)        1=有效，0=缺失
        """
        N = x.size(0)
        x_masked = x * mask                    # (N, C_in)
        feat = self.lin(x_masked)              # (N, C_out)

        row, col = edge_index
        out      = scatter_add(feat[row], col, dim=0, dim_size=N)
        mask_sum = scatter_add(mask[row], col, dim=0, dim_size=N)

        bias      = (self.lin.bias if self.lin.bias is not None
                     else torch.zeros(out.size(1), device=x.device)).view(1, -1)
        no_update = mask_sum == 0
        mask_sum  = mask_sum.masked_fill_(no_update, 1.)

        out = (out - bias) / mask_sum + bias
        out = out.masked_fill_(no_update, 0.)

        new_mask = (~no_update).float()
        return out, new_mask


class BatchedDistanceMaskedPointConv2D(nn.Module):
    """
    带“k‑NN + 距离阈值”邻接策略的 2‑D Partial Convolution（批量接口）

    参数
    ----
    k       : k‑NN 邻居上限
    radius  : 距离阈值 (同坐标单位)，仅保留 < radius 的边
    """
    def __init__(self, in_channels, out_channels, k=16, radius=0.05, bias=True):
        super().__init__()
        self.k = k
        self.radius = radius
        self.core = PartialPointConv2D(in_channels, out_channels, bias=bias)

    @torch.no_grad()
    def _build_edge_index(self, pos_flat, batch_vec):
        # 1. 固定 k 的近邻（含自环）
        ei = knn_graph(pos_flat, k=self.k, batch=batch_vec, loop=True)

        # 2. 距离过滤
        row, col = ei
        dist2 = ((pos_flat[row] - pos_flat[col]).pow(2)).sum(-1)
        mask_edges = dist2 < (self.radius ** 2)
        return ei[:, mask_edges]

    def forward(self, x, pos, mask):
        """
        x    : (B, N, C_in)
        pos  : (B, N, 2)
        mask : (B, N, 1)
        """
        B, N, _ = x.shape
        # 展平
        x_f    = x.reshape(-1, x.size(-1))       # (B*N, C_in)
        pos_f  = pos.reshape(-1, 2)              # (B*N, 2)
        mask_f = mask.reshape(-1, 1)             # (B*N, 1)
        batch_vec = torch.arange(B, device=x.device).repeat_interleave(N)

        # 构建邻接表
        ei = self._build_edge_index(pos_f, batch_vec)

        # 核心卷积
        out_f, new_mask_f = self.core(x_f, ei, mask_f)

        # 还原
        out      = out_f.reshape(B, N, -1)
        new_mask = new_mask_f.reshape(B, N, 1)
        return out, new_mask
    

if __name__ == "__main__":
    B, N = 2, 700                       # 两幅散点图，每幅 700 点
    x    = torch.randn(B, N, 32)        # 点特征
    pos  = torch.rand(B, N, 2)          # 归一化 2‑D 坐标 [0,1]^2
    mask = (torch.rand(B, N, 1) > 0.3).float()  # 30 % 缺失

    layer = BatchedDistanceMaskedPointConv2D(32, 64, k=16, radius=0.05)
    out, new_mask = layer(x, pos, mask)
    print(out.shape)        # torch.Size([2, 700, 64])
    print(new_mask.sum())   # 有效输出点数
