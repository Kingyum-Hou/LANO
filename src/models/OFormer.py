from torch import nn
from models.nn_module4OFormer.encoder_module import SpatialTemporalEncoder2D
from models.nn_module4OFormer.decoder_module import PointWiseDecoder2D


class OFormer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = SpatialTemporalEncoder2D(
            args.in_channels,
            args.encoder_emb_dim,
            args.out_seq_emb_dim,
            args.encoder_heads,
            args.encoder_depth,
        )
        self.decoder = PointWiseDecoder2D(
            args.decoder_emb_dim,
            args.out_channels,
            args.out_step,
            args.propagator_depth,
            scale=args.fourier_frequency,
            dropout=0.0,
        )
    
    def forward(self, x, input_pos, output_pos, T):
        z = self.encoder(x, input_pos)
        y = self.decoder.rollout(z, output_pos, T, input_pos)
        return y
