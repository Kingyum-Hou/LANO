from functools import wraps
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import abc


# Helpers
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = dict()

    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result

    return cached_fn


# Helper classes
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


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, channel, mult=4, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(channel, channel * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(channel * mult, channel)
        )

    def forward(self, x):
        return self.net(x)


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
    

def build_position_encoding(
        position_encoding_type,
        out_channel=None,
        project_pos_channel=-1,
        trainable_position_encoding_kwargs=None,
        fourier_position_encoding_kwargs=None,
        pos2fourier_position_encoding_kwargs=None,
):
    """
    Builds the position encodings.

    Args:
        - output_channel: refers to the number of channels of the position encodings.
        - project_pos_channel: if specified, will project the position encodings to this dimensions.
    """

    if position_encoding_type == "trainable":
        if not trainable_position_encoding_kwargs:
            raise ValueError("Make sure to pass trainable_position_encoding_kwargs")
        output_pos_enc = TrainablePositionEncoding(**trainable_position_encoding_kwargs)

    elif position_encoding_type == "fourier":
        # We don't use the index_dims arguent, as this is only known during the forward pass
        if not fourier_position_encoding_kwargs:
            raise ValueError("Make sure to pass fourier_position_encoding_kwargs")
        output_pos_enc = FourierPositionEncoding(**fourier_position_encoding_kwargs)

    elif position_encoding_type == "pos2fourier":
        if not pos2fourier_position_encoding_kwargs:
            raise ValueError("Make sure to pass pos2fourier_position_encoding_kwargs")
        output_pos_enc = FourierPositionEncoding(**pos2fourier_position_encoding_kwargs)

    else:
        raise ValueError(f"Unknown position encoding type: {position_encoding_type}")

    # Optionally, project the position encoding to a target dimension:
    position_projection = nn.Linear(out_channel, project_pos_channel) if project_pos_channel > 0 else nn.Identity()

    return output_pos_enc, position_projection


class AbstractPositionEncoding(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract positional encoding.
    """

    @property
    @abc.abstractclassmethod
    def num_dimensions(self) -> int:
        raise NotImplementedError

    @abc.abstractclassmethod
    def output_size(self, *args, **kwargs) -> int:
        raise NotImplementedError

    @abc.abstractclassmethod
    def forward(self, pos):
        raise NotImplementedError


class TrainablePositionEncoding(AbstractPositionEncoding):
    """ Trainable position encoding. """

    def __init__(self, index_dims, num_channel=128):
        super(TrainablePositionEncoding, self).__init__()
        self._num_channel = num_channel
        self._index_dims = index_dims
        index_dim = np.prod(index_dims)
        self.position_embeddings = nn.Parameter(torch.randn(index_dim, num_channel))

    @property
    def num_dimensions(self) -> int:
        if isinstance(self._index_dims, int):
            return 1
        return len(self._index_dims)

    def output_size(self, *args, **kwargs) -> int:
        return self._num_channel

    def forward(self, pos=None):
        position_embeddings = self.position_embeddings
        return position_embeddings


class FourierPositionEncoding(AbstractPositionEncoding):
    """ Fourier (Sinusoidal) position encoding. """

    def __init__(self, num_bands, max_resolution, concat_pos=True, sine_only=False):
        super(FourierPositionEncoding, self).__init__()
        self.num_bands = num_bands
        self.max_resolution = max_resolution
        self.concat_pos = concat_pos
        self.sine_only = sine_only

    @property
    def num_dimensions(self) -> int:
        return len(self.max_resolution)

    def output_size(self):
        """ Returns size of positional encodings last dimension. """
        encoding_size = sum(self.num_bands)
        if not self.sine_only:
            encoding_size *= 2
        if self.concat_pos:
            encoding_size += self.num_dimensions

        return encoding_size

    def forward(self, pos=None):
        fourier_pos_enc = generate_fourier_features(
            pos,
            num_bands=self.num_bands,
            max_resolution=self.max_resolution,
            concat_pos=self.concat_pos,
            sine_only=self.sine_only)
        return fourier_pos_enc


def generate_fourier_features(pos, num_bands, max_resolution=(2 ** 10), concat_pos=True, sine_only=False):
    """
    Generate a Fourier feature position encoding with linear spacing.

    Args:
        pos: The Tensor containing the position of n points in d dimensional space.
        num_bands: The number of frequency bands (K) to use.
        max_resolution: The maximum resolution (i.e., the number of pixels per dim). A tuple representing resoltuion for each dimension.
        concat_pos: Whether to concatenate the input position encoding to the Fourier features.
        sine_only: Whether to use a single phase (sin) or two (sin/cos) for each frequency band.
    """
    if sum(num_bands) == 0:
        return pos

    if len(pos.shape) > 2:
        batch_size = pos.shape[0]
    else:
        batch_size = None

    min_freq = 1.0

    # Nyquist frequency at the target resolution:
    freq_bands = torch.stack(
        [torch.linspace(
            start=min_freq, end=res / 2, steps=num_band) for res, num_band in zip(max_resolution, num_bands)],
        dim=0).to(pos.device)

    # Get frequency bands for each spatial dimension.
    # Output is size [n, d * num_bands]
    if batch_size is not None:
        per_pos_features = pos[:, :, :, None] * freq_bands[None, :, :]  # This is for elasticity
        per_pos_features = torch.reshape(per_pos_features, [batch_size, -1, np.prod(per_pos_features.shape[2:])])
    else:
        per_pos_features = pos[:, :, None] * freq_bands[None, :, :]
        per_pos_features = torch.reshape(per_pos_features, [-1, np.prod(per_pos_features.shape[1:])])

    if sine_only:
        # Output is size [n, d * num_bands]
        per_pos_features = torch.sin(np.pi * per_pos_features)
    else:
        # Output is size [n, 2 * d * num_bands]
        per_pos_features = torch.cat(
            [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1)

    # Concatenate the raw input positions.
    if concat_pos:
        # Adds d bands to the encoding.
        per_pos_features = torch.cat([pos, per_pos_features], dim=-1)
    return per_pos_features


class IPOTEncoder(nn.Module):
    def __init__(
            self,
            *,
            input_channel=None,
            cross_heads_channel=None,  # default: latent_channel // cross_heads_num
            num_latents=2 ** 10,
            latent_channel=2 ** 6,
            latent_init_scale=0.02,
            cross_heads_num=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            weight_tie_layers=False,
            use_query_residual=True):
        super(IPOTEncoder, self).__init__()

        self.input_channel = input_channel

        if cross_heads_channel is None:
            cross_heads_channel = int(latent_channel // cross_heads_num)

        self.use_query_residual = use_query_residual

        self.latents = nn.Parameter(torch.randn(num_latents, latent_channel) * latent_init_scale)

        self.encoder_cross_attn = PreNorm(
            latent_channel,
            Attention(
                latent_channel,
                input_channel,
                heads_num=cross_heads_num,
                heads_channel=cross_heads_channel,
                dropout=attn_dropout
            ),
            context_channel=input_channel
        )
        self.encoder_ff = PreNorm(
            latent_channel,
            FeedForward(
                latent_channel,
                mult=ff_mult,
                dropout=ff_dropout
            )
        )

    def forward(self, inputs, mask=None, return_embedding=False):
        b, *axis = inputs.shape

        # concat to channels of data and flatten axis
        inputs = rearrange(inputs, 'b ... d -> b (...) d')
        z = repeat(self.latents, 'n d -> b n d', b=b)

        if self.use_query_residual:
            z = self.encoder_cross_attn(z, context=inputs) + z
        else:
            z = self.encoder_cross_attn(z, context=inputs)

        return z
    

class IPOTProcessor(nn.Module):
    def __init__(
            self,
            *,
            self_per_cross_attn=6,
            self_heads_channel=None,  # default: latent_channel // self_heads_num
            latent_channel=2 ** 6,
            self_heads_num=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            weight_tie_layers=False,
            use_query_residual=True):
        super(IPOTProcessor, self).__init__()

        if self_heads_channel is None:
            self_heads_channel = int(latent_channel // self_heads_num)

        self.use_query_residual = use_query_residual

        self.processor_self_attn = PreNorm(
            latent_channel,
            Attention(
                latent_channel,
                heads_num=self_heads_num,
                heads_channel=self_heads_channel,
                dropout=attn_dropout
            )
        )
        self.processor_ff = PreNorm(
            latent_channel,
            FeedForward(
                latent_channel,
                mult=ff_mult,
                dropout=ff_dropout
            )
        )

        self.layers = nn.ModuleList([])

        for i in range(self_per_cross_attn):
            self.layers.append(nn.ModuleList([
                self.processor_self_attn,
                self.processor_ff
            ]))

    def forward(self, z, mask=None, return_embedding=False):
        b, *axis = z.shape

        # Processing layers
        for self_attn, self_ff in self.layers:
            if self.use_query_residual:
                z = self_attn(z, context=z) + z
            else:
                z = self_attn(z, context=z)
            z = self_ff(z) + z
        return z
    

class IPOTDecoder(nn.Module):
    def __init__(
            self,
            output_channel,
            query_channel,
            latent_channel=2 ** 8,
            output_scale=0.1,
            cross_heads_num=8,
            cross_heads_channel=None,  # default: latent_channel // cross_heads_num
            ff_mult=4,
            position_encoding_type: str = "pos2fourier",
            use_qeury_residual=False,
            concat_preprocessed_input=False,
            project_pos_channel: int = -1,
            position_encoding_only=False,
            **position_encoding_kwargs):
        super(IPOTDecoder, self).__init__()
        if cross_heads_channel is None:
            cross_heads_channel = int(latent_channel // cross_heads_num)
        self.query_channel = query_channel
        self.output_channel = output_channel
        self.output_scale = output_scale
        self.position_encoding_type = position_encoding_type
        self.use_query_residual = use_qeury_residual
        self.concat_preprocessed_input = concat_preprocessed_input
        self.position_encoding_kwargs = position_encoding_kwargs

        # If position_embedding_type is 'None', the decoder will not construct
        # any position embeddings. In that casse, you should construct your own decoder_query.

        # Position embeddings
        self.project_pos_dim = project_pos_channel
        if position_encoding_type != "none":
            self.position_embeddings, self.position_projection = build_position_encoding(
                position_encoding_type=position_encoding_type,
                out_channel=query_channel,
                project_pos_channel=project_pos_channel,
                **position_encoding_kwargs,
            )

        self.decoder_cross_attn = PreNorm(
            query_channel,
            Attention(
                query_channel,
                latent_channel,
                output_channel,
                heads_num=cross_heads_num,
                heads_channel=cross_heads_channel
            ),
            context_channel=latent_channel
        )

        self.decoder_ff = PreNorm(
            output_channel,
            FeedForward(
                output_channel,
                mult=ff_mult,
            )
        )

    @property
    def num_channels(self) -> int:
        # position embedding
        if self.project_pos_dim > 0:
            pos_channel = self.project_pos_dim
        else:
            pos_channel = self.position_embeddings.output_size()
        return pos_channel

    def _build_decoder_query(self, pos_query, network_input_is_1d: bool = True):
        """
        Construct the final input, including position encoding.

        This method expects the inputs to always have channels as last dimension.
        """
        index_dims = pos_query.shape[:-1]
        indices = np.prod(index_dims)

        # Flatten input features to a 1D index dimension if necessary.
        if len(pos_query.shape) > 2 and network_input_is_1d:
            pos_query = torch.reshape(pos_query, [indices, -1])

        if self.position_encoding_type == "trainable":
            pos_enc_query = self.position_embeddings()
        elif self.position_encoding_type == "fourier":
            pos_enc_query = self.position_embeddings(index_dims)
        elif self.position_encoding_type == "pos2fourier":
            pos_enc_query = self.position_embeddings(pos=pos_query)
        else:
            pos_enc_query = pos_query

        # Optionally project them to a target dimension.
        pos_enc_query = self.position_projection(pos_enc_query)

        return pos_enc_query, pos_query

    def forward(self, dec_query, z):
        b, *axis = z.shape

        if not exists(dec_query):
            return z

        # Build decoder query with positional embeddings
        if self.position_encoding_type != "none":
            dec_query, _ = self._build_decoder_query(dec_query)

        # Make sure query contains batch dimension
        if dec_query.ndim == 2:
            dec_query = repeat(dec_query, 'n d -> b n d', b=b)

        if dec_query.ndim > 3:
            dec_query = dec_query.reshape(dec_query.shape[0], -1, dec_query.shape[-1])

        # Make sure that queries and latents are on the same device.
        if dec_query.device != z.device:
            dec_query = dec_query.to(z.device)

        # Cross attend from decoder queries to latents.
        if self.use_query_residual:
            out = self.decoder_cross_attn(dec_query, context=z) + dec_query
        else:
            out = self.decoder_cross_attn(dec_query, context=z)

        # Optional decoder feedforward
        if exists(self.decoder_ff):
            out = self.decoder_ff(out) + out

        return out * self.output_scale
    

class AbstractPreprocessor(nn.Module):
    @property
    def num_channels(self) -> int:
        """ Returns size of preprocessor output. """
        raise NotImplementedError()


class IPOTBasicPreprocessor(AbstractPreprocessor):
    """
    Preprocessing inputs for Encoder.
    """

    def __init__(
            self,
            config=None,
            prep_type="pixels",
            spatial_downsample: int = 1,
            temporal_downsample: int = 1,
            position_encoding_type: str = "pos2fourier",
            in_channel: int = 1,
            pos_channel: int = 1,
            out_channel: int = None,
            concat_or_add_pos: str = "concat",
            project_pos_channel: int = -1,
            **position_encoding_kwargs,
    ):
        super(IPOTBasicPreprocessor, self).__init__()
        self.config = config

        if concat_or_add_pos not in ["concat", "add"]:
            raise ValueError(f"Invalid value {concat_or_add_pos} for concat_or_add_pos.")

        self.in_channel = in_channel
        self.pos_channel = pos_channel
        self.out_channel = out_channel
        self.prep_type = prep_type
        self.spatial_downsample = spatial_downsample
        self.temporal_downsample = temporal_downsample
        self.position_encoding_type = position_encoding_type
        self.concat_or_add_pos = concat_or_add_pos

        # Position embeddings
        self.project_pos_channel = project_pos_channel
        if position_encoding_type != "none":
            self.position_embeddings, self.position_projection = build_position_encoding(
                position_encoding_type=position_encoding_type,
                out_channel=out_channel,
                project_pos_channel=project_pos_channel,
                **position_encoding_kwargs,
            )

    @property
    def num_channels(self) -> int:
        # position embedding
        if self.project_pos_channel > 0:
            pos_channel = self.project_pos_channel
        else:
            pos_channel = self.position_embeddings.output_size()

        if self.concat_or_add_pos == "add":
            return pos_channel

        # inputs
        if self.prep_type == "pixels":
            input_channel = self.in_channel
        else:
            raise NotImplementedError("Not supported yet.")

        return input_channel + pos_channel

    def _build_network_inputs(self, inputs: torch.Tensor, network_input_is_1d: bool = True):
        """
        Construct the final input, including position encoding.

        This method expects the inputs to always have channels as last dimension.
        """
        self.batch_size = inputs.shape[0]
        index_dims = inputs.shape[1:-1]
        indices = np.prod(index_dims)

        # Flatten input features to a 1D index dimension if necessary.
        if len(inputs.shape) > 3 and network_input_is_1d:
            inputs = torch.reshape(inputs, [self.batch_size, indices, -1])

        inputs_function = inputs[..., :self.in_channel]
        inputs_pos      = inputs[..., self.in_channel:]

        # Construct the position encoding.
        if self.position_encoding_type == "trainable":
            inputs_pos_enc = self.position_embeddings(self.batch_size)
        elif self.position_encoding_type == "fourier":
            inputs_pos_enc = self.position_embeddings(index_dims)
        elif self.position_encoding_type == "pos2fourier":
            inputs_pos_enc = self.position_embeddings(pos=inputs_pos)
        else:
            raise NotImplementedError

        # Optionally project them to a target dimension.
        inputs_pos_enc = self.position_projection(inputs_pos_enc)

        return inputs_function, inputs_pos_enc

    def forward(self, inputs: torch.Tensor, network_input_is_1d: bool = True):
        # Split function values and positions, and build positional encoding
        if self.position_encoding_type != "none":
            inputs_function, inputs_pos_enc = self._build_network_inputs(inputs, network_input_is_1d)

        # Make sure inputs_pos_enc contains batch dimensions.
        if inputs_pos_enc.ndim == 2:
            inputs_pos_enc = repeat(inputs_pos_enc, 'n d -> b n d', b=self.batch_size)

        # Make sure that inputs_pos_enc and inputs_function are on the same device.
        if inputs_pos_enc.device != inputs_function.device:
            inputs_pos_enc = inputs_pos_enc.to(inputs_function.device)

        # Concat or add?
        if self.concat_or_add_pos == "concat":
            inputs_for_encoder = torch.cat([inputs_function, inputs_pos_enc], dim=-1)
        elif self.concat_or_add_pos == "add":
            inputs_for_encoder = inputs_function + inputs_pos_enc
        else:
            raise NotImplementedError

        return inputs_for_encoder
    

class EncoderProcessorDecoder(nn.Module):
    def __init__(
            self,
            encoder,
            processor,
            decoder,
            input_preprocessor=None,
            output_postprocessor=None,
    ):
        super(EncoderProcessorDecoder, self).__init__()
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
        self.input_preprocessor = input_preprocessor
        self.output_postprocessor = output_postprocessor

    def forward(self, inputs, decoder_query=None, T_out=None):
        trajectory = []
        # Operating input_preprocessor.
        if self.input_preprocessor:
            inputs_for_encoder = self.input_preprocessor(inputs, network_input_is_1d=False)
        else:
            inputs_for_encoder = inputs

        # Operating encoder.
        z = self.encoder(inputs_for_encoder)

        if T_out is not None:
            for _ in range(T_out):
                # Operating processor at each time steps
                z = self.processor(z)

                # Operating decoder at each time steps
                output = self.decoder(decoder_query, z)
                trajectory.append(output)

            trajectory = torch.cat(trajectory, dim=-1)

            return trajectory

        else:
            # Operator processor.
            z = self.processor(z)

            # Operating decoder.
            if decoder_query is not None:
                output = self.decoder(decoder_query, z)
            else:
                output = self.decoder(z)

            return output
