from typing import Union, Tuple, Sequence
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import random
from jax.nn import gelu


def normalize_conv(A, s1=1.0, s2=1.0):
    """Normalize weights and biases of a convolutional layer."""
    A = eqx.tree_at(lambda x: x.weight, A, A.weight * s1)
    try:
        A = eqx.tree_at(lambda x: x.bias, A, A.bias * s2)
    except Exception:
        pass
    return A


def _valid_num_groups(channels: int, desired_groups: int = 4) -> int:
    """Find the closest divisor of channels <= desired_groups."""
    if channels % desired_groups == 0:
        return desired_groups
    for g in range(min(desired_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


def compute_layer_strides(subsampling: Union[int, Sequence[int]], N_blocks: int = 4, D: int = 2):
    """Compute per-block strides for encoder and decoder given total subsampling.

    The initial projection layer performs a 2x downsample along dimensions with
    subsampling >= 2. The remaining downsampling factors are distributed across
    the N_blocks residual layers.
    """
    if isinstance(subsampling, int):
        subsampling = (subsampling,) * D
    else:
        subsampling = tuple(subsampling)

    init_strides = tuple(min(2, s) for s in subsampling)
    rem_subs = tuple(s // init_s if s >= init_s else 1 for s, init_s in zip(subsampling, init_strides))
    num_downs = tuple(int(round(np.log2(max(1, r)))) for r in rem_subs)

    enc_strides = []
    for i in range(N_blocks):
        st = tuple(2 if i < n else 1 for n in num_downs)
        if len(set(st)) == 1:
            st = st[0]
        enc_strides.append(st)

    dec_strides = list(reversed(enc_strides))
    init_st = init_strides[0] if len(set(init_strides)) == 1 else init_strides
    return init_st, enc_strides, dec_strides


class ResBlockDown(eqx.Module):
    """Residual block with strided conv1 for downsampling and skip connection."""
    conv1: eqx.nn.Conv
    norm1: eqx.nn.GroupNorm
    conv2: eqx.nn.Conv
    norm2: eqx.nn.GroupNorm
    skip: eqx.Module
    has_stride: bool = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]],
        D: int,
        num_groups: int,
        key: jax.Array,
        s1: float = 1.0,
        s2: float = 1.0,
    ):
        keys = random.split(key, 3)
        self.has_stride = stride != 1 if isinstance(stride, int) else any(s != 1 for s in stride)
        groups = _valid_num_groups(channels, num_groups)

        self.conv1 = normalize_conv(
            eqx.nn.Conv(
                num_spatial_dims=D,
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                padding="same",
                key=keys[0],
            ),
            s1=s1,
            s2=s2,
        )
        self.norm1 = eqx.nn.GroupNorm(groups=groups, channels=channels)
        self.conv2 = normalize_conv(
            eqx.nn.Conv(
                num_spatial_dims=D,
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                key=keys[1],
            ),
            s1=s1,
            s2=s2,
        )
        self.norm2 = eqx.nn.GroupNorm(groups=groups, channels=channels)

        if self.has_stride:
            self.skip = normalize_conv(
                eqx.nn.Conv(
                    num_spatial_dims=D,
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=1,
                    stride=stride,
                    padding="same",
                    key=keys[2],
                ),
                s1=s1,
                s2=s2,
            )
        else:
            self.skip = eqx.nn.Identity()

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        res = gelu(self.norm1(self.conv1(x)))
        res = self.norm2(self.conv2(res))
        return self.skip(x) + res


class ResBlockUp(eqx.Module):
    """Residual block with transposed conv1 for upsampling and skip connection."""
    conv1: eqx.Module
    norm1: eqx.nn.GroupNorm
    conv2: eqx.nn.Conv
    norm2: eqx.nn.GroupNorm
    skip: eqx.Module
    has_stride: bool = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]],
        D: int,
        num_groups: int,
        key: jax.Array,
        s1: float = 1.0,
        s2: float = 1.0,
    ):
        keys = random.split(key, 3)
        self.has_stride = stride != 1 if isinstance(stride, int) else any(s != 1 for s in stride)
        groups = _valid_num_groups(channels, num_groups)

        if self.has_stride:
            self.conv1 = normalize_conv(
                eqx.nn.ConvTranspose(
                    num_spatial_dims=D,
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding="same",
                    key=keys[0],
                ),
                s1=s1,
                s2=s2,
            )
            self.skip = normalize_conv(
                eqx.nn.ConvTranspose(
                    num_spatial_dims=D,
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=1,
                    stride=stride,
                    padding="same",
                    key=keys[2],
                ),
                s1=s1,
                s2=s2,
            )
        else:
            self.conv1 = normalize_conv(
                eqx.nn.Conv(
                    num_spatial_dims=D,
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding="same",
                    key=keys[0],
                ),
                s1=s1,
                s2=s2,
            )
            self.skip = eqx.nn.Identity()

        self.norm1 = eqx.nn.GroupNorm(groups=groups, channels=channels)
        self.conv2 = normalize_conv(
            eqx.nn.Conv(
                num_spatial_dims=D,
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                key=keys[1],
            ),
            s1=s1,
            s2=s2,
        )
        self.norm2 = eqx.nn.GroupNorm(groups=groups, channels=channels)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        res = gelu(self.norm1(self.conv1(x)))
        res = self.norm2(self.conv2(res))
        return self.skip(x) + res


class Conv_AE(eqx.Module):
    """Progressive Convolutional Autoencoder with early downsampling and tailored layer strides.

    Parameters
    ----------
    N_blocks : int
        Number of residual blocks in encoder and decoder (default: 4).
    N_features : Sequence[int]
        [in_channels, n_processor, out_channels].
    kernel_size : int or Sequence[int]
        Convolution kernel size (default: 3).
    subsampling : int or Sequence[int]
        Total subsampling factor along each spatial dimension (e.g. 4, 8, 16, 32).
    n_codes : int
        Number of channels in latent code.
    key : jax.Array
        JAX PRNG key.
    D : int, default 2
        Spatial dimensionality.
    num_groups : int, default 4
        Number of groups for GroupNorm layers.
    s1 : float, default 1.0
        Normalization factor for convolution weights.
    s2 : float, default 1.0
        Normalization factor for convolution biases.
    """
    in_proj: eqx.nn.Conv
    enc_blocks: list
    enc_out: eqx.nn.Conv
    dec_in: eqx.nn.Conv
    dec_blocks: list
    out_proj: eqx.nn.ConvTranspose
    D: int = eqx.field(static=True)

    def __init__(
        self,
        N_blocks: int,
        N_features: Sequence[int],
        kernel_size: Union[int, Sequence[int]],
        subsampling: Union[int, Sequence[int]],
        n_codes: int,
        key: jax.Array,
        D: int = 2,
        num_groups: int = 4,
        s1: float = 1.0,
        s2: float = 1.0,
        **kwargs,
    ):
        if "D" in kwargs:
            D = kwargs["D"]
        if "stride" in kwargs and subsampling is None:
            subsampling = kwargs["stride"]
        if "n_code" in kwargs:
            n_codes = kwargs["n_code"]

        self.D = D
        in_channels, n_processor, out_channels = N_features
        init_st, enc_strides, dec_strides = compute_layer_strides(subsampling, N_blocks, D)
        keys = random.split(key, 2 * N_blocks + 4)

        # in_proj: Conv with kernel_size and init_st (e.g. stride=2 downsamples right away)
        self.in_proj = normalize_conv(
            eqx.nn.Conv(
                num_spatial_dims=D,
                in_channels=in_channels,
                out_channels=n_processor,
                kernel_size=kernel_size,
                stride=init_st,
                padding="same",
                key=keys[0],
            ),
            s1=s1,
            s2=s2,
        )

        self.enc_blocks = [
            ResBlockDown(
                channels=n_processor,
                kernel_size=kernel_size,
                stride=enc_strides[i],
                D=D,
                num_groups=num_groups,
                key=keys[1 + i],
                s1=s1,
                s2=s2,
            )
            for i in range(N_blocks)
        ]

        self.enc_out = normalize_conv(
            eqx.nn.Conv(
                num_spatial_dims=D,
                in_channels=n_processor,
                out_channels=n_codes,
                kernel_size=1,
                padding="same",
                key=keys[N_blocks + 1],
            ),
            s1=s1,
            s2=s2,
        )

        self.dec_in = normalize_conv(
            eqx.nn.Conv(
                num_spatial_dims=D,
                in_channels=n_codes,
                out_channels=n_processor,
                kernel_size=1,
                padding="same",
                key=keys[N_blocks + 2],
            ),
            s1=s1,
            s2=s2,
        )

        self.dec_blocks = [
            ResBlockUp(
                channels=n_processor,
                kernel_size=kernel_size,
                stride=dec_strides[i],
                D=D,
                num_groups=num_groups,
                key=keys[N_blocks + 3 + i],
                s1=s1,
                s2=s2,
            )
            for i in range(N_blocks)
        ]

        # out_proj: ConvTranspose with kernel_size and init_st (upsamples back to full resolution)
        self.out_proj = normalize_conv(
            eqx.nn.ConvTranspose(
                num_spatial_dims=D,
                in_channels=n_processor,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=init_st,
                padding="same",
                key=keys[-1],
            ),
            s1=s1,
            s2=s2,
        )

    def encode(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        x = self.in_proj(x)
        for b in self.enc_blocks:
            x = b(x)
        return self.enc_out(x)

    def decode(self, z: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        x = self.dec_in(z)
        for b in self.dec_blocks:
            x = b(x)
        return self.out_proj(x)

    def __call__(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return self.decode(self.encode(x, *args, **kwargs), *args, **kwargs)


# Backward-compatible aliases
ResBlock = ResBlockDown
ResBlock2D = ResBlockDown
Encoder = None
Decoder = None
