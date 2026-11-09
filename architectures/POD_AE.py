from typing import Tuple, Sequence
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import random
from jax.nn import gelu


class ResMLPBlock(eqx.Module):
    """Residual MLP block with Pre-LayerNorm and GELU non-linearity."""
    norm: eqx.nn.LayerNorm
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, dim: int, key: jax.Array):
        keys = random.split(key, 2)
        self.norm = eqx.nn.LayerNorm(dim)
        self.linear1 = eqx.nn.Linear(dim, dim, key=keys[0])
        self.linear2 = eqx.nn.Linear(dim, dim, key=keys[1])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        res = gelu(self.linear1(self.norm(x)))
        res = self.linear2(res)
        return x + res


class POD_Encoder(eqx.Module):
    """Encoder mapping POD coefficients to bottleneck latent representation."""
    in_proj: eqx.nn.Linear
    blocks: list
    norm_out: eqx.nn.LayerNorm
    bottleneck: eqx.nn.Linear

    def __init__(
        self,
        n_basis: int,
        d_hidden: int,
        n_blocks: int,
        d_latent: int,
        key: jax.Array,
    ):
        keys = random.split(key, n_blocks + 2)
        self.in_proj = eqx.nn.Linear(n_basis, d_hidden, key=keys[0])
        self.blocks = [ResMLPBlock(d_hidden, keys[1 + i]) for i in range(n_blocks)]
        self.norm_out = eqx.nn.LayerNorm(d_hidden)
        self.bottleneck = eqx.nn.Linear(d_hidden, d_latent, key=keys[-1])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = gelu(self.in_proj(x))
        for b in self.blocks:
            x = b(x)
        x = self.norm_out(x)
        return self.bottleneck(x)


class POD_Decoder(eqx.Module):
    """Decoder mapping bottleneck latent representation to POD coefficients."""
    latent_in: eqx.nn.Linear
    blocks: list
    norm_out: eqx.nn.LayerNorm
    out_proj: eqx.nn.Linear

    def __init__(
        self,
        d_latent: int,
        d_hidden: int,
        n_blocks: int,
        n_basis: int,
        key: jax.Array,
    ):
        keys = random.split(key, n_blocks + 2)
        self.latent_in = eqx.nn.Linear(d_latent, d_hidden, key=keys[0])
        self.blocks = [ResMLPBlock(d_hidden, keys[1 + i]) for i in range(n_blocks)]
        self.norm_out = eqx.nn.LayerNorm(d_hidden)
        self.out_proj = eqx.nn.Linear(d_hidden, n_basis, key=keys[-1])

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        z = gelu(self.latent_in(z))
        for b in self.blocks:
            z = b(z)
        z = self.norm_out(z)
        return self.out_proj(z)


class POD_AE(eqx.Module):
    """POD-based Autoencoder with deep modern Residual MLP backbone.

    Parameters
    ----------
    n_basis : int
        Number of POD basis functions (input/output coefficient dimension).
    d_hidden : int
        Hidden dimension of the ResMLP blocks.
    n_blocks : int
        Number of ResMLP blocks in encoder and decoder.
    d_latent : int
        Bottleneck latent dimension.
    key : jax.Array
        JAX PRNGKey.
    """
    encoder: POD_Encoder
    decoder: POD_Decoder
    n_basis: int = eqx.field(static=True)
    d_hidden: int = eqx.field(static=True)
    n_blocks: int = eqx.field(static=True)
    d_latent: int = eqx.field(static=True)

    def __init__(
        self,
        n_basis: int,
        d_hidden: int,
        n_blocks: int,
        d_latent: int,
        key: jax.Array,
        **kwargs,
    ):
        self.n_basis = n_basis
        self.d_hidden = d_hidden
        self.n_blocks = n_blocks
        self.d_latent = d_latent

        keys = random.split(key, 2)
        self.encoder = POD_Encoder(n_basis, d_hidden, n_blocks, d_latent, keys[0])
        self.decoder = POD_Decoder(d_latent, d_hidden, n_blocks, n_basis, keys[1])

    def encode(self, c: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return self.encoder(c)

    def decode(self, z: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return self.decoder(z)

    def __call__(self, c: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        z = self.encode(c, *args, **kwargs)
        return self.decode(z, *args, **kwargs)

    def reconstruct(
        self,
        z: jnp.ndarray,
        V: jnp.ndarray,
        shape: Tuple[int, ...] = None,
    ) -> jnp.ndarray:
        """Reconstruct the physical field from latent code z using POD basis V."""
        c_pred = self.decode(z)
        u_flat = c_pred @ V
        if shape is not None:
            return u_flat.reshape(shape)
        return u_flat
