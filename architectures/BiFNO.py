import jax.numpy as jnp
import equinox as eqx

from jax.lax import dot_general
from jax import random, vmap
from jax.nn import gelu

def normalize_conv(A, s1=1.0, s2=1.0):
    A = eqx.tree_at(lambda x: x.weight, A, A.weight * s1)
    try:
        A = eqx.tree_at(lambda x: x.bias, A, A.bias * s2)
    except:
        pass
    return A

def rfft2_truncate(inp, M):
    inp = jnp.fft.rfft2(inp, norm='forward')[:, :, :M]
    inp = jnp.roll(inp, M//2, axis=1)[:, :M]
    return inp

def irfft2_pad(inp, M, s):
    inp = jnp.pad(inp, [(0, 0),] +[(0, s_-inp.shape[i+1]) for i, s_ in enumerate(s)])
    inp = jnp.roll(inp, -M//2, axis=1)
    inp = jnp.fft.irfft2(inp, s=s, norm='forward')
    return inp

def rfft3_truncate(inp, M):
    inp = jnp.fft.rfftn(inp, axes=(1, 2, 3), norm='forward')[:, :, :, :M]
    inp = jnp.roll(inp, M//2, axis=1)[:, :M]
    inp = jnp.roll(inp, M//2, axis=2)[:, :, :M]
    return inp

def irfft3_pad(inp, M, s):
    inp = jnp.pad(inp, [(0, 0),] +[(0, s_-inp.shape[i+1]) for i, s_ in enumerate(s)])
    inp = jnp.roll(inp, -M//2, axis=1)
    inp = jnp.roll(inp, -M//2, axis=2)
    inp = jnp.fft.irfftn(inp, axes=(1, 2, 3), s=s, norm='forward')
    return inp

class BiFNO(eqx.Module):
    encoder_a: eqx.Module
    encoder_b: eqx.Module
    decoder_a: eqx.Module
    convs1_a: list
    convs2_a: list
    convs1_b: list
    convs2_b: list
    A: jnp.array

    def __init__(self, N_layers, N_features_a, N_features_b, N_modes, D, key, s1=1.0, s2=1.0, s3=1.0):
        n_in_a, n_processor_a, n_out_a = N_features_a
        n_in_b, n_processor_b = N_features_b
        n_processor = n_processor_a + n_processor_b

        keys = random.split(key, 4)
        self.encoder_a = normalize_conv(eqx.nn.Conv(D, n_in_a, n_processor_a, 1, key=keys[0]), s1=s1, s2=s2)
        self.encoder_b = normalize_conv(eqx.nn.Conv(D, n_in_b, n_processor_b, 1, key=keys[1]), s1=s1, s2=s2)
        self.decoder_a = normalize_conv(eqx.nn.Conv(D, n_processor_a, n_out_a, 1, key=keys[2]), s1=s1, s2=s2)
        
        keys = random.split(keys[3], 2*N_layers + 1)
        self.convs1_a = [normalize_conv(eqx.nn.Conv(D, n_processor_a, n_processor_a, 1, key=key), s1=s1, s2=s2) for key in keys[:N_layers]]
        self.convs2_a = [normalize_conv(eqx.nn.Conv(D, n_processor_a, n_processor_a, 1, key=key), s1=s1, s2=s2) for key in keys[N_layers:2*N_layers]]
        
        keys = random.split(keys[-1], 2*N_layers - 1)
        self.convs1_b = [normalize_conv(eqx.nn.Conv(D, n_processor_b, n_processor_b, 1, key=key), s1=s1, s2=s2) for key in keys[:(N_layers-1)]]
        self.convs2_b = [normalize_conv(eqx.nn.Conv(D, n_processor_b, n_processor_b, 1, key=key), s1=s1, s2=s2) for key in keys[(N_layers-1):(2*N_layers-2)]]
        self.A = random.normal(keys[-1], [N_layers, n_processor, n_processor] + [N_modes,]*D, dtype=jnp.complex64) * s3

    def __call__(self, u_a, x_a, u_b, x_b):
        if u_a is None:
            u_a = x_a
        else:
            u_a = jnp.concatenate([x_a, u_a], 0)
        u_a = self.encoder_a(u_a)

        if u_b is None:
            u_b = x_b
        else:
            u_b = jnp.concatenate([x_b, u_b], 0)
        u_b = self.encoder_b(u_b)

        for i in range(self.A.shape[0]):
            u_a_, u_b_ = self.biscale_spectral_conv(u_a, u_b, self.A[i])
            u_a += gelu(self.convs2_a[i](gelu(self.convs1_a[i](u_a_))))
            if i != len(self.convs1_b):
                u_b += gelu(self.convs2_b[i](gelu(self.convs1_b[i](u_b_))))
        u_a = self.decoder_a(u_a)
        return u_a, u_b

    def biscale_spectral_conv(self, inp_1, inp_2, W):
        D = inp_1.ndim - 1
        if D == 1:
            u = self.biscale_spectral_conv_d1(inp_1, inp_2, W)
        elif D == 2:
            u = self.biscale_spectral_conv_d2(inp_1, inp_2, W)
        else:
            u = self.biscale_spectral_conv_d3(inp_1, inp_2, W)
        return u
        
    def biscale_spectral_conv_d1(self, inp_1, inp_2, W):
        coeff = jnp.concatenate([
            jnp.fft.rfft(inp_1, norm='forward', axis=1)[:, :W.shape[2]],
            jnp.fft.rfft(inp_2, norm='forward', axis=1)[:, :W.shape[2]]
        ], axis=0)
        coeff = dot_general(W, coeff, (((1,), (0,)), ((2,), (1,)))).T
        out_1 = jnp.fft.irfft(coeff[:inp_1.shape[0]], norm='forward', n=inp_1.shape[1], axis=1)
        out_2 = jnp.fft.irfft(coeff[inp_1.shape[0]:], norm='forward', n=inp_2.shape[1], axis=1)
        return out_1, out_2

    def biscale_spectral_conv_d2(self, inp_1, inp_2, W):
        coeff = jnp.concatenate([
            rfft2_truncate(inp_1, W.shape[2]),
            rfft2_truncate(inp_2, W.shape[2])
        ], axis=0)
        coeff = jnp.moveaxis(dot_general(W, coeff, (((1,), (0,)), ((2, 3), (1, 2)))), 2, 0)
        out_1 = irfft2_pad(coeff[:inp_1.shape[0]], W.shape[2], inp_1.shape[1:])
        out_2 = irfft2_pad(coeff[inp_1.shape[0]:], W.shape[2], inp_2.shape[1:])
        return out_1, out_2

    def biscale_spectral_conv_d3(self, inp_1, inp_2, W):
        coeff = jnp.concatenate([
            rfft3_truncate(inp_1, W.shape[-1]),
            rfft3_truncate(inp_2, W.shape[-1])
        ], axis=0)
        coeff = jnp.moveaxis(dot_general(W, coeff, (((1,), (0,)), ((2, 3, 4), (1, 2, 3)))), 3, 0)
        out_1 = irfft3_pad(coeff[:inp_1.shape[0]], W.shape[-1], inp_1.shape[1:])
        out_2 = irfft3_pad(coeff[inp_1.shape[0]:], W.shape[-1], inp_2.shape[1:])
        return out_1, out_2