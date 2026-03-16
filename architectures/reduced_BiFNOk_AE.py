import jax.numpy as jnp
import equinox as eqx

from jax import random

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

class BiFNOk(eqx.Module):
    encoder_a: eqx.Module
    encoder_b: eqx.Module
    decoder_a: eqx.Module
    decoder_b: eqx.Module
    convs1_a: list
    convs2_a: list
    convs1_b: list
    convs2_b: list
    A: jnp.array

    def __init__(self, N_layers, N_features_a, N_features_b, N_modes, D, kernel_size, key, s1=1.0, s2=1.0, s3=1.0):
        n_in_a, n_processor_a, n_out_a = N_features_a
        n_in_b, n_processor_b, n_out_b = N_features_b
        n_processor = n_processor_a + n_processor_b

        keys = random.split(key, 5)
        self.encoder_a = normalize_conv(eqx.nn.Conv(D, n_in_a, n_processor_a, 1, key=keys[0]), s1=s1, s2=s2)
        self.encoder_b = normalize_conv(eqx.nn.Conv(D, n_in_b, n_processor_b, 1, key=keys[1]), s1=s1, s2=s2)
        self.decoder_a = normalize_conv(eqx.nn.Conv(D, n_processor_a, n_out_a, 1, key=keys[2]), s1=s1, s2=s2)
        self.decoder_b = normalize_conv(eqx.nn.Conv(D, n_processor_b, n_out_b, 1, key=keys[3]), s1=s1, s2=s2)
        
        keys = random.split(keys[4], 2*N_layers + 1)
        self.convs1_a = [normalize_conv(eqx.nn.Conv(D, n_processor_a, n_processor_a, 1, key=key), s1=s1, s2=s2) for key in keys[:N_layers]]
        self.convs2_a = [normalize_conv(eqx.nn.Conv(D, n_processor_a, n_processor_a, 1, key=key), s1=s1, s2=s2) for key in keys[N_layers:2*N_layers]]
        
        keys = random.split(keys[-1], 2*N_layers + 1)
        self.convs1_b = [normalize_conv(eqx.nn.Conv(D, n_processor_b, n_processor_b, kernel_size, padding=kernel_size//2, key=key), s1=s1, s2=s2) for key in keys[:N_layers]]
        self.convs2_b = [normalize_conv(eqx.nn.Conv(D, n_processor_b, n_processor_b, kernel_size, padding=kernel_size//2, key=key), s1=s1, s2=s2) for key in keys[N_layers:2*N_layers]]
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
            u_b += gelu(self.convs2_b[i](gelu(self.convs1_b[i](u_b_))))
        u_a = self.decoder_a(u_a)
        u_b = self.decoder_b(u_b)
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

class coeff_encoder(eqx.Module):
    convs: list
    linear: eqx.Module

    def __init__(self, N_features_in, N_features_out, D, J_b, kernel_size, key):
        keys = random.split(key)
        convs = []
        stride = 2
        padding = kernel_size // 2
        for key in random.split(keys[0], J_b):
            convs.append(eqx.nn.Conv(D, N_features_in, N_features_in, kernel_size=kernel_size, stride=stride, padding=padding, key=key))
        self.convs = convs
        self.linear = eqx.nn.Conv(D, N_features_in, N_features_out, 1, key=keys[1])

    def __call__(self, u_b):
        for c in self.convs:
            u_b = gelu(c(u_b))
        u_b = self.linear(u_b).reshape(-1,)
        return u_b

class coeff_decoder(eqx.Module):
    convs: list
    linear: eqx.Module

    def __init__(self, N_features_in, N_features_out, D, J_b, kernel_size, key):
        keys = random.split(key)
        convs = []
        stride = 2
        padding = kernel_size // 2
        for key in random.split(keys[0], J_b):
            convs.append(eqx.nn.ConvTranspose(D, N_features_out, N_features_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1, key=key))
        self.convs = convs
        self.linear = eqx.nn.Conv(D, N_features_in, N_features_out, 1, key=keys[1])

    def __call__(self, u_b):
        shape = [-1] + [1,]*(self.convs[0].weight.ndim - 2)
        u_b = u_b.reshape(*shape)
        u_b = self.linear(u_b)
        for c in self.convs:
            u_b = gelu(c(u_b))
        return u_b

class reduced_BiFNOk_AE(eqx.Module):
    encoder: eqx.Module
    coeff_encoder: eqx.Module
    decoder: eqx.Module
    coeff_decoder: eqx.Module
    x_b: jnp.array

    def __init__(self, N_layers, N_processor, N_modes, D, J_b, N_f_b, kernel_size, n_basis, key, s1=1.0, s2=1.0, s3=1.0):
        keys = random.split(key, 4)
        self.encoder = BiFNOk(N_layers, [D + 1, N_processor, 1], [N_f_b, N_processor, N_processor], N_modes, D,  kernel_size, keys[0], s1=s1, s2=s2, s3=s3)
        self.coeff_encoder = coeff_encoder(N_processor, n_basis, D, J_b, kernel_size, keys[1])
        self.coeff_decoder = coeff_decoder(n_basis, N_processor, D, J_b, kernel_size, keys[1])
        self.decoder = BiFNOk(N_layers, [D, N_processor, 1], [N_f_b + N_processor, N_processor, 1], N_modes, D,  kernel_size, keys[2], s1=s1, s2=s2, s3=s3)
        self.x_b = random.normal(keys[2], [2,] + [N_f_b,] + [2**J_b,]*D)

    def encode(self, u_a, x_a):
        u_b = self.encoder(u_a, x_a, None, self.x_b[0])[1]
        u_b = self.coeff_encoder(u_b)
        return u_b

    def decode(self, u_b, x_a):
        u_b = self.coeff_decoder(u_b)
        u_a = self.decoder(None, x_a, u_b, self.x_b[1])[0]
        return u_a

    def __call__(self, u_a, x_a):
        # identity map
        u_b = self.encode(u_a, x_a)
        u_a = self.decode(u_b, x_a)
        return u_a