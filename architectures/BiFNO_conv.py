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

class FNO(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    convs1: list
    convs2: list
    A: jnp.array

    def __init__(self, N_layers, N_features, N_modes, D, key, s1=1.0, s2=1.0, s3=1.0):
        n_in, n_processor, n_out = N_features
        keys = random.split(key, 3 + 2*N_layers)
        self.encoder = normalize_conv(eqx.nn.Conv(D, n_in, n_processor, 1, key=keys[-1]), s1=s1, s2=s2)
        self.decoder = normalize_conv(eqx.nn.Conv(D, n_processor, n_out, 1, key=keys[-2]), s1=s1, s2=s2)
        self.convs1 = [normalize_conv(eqx.nn.Conv(D, n_processor, n_processor, 1, key=key), s1=s1, s2=s2) for key in keys[:N_layers]]
        self.convs2 = [normalize_conv(eqx.nn.Conv(D, n_processor, n_processor, 1, key=key), s1=s1, s2=s2) for key in keys[N_layers:2*N_layers]]
        self.A = random.normal(keys[-3], [N_layers, n_processor, n_processor] + [N_modes,]*D, dtype=jnp.complex64) * s3

    def __call__(self, u, x):
        u = jnp.concatenate([x, u], 0)
        u = self.encoder(u)
        for i in range(self.A.shape[0]):
            u += gelu(self.convs2[i](gelu(self.convs1[i](self.spectral_conv(u, self.A[i])))))
        u = self.decoder(u)
        return u

    def spectral_conv(self, v, A):
        D = v.ndim - 1
        if D == 1:
            u = self.spectral_conv_d1(v, A)
        elif D == 2:
            u = self.spectral_conv_d2(v, A)
        else:
            u = self.spectral_conv_d3(v, A)
        return u

    def spectral_conv_d1(self, inp, A):
        coeff = jnp.fft.rfft(inp, norm='forward', axis=1)[:, :A.shape[2]]
        coeff = dot_general(A, coeff, (((1,), (0,)), ((2,), (1,)))).T
        out = jnp.fft.irfft(coeff, norm='forward', n=inp.shape[1], axis=1)
        return out
    
    def spectral_conv_d2(self, inp, A):
        coeff = rfft2_truncate(inp, A.shape[2])
        coeff = jnp.moveaxis(dot_general(A, coeff, (((1,), (0,)), ((2, 3), (1, 2)))), 2, 0)
        out = irfft2_pad(coeff, A.shape[2], inp.shape[1:])
        return out

    def spectral_conv_d3(self, inp, A):
        coeff = rfft3_truncate(inp, A.shape[-1])
        coeff = jnp.moveaxis(dot_general(A, coeff, (((1,), (0,)), ((2, 3, 4), (1, 2, 3)))), 3, 0)
        out = irfft3_pad(coeff, A.shape[-1], inp.shape[1:])
        return out

class BiFNO_conv(eqx.Module):
    FNO_a: eqx.Module
    FNO_b: eqx.Module
    b_to_a: list
    a_to_b: list

    def __init__(self, N_layers, N_features_a, N_features_b, N_modes, D, kernel_size, key, J_a, J_b, s1=1.0, s2=1.0, s3=1.0):
        n_in_a, n_processor_a, n_out_a = N_features_a
        n_in_b, n_processor_b = N_features_b
        keys = random.split(key, 6)
        self.FNO_a = FNO(N_layers, N_features_a, N_modes, D, keys[0], s1=s1, s2=s2, s3=s3)
        self.FNO_b = FNO(N_layers, N_features_b + [1,], N_modes, D, keys[1], s1=s1, s2=s2, s3=s3)
        J_d = abs(J_a - J_b)
        if J_a > J_b:
            self.b_to_a = [normalize_conv(eqx.nn.ConvTranspose(D, n_processor_b, n_processor_a, kernel_size, padding=kernel_size//2, stride=2**J_d, output_padding=2**J_d-1, key=key), s1=s1, s2=s2) for key in random.split(keys[-2], N_layers)]
            self.a_to_b = [normalize_conv(eqx.nn.Conv(D, n_processor_a, n_processor_b, kernel_size, padding=kernel_size//2, stride=2**J_d, key=key), s1=s1, s2=s2) for key in random.split(keys[-1], N_layers)]
        else:
            self.b_to_a = [normalize_conv(eqx.nn.Conv(D, n_processor_a, n_processor_b, kernel_size, padding=kernel_size//2, stride=2**J_d, key=key), s1=s1, s2=s2) for key in random.split(keys[-1], N_layers)]
            self.a_to_b = [normalize_conv(eqx.nn.ConvTranspose(D, n_processor_b, n_processor_a, kernel_size, padding=kernel_size//2, stride=2**J_d, output_padding=2**J_d-1, key=key), s1=s1, s2=s2) for key in random.split(keys[-2], N_layers)]
            
    def __call__(self, u_a, x_a, u_b, x_b):
        if u_a is None:
            u_a = x_a
        else:
            u_a = jnp.concatenate([x_a, u_a], 0)
        u_a = self.FNO_a.encoder(u_a)

        if u_b is None:
            u_b = x_b
        else:
            u_b = jnp.concatenate([x_b, u_b], 0)
        u_b = self.FNO_b.encoder(u_b)

        for i in range(self.FNO_a.A.shape[0]):
            u_a_, u_b_ = self.b_to_a[i](u_b), self.a_to_b[i](u_a)
            u_a += gelu(self.FNO_a.convs2[i](gelu(self.FNO_a.convs1[i](self.FNO_a.spectral_conv(u_a + u_a_, self.FNO_a.A[i])))))
            if i != (self.FNO_a.A.shape[0] - 1):
                u_b += gelu(self.FNO_b.convs2[i](gelu(self.FNO_b.convs1[i](self.FNO_b.spectral_conv(u_b + u_b_, self.FNO_b.A[i])))))
        u_a = self.FNO_a.decoder(u_a)
        return u_a, u_b