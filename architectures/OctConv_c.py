# Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution, https://arxiv.org/abs/1904.05049

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

class OctConv(eqx.Module):
    encoder_a: eqx.Module
    encoder_b: eqx.Module
    decoder_a: eqx.Module
    convs1_a: list
    convs2_a: list
    convs1_b: list
    convs2_b: list
    a_to_b: list
    b_to_a: list

    def __init__(self, N_layers, N_features_a, N_features_b, D, kernel_size, key, J_d, s1=1.0, s2=1.0, s3=1.0):
        n_in_a, n_processor_a, n_out_a = N_features_a
        n_in_b, n_processor_b = N_features_b

        keys = random.split(key, 4)
        self.encoder_a = normalize_conv(eqx.nn.Conv(D, n_in_a, n_processor_a, 1, key=keys[0]), s1=s1, s2=s2)
        self.encoder_b = normalize_conv(eqx.nn.Conv(D, n_in_b, n_processor_b, 1, key=keys[1]), s1=s1, s2=s2)
        self.decoder_a = normalize_conv(eqx.nn.Conv(D, n_processor_a, n_out_a, 1, key=keys[2]), s1=s1, s2=s2)
        
        keys = random.split(keys[3], 2*N_layers + 1)
        self.convs1_a = [normalize_conv(eqx.nn.Conv(D, n_processor_a, n_processor_a, 1, key=key), s1=s1, s2=s2) for key in keys[:N_layers]]
        self.convs2_a = [normalize_conv(eqx.nn.Conv(D, n_processor_a, n_processor_a, 1, key=key), s1=s1, s2=s2) for key in keys[N_layers:2*N_layers]]
        
        keys = random.split(keys[-1], 2*N_layers + 4)
        self.convs1_b = [normalize_conv(eqx.nn.Conv(D, n_processor_b, n_processor_b, kernel_size, padding=kernel_size//2, key=key), s1=s1, s2=s2) for key in keys[:(N_layers-1)]]
        self.convs2_b = [normalize_conv(eqx.nn.Conv(D, n_processor_b, n_processor_b, kernel_size, padding=kernel_size//2, key=key), s1=s1, s2=s2) for key in keys[(N_layers-1):(2*N_layers-2)]]

        self.b_to_a = [normalize_conv(eqx.nn.ConvTranspose(D, n_processor_b, n_processor_a, kernel_size, padding=kernel_size//2, stride=2**J_d, output_padding=2**J_d-1, key=key), s1=s1, s2=s2) for key in random.split(keys[-2], N_layers)]
        self.a_to_b = [normalize_conv(eqx.nn.Conv(D, n_processor_a, n_processor_b, kernel_size, padding=kernel_size//2, stride=2**J_d, key=key), s1=s1, s2=s2) for key in random.split(keys[-1], N_layers)]
   
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

        for i in range(len(self.a_to_b)):
            u_b_, u_a_ = self.oct_conv(u_a, u_b, self.a_to_b[i], self.b_to_a[i])
            u_a += gelu(self.convs2_a[i](gelu(self.convs1_a[i](u_a + u_a_))))
            if i != len(self.convs1_b):
                u_b += gelu(self.convs2_b[i](gelu(self.convs1_b[i](u_b + u_b_))))
        u_a = self.decoder_a(u_a)
        return u_a, u_b

    def oct_conv(self, inp_a, inp_b, a_to_b, b_to_a):
        return a_to_b(inp_a), b_to_a(inp_b)