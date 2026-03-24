import os
import time
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import hashlib
import argparse

from jax import random, vmap
from jax.lax import scan, dot_general, dynamic_slice_in_dim
from jax.tree_util import tree_map, tree_flatten
from jax.nn import gelu

class conv_trunk_1D(eqx.Module):
    convs: list
    projector: list
    encoder: list

    def __init__(self, N_x, N_features, N_layers, kernel_size, key):
        N_in, N_encoder, N_out = N_features
        N_layers = min(jnp.log2(N_x).astype(int).item(), N_layers)
        keys = random.split(key, (N_layers, 2))
        self.convs = []
        N = N_encoder
        x = jnp.zeros((N_encoder, N_x))
        for key in keys:
            c = [
                eqx.nn.Conv(1, N, 2*N, kernel_size = kernel_size, padding = 'SAME', key = key[0]),
                eqx.nn.Conv(1, 2*N, 2*N, kernel_size = kernel_size, stride=2, key = key[1])
            ]
            x = c[1](c[0](x))
            N = 2*N
            self.convs.append(c)
        keys = random.split(keys[-1, -1])
        self.projector = [random.normal(keys[0], (N_out, x.shape[1]*N)) / jnp.sqrt(N_out + x.shape[1]*N//2), jnp.zeros((N_out,))]
        self.encoder = [jnp.zeros((N_encoder, 1)), random.normal(keys[1], (N_encoder, N_in)) / jnp.sqrt(N_encoder + N_in)]

    def __call__(self, u, coords):
        # u.shape = (N_features, N_x); coords.shape = (1, N_x)
        x = jnp.concatenate([u, coords], axis=0)
        x = self.encoder[0] + dot_general(self.encoder[1], x, (((1,), (0,)), ((), ()))) 
        for conv in self.convs:
            x = gelu(conv[1](gelu(conv[0](x))))
        x = self.projector[0] @ x.reshape(-1,) + self.projector[1]
        return x

class conv_trunk_2D(eqx.Module):
    convs: list
    projector: list
    encoder: list

    def __init__(self, N_x, N_features, N_layers, kernel_size, key):
        N_in, N_encoder, N_out = N_features
        N_layers = min(jnp.log2(N_x).astype(int).item(), N_layers)
        keys = random.split(key, (N_layers, 2))
        self.convs = []
        N = N_encoder
        x = jnp.zeros((N_encoder, N_x, N_x))
        for key in keys:
            c = [
                eqx.nn.Conv(2, N, 2*N, kernel_size = kernel_size, padding = 'SAME', key = key[0]),
                eqx.nn.Conv(2, 2*N, 2*N, kernel_size = kernel_size, stride=2, key = key[1])
            ]
            x = c[1](c[0](x))
            N = 2*N
            self.convs.append(c)
        keys = random.split(keys[-1, -1])
        self.projector = [random.normal(keys[0], (N_out, x.size)) / jnp.sqrt(N_out + x.shape[1]*N//2), jnp.zeros((N_out,))]
        self.encoder = [jnp.zeros((N_encoder, 1, 1)), random.normal(keys[1], (N_encoder, N_in)) / jnp.sqrt(N_encoder + N_in)]

    def __call__(self, u, coords):
        # u.shape = (N_features, N_x, N_x); coords.shape = (2, N_x, N_x)
        x = jnp.concatenate([u, coords], axis=0)
        x = self.encoder[0] + dot_general(self.encoder[1], x, (((1,), (0,)), ((), ())))
        for conv in self.convs:
            x = gelu(conv[1](gelu(conv[0](x))))
        x = self.projector[0] @ x.reshape(-1,) + self.projector[1]
        return x

class MLP(eqx.Module):
    weights: list
    biases: list

    def __init__(self, N_features, N_layers, key):
        N_in, N_processor, N_out = N_features
        keys = random.split(key, N_layers+1)
        Ns = [N_in,] + [N_processor,]*N_layers + [N_out,]
        self.biases = [jnp.zeros((No, 1)) for No in Ns[1:]]
        self.weights = [random.normal(key, (No, Ni)) / jnp.sqrt(No + Ni) for Ni, No, key in zip(Ns[:-1], Ns[1:], keys)]

    def __call__(self, coords):
        # coords.shape = (N_in, N_x, N_y, ...)
        c = coords.reshape(coords.shape[0], -1)
        for w, b in zip(self.weights, self.biases):
            c = gelu(w @ c + b)
        c = c.reshape([c.shape[0],] + list(coords.shape[1:]))
        return c

class DeepONet(eqx.Module):
    trunk: eqx.Module
    branch: eqx.Module

    def __init__(self, trunk_params, branch_params, D, key):
        N_x, N_features, N_layers, kernel_size = trunk_params
        N_features_, N_layers_ = branch_params
        keys = random.split(key)
        if D == 1:
            self.trunk = conv_trunk_1D(N_x, N_features, N_layers, kernel_size, keys[0])
        else:
            self.trunk = conv_trunk_2D(N_x, N_features, N_layers, kernel_size, keys[0])
        self.branch = MLP(N_features_, N_layers_, keys[1])

    def __call__(self, u, coords, coords_x):
        coeff = self.trunk(u, coords)
        phi = self.branch(coords_x)
        res = jnp.expand_dims(dot_general(phi, coeff, (((0,), (0,)), ((), ()))), 0)
        return res

def l2_loss(model, input, target, x, coords):
    X = model(input, x, coords).reshape(target.shape[0], -1)
    error = jnp.mean(jnp.sum((X - target.reshape(target.shape[0], -1))**2, axis=1))
    return error

def batch_l2_loss(model, input, target, x, coords):
    res = vmap(l2_loss, in_axes=(None, 0, 0, None, None))(model, input, target, x, coords)
    return jnp.mean(res)

l2_compute_loss_and_grads = eqx.filter_value_and_grad(batch_l2_loss)

def l2_make_step_scan(carry, n, optim):
    model, features, targets, x, coords, opt_state = carry
    loss, grads = l2_compute_loss_and_grads(model, features[n], targets[n], x, coords)
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, features, targets, x, coords, opt_state], loss

def make_prediction_scan(carry, i):
    model, features, x, coords = carry
    prediction = model(features[i], x, coords)
    return carry, prediction