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

class MLP_trunk(eqx.Module):
    weights: list
    biases: list

    def __init__(self, N_features, N_layers, key):
        N_in, N_processor, N_out = N_features
        keys = random.split(key, N_layers+2)
        Ns = [N_in,] + [N_processor,]*N_layers + [N_out,]
        self.biases = [jnp.zeros((No, )) for No in Ns[1:]]
        self.weights = [random.normal(key, (No, Ni)) / jnp.sqrt(No + Ni) for Ni, No, key in zip(Ns[:-1], Ns[1:], keys)]

    def __call__(self, f):
        f = self.weights[0] @ f + self.biases[0]
        for w, b in zip(self.weights[1:-1], self.biases[1:-1]):
            f = gelu(w @ f + b) + f
        f = self.weights[-1] @ f + self.biases[-1]
        return f

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

    def __init__(self, trunk_params, branch_params, key):
        N_features, N_layers = trunk_params
        N_features_, N_layers_ = branch_params
        keys = random.split(key)
        self.trunk = MLP_trunk(N_features, N_layers, keys[0])
        self.branch = MLP(N_features_, N_layers_, keys[1])

    def __call__(self, u, coords_x):
        coeff = self.trunk(u)
        phi = self.branch(coords_x)
        res = jnp.expand_dims(dot_general(phi, coeff, (((0,), (0,)), ((), ()))), 0)
        return res

def l2_loss(model, input, target, coords):
    X = model(input, coords).reshape(target.shape[0], -1)
    error = jnp.mean(jnp.sum((X - target.reshape(target.shape[0], -1))**2, axis=1))
    return error

def batch_l2_loss(model, input, target, coords):
    res = vmap(l2_loss, in_axes=(None, 0, 0, None))(model, input, target, coords)
    return jnp.mean(res)

l2_compute_loss_and_grads = eqx.filter_value_and_grad(batch_l2_loss)

def l2_make_step_scan(carry, n, optim):
    model, features, targets, coords, opt_state = carry
    loss, grads = l2_compute_loss_and_grads(model, features[n], targets[n], coords)
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, features, targets, coords, opt_state], loss

def make_prediction_scan(carry, i):
    model, features, coords = carry
    prediction = model(features[i], coords)
    return carry, prediction