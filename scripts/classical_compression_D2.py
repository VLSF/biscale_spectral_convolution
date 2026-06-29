import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pywt
import os
import argparse

from jax.lax import scan
from jax import vmap
from jax.tree import flatten, unflatten
from jax.tree_util import tree_map

def subsample(field, N):
    ind = jnp.linspace(0, field.shape[0]-1, N, dtype=int)
    subsampled_field = field[ind]
    if field.ndim >= 2:
        subsampled_field = subsampled_field[:, ind]
    if field.ndim >= 3:
        subsampled_field = subsampled_field[:, :, ind]
    return subsampled_field

def interpolation_error(x, field, N):
    x_coarse = subsample(x, N)
    field_coarse = subsample(field, N)
    interpolated = vmap(jnp.interp, in_axes=(None, None, 0))(x, x_coarse, field_coarse)
    interpolated = vmap(jnp.interp, in_axes=(None, None, 1), out_axes=1)(x, x_coarse, interpolated)
    error = jnp.linalg.norm(field - interpolated) / jnp.linalg.norm(field)
    return error

def interpolation_error_scan(carry, ind, N):
    x, fields = carry
    error = interpolation_error(x, fields[ind], N)
    return carry, error

def compresion_error_from_POD(targets, N, N_train):
    U, sigma, Vt = jnp.linalg.svd(targets[:N_train])
    errors = []
    for n in N:
        V = Vt[:n]
        error = jnp.linalg.norm((targets[N_train:] @ V.T) @ V - targets[N_train:], axis=1) / jnp.linalg.norm(targets[N_train:], axis=1)
        errors.append(jnp.mean(error))
    return errors, sigma

def compression_error(wavelet, target, N):
    coeffs = [pywt.wavedec(c, wavelet, axis=0) for c in pywt.wavedec(target, wavelet, axis=1)]
    shapes_2D = tree_map(jnp.shape, coeffs)
    coeffs = tree_map(lambda x: np.reshape(x, -1), coeffs)
    coeffs, treedef = flatten(coeffs)
    slices_1D = np.cumsum(np.array([0,] + tree_map(lambda x: np.shape(x)[0], coeffs)))
    
    coeffs = np.concatenate(coeffs)
    discard = np.argsort(np.abs(coeffs))[:-N]
    coeffs[discard] = 0
    
    coeffs = [coeffs[start:stop] for start, stop in zip(slices_1D[:-1], slices_1D[1:])]
    
    coeffs = unflatten(treedef, coeffs)
    coeffs = tree_map(lambda x, y: jnp.reshape(x, y), coeffs, shapes_2D)
    compressed = pywt.waverec([pywt.waverec(c, wavelet, axis=0) for c in coeffs], wavelet, axis=1)
    error = np.linalg.norm((compressed - target).reshape(-1, )) / np.linalg.norm((target).reshape(-1, ))
    return error

def get_argparser():
    parser = argparse.ArgumentParser()
    args = {
       "-dataset_path": {
            "help": "absolute path to dataset"
        },
       "-results_path": {
            "help": "absolute path to folder where results are stored"
        }
    }

    for key in args:
        parser.add_argument(key, **args[key])

    return parser

if __name__ == "__main__":
    parser = get_argparser()
    args = vars(parser.parse_args())
    data = jnp.load(args['dataset_path'])
    header = "dataset,N,interpolation,POD,wavelets"
    if not os.path.isfile(f'{args["results_path"]}/results.csv'):
        with open(f'{args["results_path"]}/results.csv', "w") as f:
            f.write(header)

    targets = jnp.array(data['targets'])
    x = jnp.array(data['coordinates'][0, :, 0])

    errors = []
    N_x = [64, 128, 256, 512, 1024, 2048, 4096]
    for n in N_x:
        k = int(np.sqrt(n))
        if k in [8, 16, 32, 64]:
            ind = jnp.arange(targets.shape[0])
            interpolation_error_scan_ = lambda a, b: interpolation_error_scan(a, b, k)
            errors.append(jnp.mean(scan(interpolation_error_scan_, [x, targets[:, 0]], ind)[1]))
        else:
            errors.append(jnp.nan)
    N_train = 800
    POD_errors, _ = compresion_error_from_POD(targets[:, 0].reshape(targets.shape[0], -1), N_x, N_train)
    
    targets = np.array(data['targets'])[:, 0]
    x = np.array(data['coordinates'][0, :, 0])
    wavelet = pywt.Wavelet('bior2.2')
    
    wavelet_errors = []
    for n in N_x:
        error = 0
        for t in targets:
            error += compression_error(wavelet, t, n)
        error /= targets.shape[0]
        wavelet_errors.append(error)

    for i in range(len(N_x)):
        data = f"\n{args['dataset_path']},{N_x[i]},{errors[i]},{POD_errors[i]},{wavelet_errors[i]}"
        with open(f'{args["results_path"]}/results.csv', "a") as f:
            f.write(data)