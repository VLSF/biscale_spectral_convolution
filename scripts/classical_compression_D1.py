import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pywt
import os
import argparse

from jax.lax import scan

def subsample(field, N):
    ind = jnp.linspace(0, field.shape[0]-1, N, dtype=int)
    subsampled_field = field[ind]#jnp.concatenate([field[:1], field[1:-1][::2**J], field[-1:]], axis=0)
    return subsampled_field

def interpolation_error(x, field, N):
    x_coarse = subsample(x, N)
    field_coarse = subsample(field, N)
    interpolated = jnp.interp(x, x_coarse, field_coarse)
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

def get_slices(wavelet, target):
    coeffs = pywt.wavedec(target, wavelet)
    shapes = [c.shape[0] for c in coeffs]
    counter = 0
    slices = [0,]
    for s in shapes:
        counter += s
        slices.append(counter)
    slices[-1] = -1
    return slices

def compression_error(wavelet, target, N, slices):
    coeffs = pywt.wavedec(target, wavelet)
    Coeffs = np.concatenate(coeffs)
    order = np.argsort(np.abs(Coeffs))
    Coeffs[jnp.argsort(jnp.abs(Coeffs))[:-N]] = 0
    
    coeffs_ = []
    for start, stop in zip(slices[:-1], slices[1:]):
        if stop == -1:
            coeffs_.append(Coeffs[start:])
        else:
             coeffs_.append(Coeffs[start:stop])
    
    res = pywt.waverec(coeffs_, wavelet)
    error = np.linalg.norm(res - target) / np.linalg.norm(target)
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

    x = data['coordinates'][0]
    targets = data['targets'][:, 0]
    
    N_x = [5, 10, 15, 20]
    N_train = 800
    POD_errors, sigma = compresion_error_from_POD(targets, N_x, N_train)
    errors = []
    for N in N_x:
        carry = [x, targets]
        ind = jnp.arange(targets.shape[0])
        interpolation_error_scan_ = lambda a, b: interpolation_error_scan(a, b, N)
        _, error = scan(interpolation_error_scan_, carry, ind)
        errors.append(jnp.mean(error))
    
    targets = np.array(data['targets'])
    wavelet = pywt.Wavelet('bior2.2')
    slices = get_slices(wavelet, targets[0, 0])
    
    wavelet_errors = []
    for N in N_x:
        errors_ = []
        for t in targets[:N_train, 0]:
            error = compression_error(wavelet, t, N, slices)
            errors_.append(error)
        wavelet_errors.append(np.mean(np.array(errors_)))

    for i in range(len(N_x)):
        data = f"\n{args['dataset_path']},{N_x[i]},{errors[i]},{POD_errors[i]},{wavelet_errors[i]}"
        with open(f'{args["results_path"]}/results.csv', "a") as f:
            f.write(data)