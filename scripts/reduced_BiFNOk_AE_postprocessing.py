import pandas as pd
import os
import time
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import hashlib
import argparse
import jax

from jax import random, vmap
from jax.lax import scan
from jax.tree_util import tree_map, tree_flatten
from architectures import MLP_BiFNOk_AE

def subsample_field(field, J, D):
    if D == 1:
        field = field[:, :, ::2**J]
    elif D == 2:
        field = field[:, :, ::2**J, ::2**J]
    else:
        field = field[:, :, ::2**J, ::2**J, ::2**J]
    return jnp.array(field)

def normalise_field(field, D):
    if D == 1:
        axis = [0, 2]
    elif D == 2:
        axis = [0, 2, 3]
    else:
        axis = [0, 2, 3, 4]
    norm_factor = jnp.max(jnp.abs(field), axis=axis, keepdims=True)
    norm_factor = norm_factor + (norm_factor == 0)
    field = field / norm_factor
    return field

def get_coordinates(N_x, D):
    x = jnp.linspace(0, 1, N_x)
    coordinates = jnp.stack(jnp.meshgrid(*[x for _ in range(D)]))
    return coordinates

def compute_errors(carry, ind):
    model, x_encoder, x_decoder, targets_encoder, targets_decoder = carry
    z = model.encode(targets_encoder[ind], x_encoder)
    prediction = model.decode(z, x_decoder)
    error = jnp.linalg.norm((prediction - targets_decoder[ind]).reshape(prediction.shape[0], -1)) / jnp.linalg.norm(targets_decoder[ind].reshape(prediction.shape[0], -1))
    return carry, error

def encode_dataset(carry, ind):
    model, x_encoder, targets_encoder = carry
    z = model.encode(targets_encoder[ind], x_encoder)
    return carry, z

def l2_loss(model, x_encoder, x_decoder, target_encoder, target_decoder):
    z = model.encode(target_encoder, x_encoder)
    prediction = model.decode(z, x_decoder)
    error = jnp.sum((prediction - target_decoder)**2)
    return error

def batch_l2_loss(model, x_encoder, x_decoder, target_encoder, target_decoder):
    res = vmap(l2_loss, in_axes=[None, None, None, 0, 0])(model, x_encoder, x_decoder, target_encoder, target_decoder)
    return jnp.mean(res)

compute_loss_and_grads = eqx.filter_value_and_grad(batch_l2_loss)

def make_step_scan(carry, n, optim):
    model, x_encoder, x_decoder, targets_encoder, targets_decoder, opt_state = carry
    loss, grads = compute_loss_and_grads(model, x_encoder, x_decoder, targets_encoder[n], targets_decoder[n])
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, x_encoder, x_decoder, targets_encoder, targets_decoder, opt_state], loss

def get_argparser():
    parser = argparse.ArgumentParser()
    args = {
       "-results_path": {
            "help": "absolute path to the results.csv"
        },
       "-save_path": {
            "help": "absolute path where to save postprocessing results"
        },
        "-n_basis": {
            "default": 5,
            "type": int,
            "help": "size of the code"
        },
        "-J_encoder": {
            "default": 2,
            "type": int,
            "help": "subsampling rate for each spatial dimension used for training"
        },
        "-J_decoder": {
            "default": 2,
            "type": int,
            "help": "subsampling rate for each spatial dimension used for training"
        }
    }

    for key in args:
        parser.add_argument(key, **args[key])

    return parser

if __name__ == "__main__":
    parser = get_argparser()
    Args = vars(parser.parse_args())
    results_path = Args['results_path']
    J_encoder = Args['J_encoder']
    J_decoder = Args['J_decoder']
    n_basis = Args['n_basis']

    if not os.path.isfile(f'{Args["save_path"]}/postprocessing_reconstruction_quality.csv'):
        with open(f'{Args["save_path"]}/postprocessing_reconstruction_quality.csv', "w") as f:
            header = 'equation,n_basis,J_encoder_train,J_decoder_train,J_encoder_test,J_decoder_test,reconstruction_error'
            f.write(header)

    if not os.path.isfile(f'{Args["save_path"]}/postprocessing_codes_quality.csv'):
        with open(f'{Args["save_path"]}/postprocessing_codes_quality.csv', "w") as f:
            header = 'equation,n_basis,J_encoder_train,J_encoder_test,code_error'
            f.write(header)

    res = pd.read_csv(results_path)
    res = res[res['J_encoder'] == J_encoder]
    res = res[res['J_decoder'] == J_decoder]
    res = res[res['n_basis'] == n_basis]
    equations = res['dataset_path'].unique()

    for equation in equations:
        res_ = res[res['dataset_path'] == equation]
        res_ = res_.iloc[res_['test_error'].argmin()]
        args = {key:res_[key] for key in res_.keys()}
    
        key = random.PRNGKey(args["key"])
        keys = random.split(key, 3)
        data = jnp.load(args['dataset_path'])
        targets = data['targets']
        D = targets.ndim - 2
        targets = normalise_field(targets, D)
        perm = random.permutation(keys[0], targets.shape[0])
        targets_encoder = subsample_field(targets[perm], args['J_encoder'], D)
        targets_decoder = subsample_field(targets[perm], args['J_decoder'], D)
        
        x_encoder = subsample_field(np.expand_dims(data['coordinates'], 0), args['J_encoder'], D)[0]
        x_decoder = subsample_field(np.expand_dims(data['coordinates'], 0), args['J_decoder'], D)[0]
        
        N_run = args["N_epoch"] * args["N_train"] // args["N_batch"]
        N_drop = args["N_drop"] * args["N_train"] // args["N_batch"]
        N_stop = args["stop_each"] * args["N_train"] // args["N_batch"]
        
        N_layers = args["N_layers"].item()
        N_processor = args["N_processor"].item()
        N_modes = args["N_modes"].item()
        J_b = args["J_b"].item()
        N_f_b = args["N_f_b"].item()
        kernel_size = args["kernel_size"].item()
        n_basis = args["n_basis"].item()
        model = MLP_BiFNOk_AE.MLP_BiFNOk_AE(N_layers, N_processor, N_modes, D, J_b, N_f_b, kernel_size, n_basis, keys[1], s1=args['s1'], s2=0, s3=args['s3'])
        model = eqx.tree_deserialise_leaves(res_['results_path'] + "/" + f"model_{res_['hash']}.eqx", model)

        to_append = ""
        # overall quality of reconstruction depending on the resolution of input and output
        for J_encoder_new in [4, 3, 2, 1, 0]:
            for J_decoder_new in [4, 3, 2, 1, 0]:
                targets_encoder_ = subsample_field(targets[perm], J_encoder_new , D)
                targets_decoder_ = subsample_field(targets[perm], J_decoder_new, D)
                
                x_encoder_ = subsample_field(np.expand_dims(data['coordinates'], 0), J_encoder_new, D)[0]
                x_decoder_ = subsample_field(np.expand_dims(data['coordinates'], 0), J_decoder_new, D)[0]

                if x_encoder_.shape[-1] < 32 or x_decoder_.shape[-1] < 32:
                    pass
                else:
                    test_ind = -(1 + jnp.arange(args["N_test"]))
                    test_rel_errors = jnp.mean(scan(compute_errors, [model, x_encoder_, x_decoder_, targets_encoder_, targets_decoder_], test_ind)[1])
                    to_append += f"\n{args['dataset_path'].split('/')[-1].split('.')[0]},{n_basis},{args['J_encoder']},{args['J_decoder']},{J_encoder_new},{J_decoder_new},{test_rel_errors}"
                    
        with open(f'{Args["save_path"]}/postprocessing_reconstruction_quality.csv', "a") as f:
            f.write(to_append)
        
        # distance between codes for distinct resolution of encoder
        test_ind = -(1 + jnp.arange(args["N_test"]))
        targets_encoder = subsample_field(targets[perm], args['J_encoder'], D)
        targets_decoder = subsample_field(targets[perm], args['J_decoder'], D)
        
        x_encoder = subsample_field(np.expand_dims(data['coordinates'], 0), args['J_encoder'], D)[0]
        x_decoder = subsample_field(np.expand_dims(data['coordinates'], 0), args['J_decoder'], D)[0]
        
        gt_codes = scan(encode_dataset, [model, x_encoder, targets_encoder], test_ind)[1]
        gt_norm = jnp.linalg.norm(gt_codes, axis=1)

        to_append = ""
        for J_encoder_new in [4, 3, 2, 1, 0]:
            targets_encoder_ = subsample_field(targets[perm], J_encoder_new , D)
            x_encoder_ = subsample_field(np.expand_dims(data['coordinates'], 0), J_encoder_new , D)[0]
            if x_encoder_.shape[-1] < 32:
                pass
            else:
                test_ind = -(1 + jnp.arange(args["N_test"]))
                codes = scan(encode_dataset, [model, x_encoder_, targets_encoder_], test_ind)[1]
                test_rel_errors = jnp.mean(jnp.linalg.norm(gt_codes - codes, axis=1) / gt_norm)
                to_append += f"\n{args['dataset_path'].split('/')[-1].split('.')[0]},{n_basis},{args['J_encoder']},{J_encoder_new},{test_rel_errors}"
    
        with open(f'{Args["save_path"]}/postprocessing_codes_quality.csv', "a") as f:
            f.write(to_append)