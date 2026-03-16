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
from architectures import reduced_BiFNOk_AE

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
       "-dataset_path": {
            "help": "absolute path to dataset"
        },
       "-results_path": {
            "help": "absolute path to folder where results are stored"
        },       
        "-learning_rate": {
            "default": 1e-4,
            "type": float,
            "help": "learning rate"
        },
        "-s1": {
            "default": 1e-2,
            "type": float,
            "help": "normalisation factor for convolutions"
        },
        "-s3": {
            "default": 1e-2,
            "type": float,
            "help": "normalisation factor for the Fourier kernel"
        },
        "-gamma": {
            "default": 0.5,
            "type": float,
            "help": "decay parameter for the exponential decay of learning rate per N_drop epochs"
        },
        "-N_drop": {
            "default": 100,
            "type": int,
            "help": "multiply learning rate by gamma each N_drop epoch"
        },
        "-N_batch": {
            "default": 10,
            "type": int,
            "help": "number of samples used to average gradient"
        },
        "-N_train": {
            "default": 800,
            "type": int,
            "help": "number of samples in the training set"
        },
        "-N_val": {
            "default": 100,
            "type": int,
            "help": "number of samples in the validation set"
        },
        "-N_test": {
            "default": 100,
            "type": int,
            "help": "number of samples in the test set"
        },
        "-N_epoch": {
            "default": 500,
            "type": int,
            "help": "number of updates of the model weights = N_epoch * N_train // N_batch"
        },
        "-stop_each": {
            "default": 50,
            "type": int,
            "help": "stop each N_epoch to evaluate the model and make checkpoint"
        },
        "-N_layers": {
            "default": 4,
            "type": int,
            "help": "number of layers in FNO"
        },
        "-N_processor": {
            "default": 32,
            "type": int,
            "help": "number of features in processor"
        },
        "-N_modes": {
            "default": 16,
            "type": int,
            "help": "number of modes in joint FNO kernel"
        },
        "-J_b": {
            "default": 5,
            "type": int,
            "help": "2**J_b points along each dimension in channel b"
        },
        "-N_f_b": {
            "default": 1,
            "type": int,
            "help": "number of learnable global fields in encoder and decoder"
        },
        "-kernel_size": {
            "default": 3,
            "type": int,
            "help": "size of convolution kernel"
        },
        "-n_basis": {
            "default": 30,
            "type": int,
            "help": "size of the code"
        },
        "-J_encoder": {
            "default": 0,
            "type": int,
            "help": "subsampling rate for each spatial dimension, e.g., x[:, ::2**J_a], encoder"
        },
        "-J_decoder": {
            "default": 0,
            "type": int,
            "help": "subsampling rate for each spatial dimension, e.g., x[:, ::2**J_b], decoder"
        },
        "-key": {
            "default": 14,
            "type": int,
            "help": "PRNGKey that seed all randomness in the code"
        },
        "-optim": {
            "default": 'adam',
            "type": str,
            "choices": ["lion", "adam"],
            "help": "optimiser"
        }
    }

    for key in args:
        parser.add_argument(key, **args[key])

    return parser

if __name__ == "__main__":
    parser = get_argparser()
    args = vars(parser.parse_args())
    exp_hash = "".join([str(args[a]) for a in sorted(args)])
    exp_hash = hashlib.sha256(str.encode(exp_hash)).hexdigest()
    dataset_name = args['dataset_path'].split("/")[-1]
    header = ",".join([key for key in args.keys()])
    header += ",hash,final_loss,model_size,training_time,train_error,test_error,val_error,best_n"

    if not os.path.isfile(f'{args["results_path"]}/results.csv'):
        with open(f'{args["results_path"]}/results.csv', "w") as f:
            f.write(header)
    
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
    
    N_layers = args["N_layers"]
    N_processor = args["N_processor"]
    N_modes = args["N_modes"]
    J_b = args["J_b"]
    N_f_b = args["N_f_b"]
    kernel_size = args["kernel_size"]
    n_basis = args["n_basis"]
    model = reduced_BiFNOk_AE.reduced_BiFNOk_AE(N_layers, N_processor, N_modes, D, J_b, N_f_b, kernel_size, n_basis, keys[1], s1=args['s1'], s2=0, s3=args['s3'])
    
    model_size = sum(jax.tree.map(lambda x: (2*jnp.size(x) if x.dtype == jnp.complex64 else jnp.size(x)) if not (x is None) else 0, jax.tree.flatten(model)[0], is_leaf=eqx.is_array))

    learning_rate = optax.exponential_decay(args["learning_rate"], N_drop, args["gamma"])
    if args["optim"] == "lion":
        optim = optax.lion(learning_rate=learning_rate)
    else:
        optim = optax.adam(learning_rate=learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    
    nn = random.choice(keys[2], args["N_train"], shape = (N_run//N_stop, N_stop, args["N_batch"]))
    val_ind = args["N_train"] + jnp.arange(args["N_val"])
    train_ind = jnp.arange(args["N_train"])
    test_ind = -(1 + jnp.arange(args["N_test"]))
    carry = [model, x_encoder, x_decoder, targets_encoder, targets_decoder, opt_state]
    make_step_scan_ = lambda a, b: make_step_scan(a, b, optim)

    training_time = 0
    models = []
    opt_states = []
    val_rel_errors = []
    training_times = []
    histories = []
    for nn_ in nn:
        start = time.time()
        carry, history = scan(make_step_scan_, carry, nn_)
        stop = time.time()
        training_time = training_time + stop - start
        model = carry[0]
        opt_state = carry[-1]
        models.append(model)
        opt_states.append(opt_state)
        rel_errors = scan(compute_errors, [model, x_encoder, x_decoder, targets_encoder, targets_decoder], val_ind)[1]
        val_rel_errors.append(jnp.mean(rel_errors))
        training_times.append(training_time)
        histories.append(history)
        if jnp.isnan(val_rel_errors[-1]).item():
            break
        
    val_rel_errors = jnp.array(val_rel_errors)
    val_rel_errors = jnp.nan_to_num(val_rel_errors, nan=jnp.inf)
    best_n = jnp.argmin(val_rel_errors)
    concat_n = min(best_n + 1, len(histories))
    
    history = jnp.concatenate(histories[:concat_n])
    eqx.tree_serialise_leaves(f'{args["results_path"]}/model_{exp_hash}.eqx', models[best_n])

    train_rel_errors = jnp.mean(scan(compute_errors, [models[best_n], x_encoder, x_decoder, targets_encoder, targets_decoder], train_ind)[1])
    val_rel_errors = jnp.mean(scan(compute_errors, [models[best_n], x_encoder, x_decoder, targets_encoder, targets_decoder], val_ind)[1])
    test_rel_errors = jnp.mean(scan(compute_errors, [models[best_n], x_encoder, x_decoder, targets_encoder, targets_decoder], test_ind)[1])
    data = "\n" + ",".join([str(args[key]) for key in args.keys()])
    data += f",{exp_hash},{history[-1]},{model_size},{training_times[best_n]},{train_rel_errors},{test_rel_errors},{val_rel_errors},{best_n}"
                                
    with open(f'{args["results_path"]}/results.csv', "a") as f:
        f.write(data)
                                
    jnp.savez(f'{args["results_path"]}/metrics_{exp_hash}.npz', rel_errors=rel_errors, history=history, permutation=perm)