import os
import time
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import hashlib
import argparse

from jax import random, vmap
from jax.lax import scan
from jax.tree_util import tree_map, tree_flatten
from architectures import BiFNO

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
    model, features_a, x_a, features_b, x_b, targets = carry
    if features_a is None:
        prediction = model(features_a, x_a, features_b[ind], x_b)[0]
    elif features_b is None:
        prediction = model(features_a[ind], x_a, features_b, x_b)[0]
    else:
        prediction = model(features_a[ind], x_a, features_b[ind], x_b)[0]
    error = jnp.linalg.norm((prediction - targets[ind]).reshape(prediction.shape[0], -1)) / jnp.linalg.norm(targets[ind].reshape(prediction.shape[0], -1))
    return carry, error

def l2_loss(model, feature_a, x_a, feature_b, x_b, target):
    X = model(feature_a, x_a, feature_b, x_b)[0]
    error = jnp.sum((X - target)**2)
    return error

def batch_l2_loss(model, feature_a, x_a, feature_b, x_b, target):
    if features_a is None:
        in_axes = (None, None, None, 0, None, 0)
    elif features_b is None:
        in_axes = (None, 0, None, None, None, 0)
    else:
        in_axes = (None, 0, None, 0, None, 0)
    res = vmap(l2_loss, in_axes=in_axes)(model, feature_a, x_a, feature_b, x_b, target)
    return jnp.mean(res)

compute_loss_and_grads = eqx.filter_value_and_grad(batch_l2_loss)

def make_step_scan(carry, n, optim):
    model, features_a, x_a, features_b, x_b, targets, opt_state = carry
    if features_a is None:
        loss, grads = compute_loss_and_grads(model, features_a, x_a, features_b[n], x_b, targets[n])
    elif features_b is None:
        loss, grads = compute_loss_and_grads(model, features_a[n], x_a, features_b, x_b, targets[n])
    else:
        loss, grads = compute_loss_and_grads(model, features_a[n], x_a, features_b[n], x_b, targets[n])
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, features_a, x_a, features_b, x_b, targets, opt_state], loss

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
            "default": 1000,
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
            "help": "number of layers in MLP"
        },
        "-N_processor_a": {
            "default": 32,
            "type": int,
            "help": "number of features in processor, channel a"
        },
        "-N_modes": {
            "default": 16,
            "type": int,
            "help": "number of modes in joint FNO kernel"
        },
        "-N_processor_b": {
            "default": 32,
            "type": int,
            "help": "number of features in processor, channel b"
        },
        "-J_a": {
            "default": 0,
            "type": int,
            "help": "subsampling rate for each spatial dimension, e.g., x[:, ::2**J_a], channel a"
        },
        "-J_b": {
            "default": 0,
            "type": int,
            "help": "subsampling rate for each spatial dimension, e.g., x[:, ::2**J_b], channel b"
        },
        "-Nx_b": {
            "default": 0,
            "type": int,
            "help": "if nonzero, separate coordinates x_b are created with specified resolution"
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
        },
        "-features_input": {
            "default": 'a',
            "type": str,
            "choices": ["a", "b"],
            "help": "selected channel take features as input"
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
    targets = normalise_field(subsample_field(targets, args['J_a'], D), D)
    perm = random.permutation(keys[0], targets.shape[0])
    targets = targets[perm]
    
    if args['features_input'] == 'a':
        features_a = normalise_field(subsample_field(data['features'], args['J_a'], D), D)[perm]
        features_b = None
    else:
        features_a = None
        features_b = normalise_field(subsample_field(data['features'], args['J_b'], D), D)[perm]

    coordinates_a = subsample_field(np.expand_dims(data['coordinates'], 0), args['J_a'], D)[0]
    if args['Nx_b'] != 0:
        coordinates_b = get_coordinates(args['Nx_b'], D)
    else:
        coordinates_b = subsample_field(np.expand_dims(data['coordinates'], 0), args['J_b'], D)[0]

    N_run = args["N_epoch"] * args["N_train"] // args["N_batch"]
    N_drop = args["N_drop"] * args["N_train"] // args["N_batch"]
    N_stop = args["stop_each"] * args["N_train"] // args["N_batch"]
    
    N_layers = args["N_layers"]
    N_features_a = [coordinates_a.shape[0] + (0 if features_a is None else features_a.shape[1]), args['N_processor_a'], targets.shape[1]]
    N_features_b = [coordinates_b.shape[0] + (0 if features_b is None else features_b.shape[1]), args['N_processor_b']]
    N_modes = args["N_modes"]
    model = BiFNO.BiFNO(N_layers, N_features_a, N_features_b, N_modes, D, keys[1], s1=args['s1'], s2=0, s3=args['s3'])
    
    model_size = sum(tree_map(lambda x: jnp.size(x) if not (x is None) else 0, tree_flatten(model)[0], is_leaf=eqx.is_array))
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
    carry = [model, features_a, coordinates_a, features_b, coordinates_b, targets, opt_state]
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
        rel_errors = scan(compute_errors, [model, features_a, coordinates_a, features_b, coordinates_b, targets], val_ind)[1]
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
    eqx.tree_serialise_leaves(f'{args["results_path"]}/opt_state_{exp_hash}.eqx', opt_states[best_n])

    train_rel_errors = jnp.mean(scan(compute_errors, [models[best_n], features_a, coordinates_a, features_b, coordinates_b, targets], train_ind)[1])
    val_rel_errors = jnp.mean(scan(compute_errors, [models[best_n], features_a, coordinates_a, features_b, coordinates_b, targets], val_ind)[1])
    test_rel_errors = jnp.mean(scan(compute_errors, [models[best_n], features_a, coordinates_a, features_b, coordinates_b, targets], test_ind)[1])
    data = "\n" + ",".join([str(args[key]) for key in args.keys()])
    data += f",{exp_hash},{history[-1]},{model_size},{training_times[best_n]},{train_rel_errors},{test_rel_errors},{val_rel_errors},{best_n}"
                                
    with open(f'{args["results_path"]}/results.csv', "a") as f:
        f.write(data)
                                
    jnp.savez(f'{args["results_path"]}/metrics_{exp_hash}.npz', rel_errors=rel_errors, history=history, permutation=perm)