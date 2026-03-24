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
from architectures import conv_DeepONet

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

def get_argparser():
    parser = argparse.ArgumentParser()
    args = {
       "-dataset_path": {
            "help": "absolute path to dataset with eigenvectors"
        },
       "-results_path": {
            "help": "absolute path to folder where results are stored"
        },       
        "-learning_rate": {
            "default": 1e-4,
            "type": float,
            "help": "learning rate"
        },
        "-gamma": {
            "default": 0.5,
            "type": float,
            "help": "decay parameter for the exponential decay of learning rate per N_drop epochs"
        },
        "-N_drop": {
            "default": 10000,
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
        "-N_epoch": {
            "default": 1000,
            "type": int,
            "help": "number of updates of the model weights = N_epoch * N_train // N_batch"
        },
        "-stop_each": {
            "default": 50,
            "type": int,
            "help": "stop each N_epoch to evaluate the model and make checkpoint"
        },
        "-N_layers_trunk": {
            "default": 4,
            "type": int,
            "help": "number of layers in trunk net"
        },
        "-N_layers_branch": {
            "default": 3,
            "type": int,
            "help": "number of layers in branch net"
        },
        "-kernel_size_trunk": {
            "default": 3,
            "type": int,
            "help": "kernel size of conv in trunk net"
        },
        "-N_encoder_trunk": {
            "default": 5,
            "type": int,
            "help": "number of features in trunk net after encoder, doupls each layer"
        },
        "-N_basis": {
            "default": 100,
            "type": int,
            "help": "number of basis functions in branch net"
        },
        "-N_features_branch": {
            "default": 64,
            "type": int,
            "help": "number of hidden neurons in branch net"
        },
        "-key": {
            "default": 14,
            "type": int,
            "help": "PRNGKey that seed all randomness in the code"
        }
    }

    for key in args:
        parser.add_argument(key, **args[key])

    return parser

if __name__ == "__main__":
    parser = get_argparser()
    args = vars(parser.parse_args())
    header = ",".join([key for key in args.keys()])
    header += ",hash,final_loss,model_size,training_time,train_error,test_error,val_error,best_n"

    if not os.path.isfile(f'{args["results_path"]}/results.csv'):
        with open(f'{args["results_path"]}/results.csv', "w") as f:
            f.write(header)
            
    key = random.PRNGKey(args["key"])
    keys = random.split(key, 3)
    data = np.load(args['dataset_path'])
    coordinates = jnp.array(data['coordinates'])
    D = data['coordinates'].shape[0]
    features = jnp.array(data['features'])
    targets = jnp.array(data["targets"])
    features = normalise_field(features, D)
    targets = normalise_field(targets, D)
    perm = random.permutation(keys[0], targets.shape[0])
    targets = targets[perm]
    features = features[perm]

    exp_hash = "".join([str(args[a]) for a in sorted(args)])
    exp_hash = hashlib.sha256(str.encode(exp_hash)).hexdigest()

    D = features.ndim - 2
    N_run = args["N_epoch"] * args["N_train"] // args["N_batch"]
    N_drop = args["N_drop"] * args["N_train"] // args["N_batch"]
    N_stop = args["stop_each"] * args["N_train"] // args["N_batch"]

    key = random.PRNGKey(args["key"])
    keys = random.split(key, 3)
    
    trunk_params = [coordinates.shape[-1], [coordinates.shape[0] + features.shape[1], args["N_encoder_trunk"], args["N_basis"]], args["N_layers_trunk"], args["kernel_size_trunk"]]
    branch_params = [[coordinates.shape[0], args["N_features_branch"], args["N_basis"]], args["N_layers_branch"]]
    model = conv_DeepONet.DeepONet(trunk_params, branch_params, D, keys[0])
    
    model_size = sum(tree_map(lambda x: jnp.size(x) if not (x is None) else 0, tree_flatten(model)[0], is_leaf=eqx.is_array))
    learning_rate = optax.exponential_decay(args["learning_rate"], N_drop, args["gamma"])
    optim = optax.lion(learning_rate=learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    
    nn = random.choice(keys[1], args["N_train"], shape = (N_run//N_stop, N_stop, args["N_batch"]))
    carry = [model, features, targets, coordinates, coordinates, opt_state]
    make_step_scan_ = lambda a, b: conv_DeepONet.l2_make_step_scan(a, b, optim)

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
        _, predictions = scan(conv_DeepONet.make_prediction_scan, [model, features[args["N_train"]:(args["N_train"]+args["N_val"])], coordinates, coordinates], jnp.arange(features[args["N_train"]:(args["N_train"]+args["N_val"])].shape[0]))
        rel_errors = jnp.linalg.norm(predictions.reshape(predictions.shape[0], -1) - targets[args["N_train"]:(args["N_train"]+args["N_val"])].reshape(targets[args["N_train"]:(args["N_train"]+args["N_val"])].shape[0], -1), axis=1) / jnp.linalg.norm(targets[args["N_train"]:(args["N_train"]+args["N_val"])].reshape(targets[args["N_train"]:(args["N_train"]+args["N_val"])].shape[0], -1), axis=1)
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

    _, predictions = scan(conv_DeepONet.make_prediction_scan, [models[best_n], features, coordinates, coordinates], jnp.arange(features.shape[0]))
    rel_errors = jnp.linalg.norm(predictions.reshape(predictions.shape[0], -1) - targets.reshape(targets.shape[0], -1), axis=1) / jnp.linalg.norm(targets.reshape(targets.shape[0], -1), axis=1)
    
    train_rel_errors = jnp.mean(rel_errors[:args["N_train"]])
    val_rel_errors = jnp.mean(rel_errors[args["N_train"]:(args["N_train"]+args["N_val"])])
    test_rel_errors = jnp.mean(rel_errors[(args["N_train"]+args["N_val"]):])

    data = "\n" + ",".join([str(args[key]) for key in args.keys()])
    data += f",{exp_hash},{history[-1]},{model_size},{training_times[best_n]},{train_rel_errors},{test_rel_errors},{val_rel_errors},{best_n}"

    with open(f'{args["results_path"]}/results.csv', "a") as f:
        f.write(data)
    
    jnp.savez(f'{args["results_path"]}/metrics_{exp_hash}.npz', rel_errors=rel_errors, history=history, perm=perm)