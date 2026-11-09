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
from architectures import Conv_AE


def subsample_field(field, J, D):
    if D == 1:
        field = field[:, :, :: 2**J]
    elif D == 2:
        field = field[:, :, :: 2**J, :: 2**J]
    else:
        field = field[:, :, :: 2**J, :: 2**J, :: 2**J]
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


def parse_tuple_or_int(val):
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        if "," in val:
            return tuple(int(x.strip()) for x in val.split(","))
        try:
            return int(val)
        except ValueError:
            return val
    return val


def compute_errors(carry, ind):
    model, targets_encoder, targets_decoder = carry
    z = model.encode(targets_encoder[ind])
    prediction = model.decode(z)
    error = jnp.linalg.norm(
        (prediction - targets_decoder[ind]).reshape(prediction.shape[0], -1)
    ) / jnp.linalg.norm(targets_decoder[ind].reshape(prediction.shape[0], -1))
    return carry, error


def l2_loss(model, target_encoder, target_decoder):
    z = model.encode(target_encoder)
    prediction = model.decode(z)
    error = jnp.sum((prediction - target_decoder) ** 2)
    return error


def batch_l2_loss(model, target_encoder, target_decoder):
    res = vmap(l2_loss, in_axes=[None, 0, 0])(model, target_encoder, target_decoder)
    return jnp.mean(res)


compute_loss_and_grads = eqx.filter_value_and_grad(batch_l2_loss)


def make_step_scan(carry, n, optim):
    model, targets_encoder, targets_decoder, opt_state = carry
    loss, grads = compute_loss_and_grads(
        model, targets_encoder[n], targets_decoder[n]
    )
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, targets_encoder, targets_decoder, opt_state], loss


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
            "help": "learning rate",
        },
        "-s1": {
            "default": 1e-2,
            "type": float,
            "help": "normalisation factor for convolutions",
        },
        "-s2": {
            "default": 0.0,
            "type": float,
            "help": "normalisation factor for convolution biases",
        },
        "-gamma": {
            "default": 0.5,
            "type": float,
            "help": "decay parameter for the exponential decay of learning rate per N_drop epochs",
        },
        "-N_drop": {
            "default": 100,
            "type": int,
            "help": "multiply learning rate by gamma each N_drop epoch",
        },
        "-N_batch": {
            "default": 10,
            "type": int,
            "help": "number of samples used to average gradient",
        },
        "-N_train": {
            "default": 800,
            "type": int,
            "help": "number of samples in the training set",
        },
        "-N_val": {
            "default": 100,
            "type": int,
            "help": "number of samples in the validation set",
        },
        "-N_test": {
            "default": 100,
            "type": int,
            "help": "number of samples in the test set",
        },
        "-N_epoch": {
            "default": 500,
            "type": int,
            "help": "number of updates of the model weights = N_epoch * N_train // N_batch",
        },
        "-stop_each": {
            "default": 50,
            "type": int,
            "help": "stop each N_epoch to evaluate the model and make checkpoint",
        },
        "-N_blocks": {
            "default": 4,
            "type": int,
            "help": "number of residual blocks in encoder and decoder",
        },
        "-N_processor": {
            "default": 264,
            "type": int,
            "help": "number of features in processor",
        },
        "-kernel_size": {
            "default": "3",
            "type": str,
            "help": "size of convolution kernel (int or comma-separated ints)",
        },
        "-subsampling": {
            "default": "32",
            "type": str,
            "help": "subsampling factor along each dimension made by the encoder",
        },
        "-n_codes": {
            "default": 1,
            "type": int,
            "help": "number of channels in the code",
        },
        "-num_groups": {
            "default": 4,
            "type": int,
            "help": "number of groups for GroupNorm",
        },
        "-J_encoder": {
            "default": 0,
            "type": int,
            "help": "subsampling rate for each spatial dimension, e.g., x[:, ::2**J_a], encoder",
        },
        "-J_decoder": {
            "default": 0,
            "type": int,
            "help": "subsampling rate for each spatial dimension, e.g., x[:, ::2**J_b], decoder",
        },
        "-key": {
            "default": 14,
            "type": int,
            "help": "PRNGKey that seed all randomness in the code",
        },
        "-optim": {
            "default": "adam",
            "type": str,
            "choices": ["lion", "adam"],
            "help": "optimiser",
        },
    }

    for key in args:
        parser.add_argument(key, **args[key])

    return parser


if __name__ == "__main__":
    parser = get_argparser()
    args = vars(parser.parse_args())
    exp_hash = "".join([str(args[a]) for a in sorted(args)])
    exp_hash = hashlib.sha256(str.encode(exp_hash)).hexdigest()
    dataset_name = args["dataset_path"].split("/")[-1]

    os.makedirs(args["results_path"], exist_ok=True)
    header = ",".join([key for key in args.keys()])
    header += ",hash,final_loss,model_size,training_time,train_error,test_error,val_error,n_basis,best_n"

    if not os.path.isfile(f'{args["results_path"]}/results.csv'):
        with open(f'{args["results_path"]}/results.csv', "w") as f:
            f.write(header)

    key = random.PRNGKey(args["key"])
    keys = random.split(key, 3)
    data = jnp.load(args["dataset_path"])
    targets = data["targets"]
    D = targets.ndim - 2
    targets = normalise_field(targets, D)
    perm = random.permutation(keys[0], targets.shape[0])
    targets_encoder = subsample_field(targets[perm], args["J_encoder"], D)
    targets_decoder = subsample_field(targets[perm], args["J_decoder"], D)

    stop_each = min(args["stop_each"], args["N_epoch"])
    N_run = args["N_epoch"] * args["N_train"] // args["N_batch"]
    N_drop = args["N_drop"] * args["N_train"] // args["N_batch"]
    N_stop = stop_each * args["N_train"] // args["N_batch"]

    N_blocks = args["N_blocks"]
    N_features = [targets_encoder.shape[1], args["N_processor"], targets_decoder.shape[1]]
    kernel_size = parse_tuple_or_int(args["kernel_size"])
    subsampling = parse_tuple_or_int(args["subsampling"])
    n_codes = args["n_codes"]
    num_groups = args["num_groups"]

    model = Conv_AE.Conv_AE(
        N_blocks=N_blocks,
        N_features=N_features,
        kernel_size=kernel_size,
        subsampling=subsampling,
        n_codes=n_codes,
        key=keys[1],
        D=D,
        num_groups=num_groups,
        s1=args["s1"],
        s2=args["s2"],
    )

    z_sample = model.encode(targets_encoder[0])
    n_basis = int(np.prod(z_sample.shape))
    model_size = sum(
        jnp.size(x) for x in tree_flatten(model)[0] if eqx.is_array(x)
    )

    learning_rate = optax.exponential_decay(
        args["learning_rate"], N_drop, args["gamma"]
    )
    if args["optim"] == "lion":
        optim = optax.lion(learning_rate=learning_rate)
    else:
        optim = optax.adam(learning_rate=learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    n_chunks = max(1, N_run // N_stop)
    nn = random.choice(
        keys[2], args["N_train"], shape=(n_chunks, N_stop, args["N_batch"])
    )
    val_ind = args["N_train"] + jnp.arange(args["N_val"])
    train_ind = jnp.arange(args["N_train"])
    test_ind = -(1 + jnp.arange(args["N_test"]))
    carry = [model, targets_encoder, targets_decoder, opt_state]
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
        rel_errors = scan(
            compute_errors, [model, targets_encoder, targets_decoder], val_ind
        )[1]
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
    eqx.tree_serialise_leaves(
        f'{args["results_path"]}/model_{exp_hash}.eqx', models[best_n]
    )

    train_rel_errors = jnp.mean(
        scan(compute_errors, [models[best_n], targets_encoder, targets_decoder], train_ind)[1]
    )
    val_rel_errors = jnp.mean(
        scan(compute_errors, [models[best_n], targets_encoder, targets_decoder], val_ind)[1]
    )
    test_rel_errors = jnp.mean(
        scan(compute_errors, [models[best_n], targets_encoder, targets_decoder], test_ind)[1]
    )
    data = "\n" + ",".join([str(args[key]) for key in args.keys()])
    data += f",{exp_hash},{history[-1]},{model_size},{training_times[best_n]},{train_rel_errors},{test_rel_errors},{val_rel_errors},{n_basis},{best_n}"

    with open(f'{args["results_path"]}/results.csv', "a") as f:
        f.write(data)

    jnp.savez(
        f'{args["results_path"]}/metrics_{exp_hash}.npz',
        rel_errors=rel_errors,
        history=history,
        permutation=perm,
    )
