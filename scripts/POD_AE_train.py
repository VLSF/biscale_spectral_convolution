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
from architectures import POD_AE


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


def compute_POD_basis(train_fields):
    """Compute orthonormal spatial POD basis on GPU using SVD without truncation."""
    train_flat = train_fields.reshape(train_fields.shape[0], -1)
    # SVD: train_flat = U S V
    # V has shape (N_train, M)
    V = jnp.linalg.svd(train_flat, full_matrices=False)[2]
    return V


def project_onto_POD(fields, V):
    """Project physical fields onto POD basis to get coefficients without truncation."""
    fields_flat = fields.reshape(fields.shape[0], -1)
    coeffs = fields_flat @ V.T
    return coeffs


def compute_errors(carry, ind, targets_coeff, targets_flat, V):
    model = carry[0]
    c = targets_coeff[ind]
    z = model.encode(c)
    c_pred = model.decode(z)
    u_pred_flat = c_pred @ V
    target_flat = targets_flat[ind]
    error = jnp.linalg.norm(u_pred_flat - target_flat) / jnp.linalg.norm(target_flat)
    return carry, error


def l2_loss(model, coeff, target_flat, V):
    c_pred = model(coeff)
    u_pred_flat = c_pred @ V
    error = jnp.sum((u_pred_flat - target_flat) ** 2)
    return error


def batch_l2_loss(model, batch_coeffs, batch_targets_flat, V):
    res = vmap(l2_loss, in_axes=[None, 0, 0, None])(
        model, batch_coeffs, batch_targets_flat, V
    )
    return jnp.mean(res)


compute_loss_and_grads = eqx.filter_value_and_grad(batch_l2_loss)


def make_step_scan(carry, n, optim, V):
    model, targets_coeff, targets_flat, opt_state = carry
    loss, grads = compute_loss_and_grads(
        model, targets_coeff[n], targets_flat[n], V
    )
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, targets_coeff, targets_flat, opt_state], loss


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
            "default": 1001,
            "type": int,
            "help": "number of updates of the model weights = N_epoch * N_train // N_batch",
        },
        "-stop_each": {
            "default": 50,
            "type": int,
            "help": "stop each N_epoch to evaluate the model and make checkpoint",
        },
        "-d_latent": {
            "default": 256,
            "type": int,
            "help": "bottleneck latent dimension (e.g. 64, 128, 256, 512, 1024, 4096)",
        },
        "-d_hidden": {
            "default": 768,
            "type": int,
            "help": "hidden feature dimension of the ResMLP blocks",
        },
        "-N_blocks": {
            "default": 3,
            "type": int,
            "help": "number of ResMLP blocks in encoder and decoder",
        },
        "-key": {
            "default": 14,
            "type": int,
            "help": "PRNGKey that seeds all randomness in the code",
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

    os.makedirs(args["results_path"], exist_ok=True)
    header = ",".join([key for key in args.keys()])
    header += ",hash,final_loss,model_size,training_time,train_error,test_error,val_error,d_latent,best_n"

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
    targets = targets[perm]

    # Stage 1: Compute POD basis on GPU from training set without truncation
    train_targets = targets[: args["N_train"]]
    V = compute_POD_basis(train_targets)
    n_basis = V.shape[0]

    # Stage 2: Project all targets onto the POD basis without truncation
    targets_coeff = project_onto_POD(targets, V)
    targets_flat = targets.reshape(targets.shape[0], -1)

    stop_each = min(args["stop_each"], args["N_epoch"])
    N_run = args["N_epoch"] * args["N_train"] // args["N_batch"]
    N_drop = args["N_drop"] * args["N_train"] // args["N_batch"]
    N_stop = stop_each * args["N_train"] // args["N_batch"]

    model = POD_AE.POD_AE(
        n_basis=n_basis,
        d_hidden=args["d_hidden"],
        n_blocks=args["N_blocks"],
        d_latent=args["d_latent"],
        key=keys[1],
    )

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

    carry = [model, targets_coeff, targets_flat, opt_state]
    make_step_scan_ = lambda a, b: make_step_scan(a, b, optim, V)

    compute_errors_ = lambda c, ind: compute_errors(
        c, ind, targets_coeff, targets_flat, V
    )

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

        rel_errors = scan(compute_errors_, [model], val_ind)[1]
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
        scan(compute_errors_, [models[best_n]], train_ind)[1]
    )
    val_rel_errors = jnp.mean(
        scan(compute_errors_, [models[best_n]], val_ind)[1]
    )
    test_rel_errors = jnp.mean(
        scan(compute_errors_, [models[best_n]], test_ind)[1]
    )

    data = "\n" + ",".join([str(args[key]) for key in args.keys()])
    data += f",{exp_hash},{history[-1]},{model_size},{training_times[best_n]},{train_rel_errors},{test_rel_errors},{val_rel_errors},{args['d_latent']},{best_n}"

    with open(f'{args["results_path"]}/results.csv', "a") as f:
        f.write(data)

    jnp.savez(
        f'{args["results_path"]}/metrics_{exp_hash}.npz',
        rel_errors=rel_errors,
        history=history,
        permutation=perm,
        V=V,
    )
