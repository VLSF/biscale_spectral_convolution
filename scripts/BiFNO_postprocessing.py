import os
import argparse
import pandas as pd
import jax.numpy as jnp
import numpy as np
import BiFNO_train
import equinox as eqx

from architectures import FNO, BiFNO
from jax import random, jit
from jax.lax import scan

def get_argparser():
    parser = argparse.ArgumentParser()
    args = {
       "-results_path": {
            "help": "absolute path to folder where results are stored"
        },
       "-postprocessing_results_path": {
            "help": "absolute path to folder where postprocessing results will be stored"
        }
    }
    for key in args:
        parser.add_argument(key, **args[key])

    return parser

if __name__ == "__main__":
    parser = get_argparser()
    Args = vars(parser.parse_args())
    Data = pd.read_csv(f"{Args['results_path']}/results.csv")
    high_res_eval = "hash,train_error,test_error,val_error,flops"
    for i in range(len(Data)):
        args = Data.iloc[i]

        key = random.PRNGKey(args["key"])
        keys = random.split(key, 3)
        data = jnp.load(args['dataset_path'])
        targets = data['targets']
        D = targets.ndim - 2
        targets = BiFNO_train.normalise_field(targets, D)
        perm = random.permutation(keys[0], targets.shape[0])
        targets = targets[perm]
        
        if args['features_input'] == 'a':
            features_a = BiFNO_train.normalise_field(data['features'], D)[perm]
            features_b = None
        else:
            features_a = None
            features_b = BiFNO_train.normalise_field(BiFNO_train.subsample_field(data['features'], args['J_b'], D), D)[perm]
        
        coordinates_a = data['coordinates']
        if args['Nx_b'] != 0:
            coordinates_b = BiFNO_train.get_coordinates(args['Nx_b'], D)
        else:
            coordinates_b = BiFNO_train.subsample_field(np.expand_dims(data['coordinates'], 0), args['J_b'], D)[0]
        
        N_layers = args["N_layers"]
        N_features_a = [coordinates_a.shape[0] + (0 if features_a is None else features_a.shape[1]), args['N_processor_a'], targets.shape[1]]
        N_features_b = [coordinates_b.shape[0] + (0 if features_b is None else features_b.shape[1]), args['N_processor_b']]
        
        val_ind = args["N_train"] + jnp.arange(args["N_val"])
        train_ind = jnp.arange(args["N_train"])
        test_ind = -(1 + jnp.arange(args["N_test"]))
        
        N_modes = args["N_modes"]
        model = BiFNO.BiFNO(N_layers, N_features_a, N_features_b, N_modes, D, keys[1], s1=args['s1'], s2=0, s3=args['s3'])
        model = eqx.tree_deserialise_leaves(f'{args["results_path"]}/model_{args["hash"]}.eqx', model)

        call_model = jit(lambda a, b, c, d: model(a, b, c, d)[0])
        if features_a is None:
            cost = call_model.trace(features_a, coordinates_a, features_b[0], coordinates_b).lower().compile().cost_analysis()[0]['flops']
        elif features_b is None:
            cost = call_model.trace(features_a[0], coordinates_a, features_b, coordinates_b).lower().compile().cost_analysis()[0]['flops']
        else:
            cost = call_model.trace(features_a[0], coordinates_a, features_b[0], coordinates_b).lower().compile().cost_analysis()[0]['flops']
        train_rel_errors = jnp.mean(scan(BiFNO_train.compute_errors, [model, features_a, coordinates_a, features_b, coordinates_b, targets], train_ind)[1])
        val_rel_errors = jnp.mean(scan(BiFNO_train.compute_errors, [model, features_a, coordinates_a, features_b, coordinates_b, targets], val_ind)[1])
        test_rel_errors = jnp.mean(scan(BiFNO_train.compute_errors, [model, features_a, coordinates_a, features_b, coordinates_b, targets], test_ind)[1])
        high_res_eval += f"\n{args['hash']},{jnp.mean(train_rel_errors)},{jnp.mean(test_rel_errors)},{jnp.mean(val_rel_errors)},{cost}"

    with open(f'{Args["postprocessing_results_path"]}/postprocessing_results.csv', "w") as f:
        f.write(high_res_eval)