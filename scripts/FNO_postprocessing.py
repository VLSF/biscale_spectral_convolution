import os
import argparse
import pandas as pd
import jax.numpy as jnp
import numpy as np
import FNO_train
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
        targets = FNO_train.normalise_field(targets, D)
        perm = random.permutation(keys[0], targets.shape[0])
        targets = targets[perm]
        
        N_layers = args["N_layers"]
        
        val_ind = args["N_train"] + jnp.arange(args["N_val"])
        train_ind = jnp.arange(args["N_train"])
        test_ind = -(1 + jnp.arange(args["N_test"]))
        
        N_modes = args["N_modes"]
        features = FNO_train.normalise_field(data['features'], D)[perm]
        coordinates = data['coordinates']

        N_features = [coordinates.shape[0] + features.shape[1], args['N_processor'], targets.shape[1]]
        model = FNO.FNO(N_layers, N_features, N_modes, D, keys[1], s1=args['s1'], s2=0, s3=args['s3'])
        model = eqx.tree_deserialise_leaves(f'{args["results_path"]}/model_{args["hash"]}.eqx', model)
        call_model = jit(lambda a, b: model(a, b))
        cost = call_model.trace(features[0], coordinates).lower().compile().cost_analysis()[0]['flops']
        
        train_rel_errors = jnp.mean(scan(FNO_train.compute_errors, [model, features, coordinates, targets], train_ind)[1])
        val_rel_errors = jnp.mean(scan(FNO_train.compute_errors, [model, features, coordinates, targets], val_ind)[1])
        test_rel_errors = jnp.mean(scan(FNO_train.compute_errors, [model, features, coordinates, targets], test_ind)[1])
        
        high_res_eval += f"\n{args['hash']},{jnp.mean(train_rel_errors)},{jnp.mean(test_rel_errors)},{jnp.mean(val_rel_errors)},{cost}"

    with open(f'{Args["postprocessing_results_path"]}/postprocessing_results.csv', "w") as f:
        f.write(high_res_eval)