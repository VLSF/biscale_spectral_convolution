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
    
    J_min = Data['J_a'].min()
    J_max = Data['J_a'].max()
    resolution = jnp.arange(J_min, J_max+1)
    
    header = "hash,resolution,train_error,test_error,val_error,flops"
    with open(f'{Args["postprocessing_results_path"]}/upsampling_results.csv', "w") as f:
            f.write(header)
            
    Datasets = Data['dataset_path'].unique()
    for dataset_ in Datasets:
        Data_ = Data[Data['dataset_path'] == dataset_]
        data = jnp.load(dataset_)
        targets_ = data['targets']
        features_ = data['features']
        coordinates_ = data['coordinates']
        D = targets_.ndim - 2
        
        for i in range(len(Data_)):
            args = Data_.iloc[i]
            resolution_ = resolution if args['N_modes'] == 16 else resolution[:-1]
            if args['J_a'] != resolution_[-1]:
                pass
            else:   
                key = random.PRNGKey(args["key"])
                keys = random.split(key, 3)
                perm = random.permutation(keys[0], targets_.shape[0])
                
                N_layers = args["N_layers"]
                N_features_a = [D + (0 if args['features_input'] == 'b' else features_.shape[1]), args['N_processor_a'], targets_.shape[1]]
                N_features_b = [D + (0 if args['features_input'] == 'a' else features_.shape[1]), args['N_processor_b']]
                
                val_ind = args["N_train"] + jnp.arange(args["N_val"])
                train_ind = jnp.arange(args["N_train"])
                test_ind = -(1 + jnp.arange(args["N_test"]))
                
                N_modes = args["N_modes"]
                model = BiFNO.BiFNO(N_layers, N_features_a, N_features_b, N_modes, D, keys[1], s1=args['s1'], s2=0, s3=args['s3'])
                model = eqx.tree_deserialise_leaves(f'{args["results_path"]}/model_{args["hash"]}.eqx', model)
        
                call_model = jit(lambda a, b, c, d: model(a, b, c, d)[0])
            
                for j in resolution_:
                    targets = BiFNO_train.normalise_field(BiFNO_train.subsample_field(targets_, j, D), D)
                    perm = random.permutation(keys[0], targets.shape[0])
                    targets = targets[perm]
                    
                    if args['features_input'] == 'a':
                        features_a = BiFNO_train.normalise_field(BiFNO_train.subsample_field(features_, j, D), D)[perm]
                        features_b = None
                    else:
                        features_a = None
                        features_b = BiFNO_train.normalise_field(BiFNO_train.subsample_field(features_, args['J_b'], D), D)[perm]

                    coordinates_a = BiFNO_train.subsample_field(np.expand_dims(coordinates_, 0), j, D)[0]
                    if args['Nx_b'] != 0:
                        coordinates_b = BiFNO_train.get_coordinates(args['Nx_b'], D)
                    else:
                        coordinates_b = BiFNO_train.subsample_field(np.expand_dims(coordinates_, 0), args['J_b'], D)[0]
                    
                    if features_a is None:
                        try:
                            cost = call_model.trace(features_a, coordinates_a, features_b[0], coordinates_b).lower().compile().cost_analysis()[0]['flops']
                        except:
                            cost = call_model.trace(features_a, coordinates_a, features_b[0], coordinates_b).lower().compile().cost_analysis()['flops']
                    elif features_b is None:
                        try:
                            cost = call_model.trace(features_a[0], coordinates_a, features_b, coordinates_b).lower().compile().cost_analysis()[0]['flops']
                        except:
                            cost = call_model.trace(features_a[0], coordinates_a, features_b, coordinates_b).lower().compile().cost_analysis()['flops']
                    else:
                        try:
                            cost = call_model.trace(features_a[0], coordinates_a, features_b[0], coordinates_b).lower().compile().cost_analysis()[0]['flops']
                        except:
                            cost = call_model.trace(features_a[0], coordinates_a, features_b[0], coordinates_b).lower().compile().cost_analysis()['flops']
                    train_rel_errors = jnp.mean(scan(BiFNO_train.compute_errors, [model, features_a, coordinates_a, features_b, coordinates_b, targets], train_ind)[1])
                    val_rel_errors = jnp.mean(scan(BiFNO_train.compute_errors, [model, features_a, coordinates_a, features_b, coordinates_b, targets], val_ind)[1])
                    test_rel_errors = jnp.mean(scan(BiFNO_train.compute_errors, [model, features_a, coordinates_a, features_b, coordinates_b, targets], test_ind)[1])
                    with open(f'{Args["postprocessing_results_path"]}/upsampling_results.csv', "a") as f:
                        res_ = f"\n{args['hash']},{j},{jnp.mean(train_rel_errors)},{jnp.mean(test_rel_errors)},{jnp.mean(val_rel_errors)},{cost}"
                        f.write(res_)