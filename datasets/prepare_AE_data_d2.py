import jax.numpy as jnp

if __name__ == "__main__":
    data = jnp.load("diffusion_d2_simplified.npz")
    for key in data.keys():
        print(key, data[key].shape)
    
    data = {
        'targets': data['targets'],
        'coordinates': data['coordinates']
    }
    jnp.savez("diffusion_d2_AE.npz", **data)
    
    data = jnp.load("KdV_d2.npz")
    for key in data.keys():
        print(key, data[key].shape)
    
    data = {
        'targets': data['targets'],
        'coordinates': data['coordinates']
    }
    jnp.savez("KdV_d2_AE.npz", **data)
    
    data = jnp.load("Burgers_dataset_d2_2.npz")
    for key in data.keys():
        print(key, data[key].shape)
    
    data = {
        'targets': data['targets'],
        'coordinates': data['coordinates']
    }
    jnp.savez("Burgers_dataset_d2_2_AE.npz", **data)
    
    data = jnp.load("Burgers_dataset_d2_1.npz")
    for key in data.keys():
        print(key, data[key].shape)
    
    data = {
        'targets': data['targets'],
        'coordinates': data['coordinates']
    }
    jnp.savez("Burgers_dataset_d2_1_AE.npz", **data)