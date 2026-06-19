import jax.numpy as jnp

if __name__ == "__main__":
    data = jnp.load("KdV.npz")
    for key in data.keys():
        print(key, data[key].shape)
    
    x = data['x']
    t = data['t']
    coordinates = jnp.expand_dims(x, 0)
    targets = data['solutions'][:, -1:, :]
    
    data = {
        'targets': targets,
        'coordinates': coordinates
    }
    jnp.savez("KdV_AE.npz", **data)

    data = jnp.load("Burgers_dataset_d2_2.npz")
    for key in data.keys():
        print(key, data[key].shape)
    
    coordinates = data['coordinates']
    coordinates = jnp.expand_dims(coordinates[1, 0, :], 0)
    targets = data['targets'][:, 0, -1:]
    
    data = {
        'targets': targets,
        'coordinates': coordinates
    }
    jnp.savez("Burgers_AE.npz", **data)

    data = jnp.load("diffusion_dataset.npz")
    for key in data.keys():
        print(key, data[key].shape)
    
    coordinates = data['coordinates']
    targets = data['targets']
    
    data = {
        'targets': targets,
        'coordinates': coordinates
    }
    jnp.savez("Diffusion_AE.npz", **data)

    data = jnp.load("Cahn_Hilliard_d1.npz")
    for key in data.keys():
        print(key, data[key].shape)
    
    coordinates = data['coordinates']
    targets = data['targets']
    
    data = {
        'targets': targets,
        'coordinates': coordinates
    }
    jnp.savez("Cahn_Hilliard_AE.npz", **data)