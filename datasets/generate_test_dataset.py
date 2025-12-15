import sys
import jax.numpy as jnp

from jax import random
from jax.lax import dot_general

if __name__ == "__main__":
    key = random.PRNGKey(33)
    D = int(sys.argv[1])
    n = int(sys.argv[2])
    s = [-1,] + [1,]*D
    x = jnp.linspace(0, 1, n)
    N_samples = 1200
    coordinates = jnp.stack(jnp.meshgrid(*[x for _ in range(D)]))
    w = 1 + jnp.abs(random.normal(key, (N_samples, D)))*2*jnp.pi
    features = jnp.expand_dims(jnp.cos(dot_general(w, coordinates, (((1,), (0,)), ((), ())))), 1)
    targets = jnp.expand_dims(jnp.sum(w, axis=1).reshape(*s) * jnp.sin(dot_general(w, coordinates, (((1,), (0,)), ((), ())))), 1)
    data = {
        "features": features,
        "targets": targets,
        "coordinates": coordinates
    }
    jnp.savez(f"test_dataset_{D}_{n}.npz", **data)