import jax.numpy as jnp

from jax.lax import scan
from jax import grad, vmap, config, random

def get_integration_data(N, h, dt, D, gamma):
    k = 2*jnp.pi*1j*jnp.fft.fftfreq(N, d = h)
    k2 = k**2
    alpha = dt * k2 
    beta = 1 + D * alpha + gamma * D * alpha**2 / dt
    return alpha, beta

def get_random_field(key, s):
    c = random.normal(key, s, dtype=jnp.complex64)
    return jnp.real(jnp.fft.ifftn(c))

def integrate_scan(carry, ind):
    c, alpha, beta = carry
    s = c.shape
    c = jnp.real(jnp.fft.ifftn((jnp.fft.fftn(c, s=s) + alpha * jnp.fft.fftn(c**3, s=s)) / beta, s=s))
    return [c, alpha, beta], c

def get_sample(key, x, alpha, beta):
    c = get_random_field(key, x.shape)
    
    N_t = 500
    ind = jnp.arange(N_t)
    carry = [c, alpha, beta]
    carry, trajectory = scan(integrate_scan, carry, ind)
    trajectory = jnp.concatenate([jnp.expand_dims(c, 0), trajectory], 0)
    
    feature = trajectory[50]
    target = trajectory[-1]
    return feature, target

if __name__ == "__main__":
    N = 512
    x = jnp.linspace(0, 1, N+1)[:-1]
    h = x[1] - x[0]
    D = 1e-4
    gamma = 0.5
    dt = 1e-6
    key = random.PRNGKey(663)
    alpha, beta = get_integration_data(N, h, dt, D, gamma)
    N_samples = 1000
    keys = random.split(key, N_samples)
    features, targets = vmap(get_sample, in_axes=(0, None, None, None))(keys, x, alpha, beta)
    
    data = {
        "features": jnp.expand_dims(features, 1),
        "targets": jnp.expand_dims(targets, 1),
        "coordinates": jnp.expand_dims(x, 0)
    }
    
    jnp.savez("Cahn_Hilliard_d1.npz", **data)