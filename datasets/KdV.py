# Zabusky-Kruskal scheme for KdV, https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.15.240
# See also https://scicomp.stackexchange.com/questions/33414/confusion-about-zabusky-and-kruskals-stepper-for-the-kdv-equation

import sys
import jax.numpy as jnp

from jax.lax import scan
from jax import random

def F(u, delta, dx):
    res = -(jnp.roll(u, 1) + jnp.roll(u, -1) + u) * (jnp.roll(u, -1) - jnp.roll(u, 1))/(6*dx)
    res = res - (jnp.roll(u, -2) - 2*jnp.roll(u, -1) + 2*jnp.roll(u, 1) - jnp.roll(u, 2)) * delta**2 / (2*dx**3)
    return res

def integration_step(carry, t):
    u_current, u_prev, dt, dx, delta = carry
    u_next = u_prev + 2*dt*F(u_current, delta, dx)
    return [u_next, u_current, dt, dx, delta], u_next

def RK4_init(u, dt, dx, delta):
    k1 = F(u, delta, dx)
    k2 = F(u + k1*dt/2, delta, dx)
    k3 = F(u + k2*dt/2, delta, dx)
    k4 = F(u + k3*dt, delta, dx)
    return u + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

def get_weights(N, n, alpha):
    w = jnp.fft.rfftfreq(N)
    w = 1 / (1 + alpha*(2*jnp.pi*w)**2)**n
    return w

def get_random_function(key, w, N):
    c = random.normal(key, (w.shape[0],), dtype=jnp.complex64) * w
    f = jnp.fft.irfft(c, n=N)
    return f

def get_initial_conditions(key, w, N):
    f = get_random_function(key, w, N)
    f = f / jnp.linalg.norm(f, ord=jnp.inf)
    return f

def generate_dataset(key, N_samples):
    N_x = 512
    N_t = 2**20
    x = jnp.linspace(0, 1, N_x+1)[:-1]
    dx = x[1] - x[0]
    delta = 0.022
    dt = 0.05 * dx**3 / delta**2
    t = jnp.arange(0, N_t - 2)*dt
    
    n = 4
    alpha = 200
    w = get_weights(N_x, n, alpha)

    keys = random.split(key, N_samples)
    data = {
        "solutions": [],
        "x": x,
        "t": t[::2**(11)]
    }
    for key in keys:
        u_prev = get_initial_conditions(key, w, N_x)
        u_current = RK4_init(u_prev, dt, dx, delta)
        carry = [u_current, u_prev, dt, dx, delta]
        _, sol = scan(integration_step, carry, t)
        sol = jnp.concatenate([u_prev.reshape(1, -1), u_current.reshape(1, -1), sol], axis=0)
        data["solutions"].append(sol[::2**(11)])
        
    data["solutions"] = jnp.array(data["solutions"])
    return data

if __name__ == "__main__":
    dataset_path = sys.argv[1]
    key = random.PRNGKey(444111)
    N_samples = 1000
    data = generate_dataset(key, N_samples)
    jnp.savez(dataset_path, **data)