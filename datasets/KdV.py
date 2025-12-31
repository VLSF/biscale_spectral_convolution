# Zabusky-Kruskal scheme for KdV, https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.15.240
# See also https://scicomp.stackexchange.com/questions/33414/confusion-about-zabusky-and-kruskals-stepper-for-the-kdv-equation

import sys
import jax.numpy as jnp
import time

from jax.lax import scan
from jax import random, vmap

def F(u, delta, dx):
    res = -(jnp.roll(u, 1) + jnp.roll(u, -1) + u) * (jnp.roll(u, -1) - jnp.roll(u, 1))/(6*dx)
    res = res - (jnp.roll(u, -2) - 2*jnp.roll(u, -1) + 2*jnp.roll(u, 1) - jnp.roll(u, 2)) * delta**2 / (2*dx**3)
    return res

def integration_step(carry, t):
    u_current, u_prev, dt, dx, delta = carry
    u_next = u_prev + 2*dt*F(u_current, delta, dx)
    return [u_next, u_current, dt, dx, delta], u_next

def integration_step_vmap(carry, t):
    u_current, u_prev, dt, dx, delta = carry
    u_next = u_prev + 2*dt*vmap(F, in_axes=(0, None, None))(u_current, delta, dx)
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

def generate_dataset(key, N_samples, save_logs_to):
    N_batch = 25
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

    keys = random.split(key, (N_samples//N_batch, N_batch))
    data = {
        "solutions": [],
        "x": x,
        "t": t[::2**(11)]
    }

    j = 1
    for key in keys:
        start = time.time()
        u_prev = vmap(get_initial_conditions, in_axes=(0, None, None))(key, w, N_x)
        u_current = vmap(RK4_init, in_axes=(0, None, None, None))(u_prev, dt, dx, delta)
        carry = [u_current, u_prev, dt, dx, delta]
        _, sol = scan(integration_step_vmap, carry, t)
        sol = jnp.concatenate([u_prev.reshape(1, N_batch, -1), sol[(2**11-2)::2**11]], axis=0)
        sol = jnp.transpose(sol, (1, 0, 2))
        data["solutions"].append(sol)
        stop = time.time()
        elapsed_time = stop - start
        with open(save_logs_to, "a+") as f:
            f.write(f"\nround {j}/{N_samples//N_batch}, elapsed_time {elapsed_time}")
        j += 1
            
        
    data["solutions"] = jnp.concatenate(data["solutions"], axis=0)
    return data

if __name__ == "__main__":
    dataset_path = sys.argv[1]
    save_logs_to = sys.argv[2]
    key = random.PRNGKey(444111)
    N_samples = 25
    data = generate_dataset(key, N_samples, save_logs_to)
    jnp.savez(dataset_path, **data)