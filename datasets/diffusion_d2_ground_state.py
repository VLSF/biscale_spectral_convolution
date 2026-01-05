import sys
import jax.numpy as jnp
import numpy as np

from jax import random

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, eigsh, LinearOperator, splu

def get_weights(N, h, decay, power):
    k = 2*np.pi*np.fft.fftfreq(N, d=h)
    K_x, K_y = np.meshgrid(k, k, indexing="ij")
    w = 1 / (1 + decay*(K_x**2 + K_y**2))**power
    return w

def random_series(w, c_a):
    res = np.real(np.fft.ifftn(c_a * w, norm="ortho"))
    res = res - np.mean(res)
    return res

def a_rand(X, Y, params, key):
    w, alpha, beta, slope, A = params
    c_a = np.array(random.normal(key, X.shape, dtype=jnp.complex64)) * A
    a = alpha + (beta-alpha)*(jnp.tanh(slope*random_series(w, c_a)) + 1)/2
    return a

def get_discretization_data(N):
    N_x = N_y = N

    x, y = np.linspace(0, 1, 2*N_x+3)[1:-1], np.linspace(0, 1, 2*N_y+3)[1:-1]
    X, Y = np.meshgrid(x, y, indexing="ij")

    h_x, h_y = x[1] - x[0], y[1] - y[0]
    x_half, y_half = np.linspace(h_x/2, 1-h_x/2, N_x+1), np.linspace(h_y/2, 1-h_y/2, N_y+1)
    X_half, _ = np.meshgrid(x_half, y, indexing="ij")
    _, Y_half = np.meshgrid(x, y_half, indexing="ij")

    i, j = np.arange(N_x, dtype=int), np.arange(N_y, dtype=int)
    I, J = np.meshgrid(i, j, indexing="ij")
    lex = lambda i, j, N_x: (j + i*N_x)
    diag = lex(I, J, N_x)
    i_m = lex(I-1, J, N_x)
    i_p = lex(I+1, J, N_x)
    j_m = lex(I, J-1, N_x)
    j_p = lex(I, J+1, N_x)

    mask_i_m = (I-1) >= 0
    mask_i_p = (I+1) < N_x
    mask_j_m = (J-1) >= 0
    mask_j_p = (J+1) < N_y

    rows = np.concatenate([diag.reshape(-1,), diag[mask_i_m], diag[mask_i_p], diag[mask_j_m], diag[mask_j_p]])
    cols = np.concatenate([diag.reshape(-1,), i_m[mask_i_m], i_p[mask_i_p], j_m[mask_j_m], j_p[mask_j_p]])
    indices = np.stack([cols, rows], 1)
    return mask_i_m, mask_i_p, mask_j_m, mask_j_p, X_half, Y_half, X, Y, indices

def get_matrix(discretization_data, params, key):
    mask_i_m, mask_i_p, mask_j_m, mask_j_p, X_half, Y_half, X, Y, indices = discretization_data
    N = (X.shape[0] - 1) // 2
    a = a_rand(X, Y, params, key)
    a1_p = a[::2][1:, 1::2]
    a1_m = a[::2][:-1, 1::2]
    a2_p = a[:, ::2][1::2, 1:]
    a2_m = a[:, ::2][1::2, :-1]
    a1 = a[1::2, 1::2]
    data = np.concatenate([
        (a1_p + a1_m + a2_p + a2_m).reshape(-1,),
        -a1_m[mask_i_m],
        -a1_p[mask_i_p],
        -a2_m[mask_j_m],
        -a2_p[mask_j_p],
    ])

    A = coo_matrix((data, (indices[:, 0], indices[:, 1])), shape=(N**2, N**2)).tocsc()
    return A, a1

def generate_dataset(key, N_samples):
    N_x = 256
    h = 0.5/(N_x+1)
    alpha = 1
    beta = 30
    slope = 2
    A = 1000
    decay = 1e-2
    power = 2.0
    N_eig = 1
    w = get_weights(2*N_x+1, h, decay, power)
    params = [w, alpha, beta, slope, A]
    discretization_data = get_discretization_data(N_x)
    keys = random.split(key, N_samples)
    data = {
        "features": [],
        "targets": [],
        "coordinates": jnp.stack([discretization_data[-3][1::2, 1::2], discretization_data[-2][1::2, 1::2]])
    }
    
    for key in keys:
        matrix, a = get_matrix(discretization_data, params, key)
        A_ = splu(matrix)
        L_ = LinearOperator(matrix.shape, matvec=lambda x: A_.solve(x), dtype=matrix.dtype)
        vals, gs = eigsh(L_, k=N_eig)
        gs = np.array(gs).reshape(N_x, N_x)
        sign = gs.reshape(-1,)
        sign = np.sign(sign[np.argmax(np.abs(sign))])
        gs = gs / sign
        data["features"].append(a)
        data["targets"].append(gs)
    data["features"] = np.array(data["features"]).reshape(N_samples, 1, N_x, N_x)
    data["targets"] = np.array(data["targets"]).reshape(N_samples, 1, N_x, N_x)
    return data

if __name__ == "__main__":
    dataset_path = sys.argv[1]
    key = random.PRNGKey(3359)
    N_samples = 1000
    data = generate_dataset(key, N_samples)
    jnp.savez(dataset_path, **data)