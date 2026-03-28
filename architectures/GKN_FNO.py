import jax.numpy as jnp
import equinox as eqx

from jax.lax import dot_general, scan
from jax import random, vmap
from jax.nn import gelu

def normalize_conv(A, s1=1.0, s2=1.0):
    A = eqx.tree_at(lambda x: x.weight, A, A.weight * s1)
    try:
        A = eqx.tree_at(lambda x: x.bias, A, A.bias * s2)
    except:
        pass
    return A

def rfft2_truncate(inp, M):
    inp = jnp.fft.rfft2(inp, norm='forward')[:, :, :M]
    inp = jnp.roll(inp, M//2, axis=1)[:, :M]
    return inp

def irfft2_pad(inp, M, s):
    inp = jnp.pad(inp, [(0, 0),] +[(0, s_-inp.shape[i+1]) for i, s_ in enumerate(s)])
    inp = jnp.roll(inp, -M//2, axis=1)
    inp = jnp.fft.irfft2(inp, s=s, norm='forward')
    return inp

def rfft3_truncate(inp, M):
    inp = jnp.fft.rfftn(inp, axes=(1, 2, 3), norm='forward')[:, :, :, :M]
    inp = jnp.roll(inp, M//2, axis=1)[:, :M]
    inp = jnp.roll(inp, M//2, axis=2)[:, :, :M]
    return inp

def irfft3_pad(inp, M, s):
    inp = jnp.pad(inp, [(0, 0),] +[(0, s_-inp.shape[i+1]) for i, s_ in enumerate(s)])
    inp = jnp.roll(inp, -M//2, axis=1)
    inp = jnp.roll(inp, -M//2, axis=2)
    inp = jnp.fft.irfftn(inp, axes=(1, 2, 3), s=s, norm='forward')
    return inp

def get_neighbours(carry, ind, N_max):
    x0, x1, R = carry
    point_a = x0[:, ind].reshape(x0.shape[0], 1)
    distances = jnp.linalg.norm(x1 - point_a, axis=0)
    mask = (distances <= R)
    indices_1 = jnp.arange(x1.shape[1])*mask - jnp.logical_not(mask)
    indices_1 = jnp.sort(indices_1)[-N_max:]
    indices_0 = jnp.array([ind,]*indices_1.shape[0])
    return carry, [indices_0, indices_1]

def get_indices(x0, x1, R):
    D = x0.shape[0]
    x0 = x0.reshape(x0.shape[0], -1)
    x1 = x1.reshape(x1.shape[0], -1)
    h = min(jnp.sort(jnp.sum(x0, axis=0))[1], jnp.sort(jnp.sum(x1, axis=0))[1])
    N_max = int((2*R / (h))**D)
    get_neighbours_ = lambda a, b: get_neighbours(a, b, N_max)
    ind = jnp.arange(x0.shape[1])
    carry = [x0, x1, R]
    _, [ind_0, ind_1] = scan(get_neighbours_, carry, ind) 
    ind_0 = ind_0.reshape(-1,)
    ind_1 = ind_1.reshape(-1,)
    ind_0 = ind_0[ind_1 >= 0]
    ind_1 = ind_1[ind_1 >= 0]
    return ind_0, ind_1

class GKN(eqx.Module):
    convs: list

    def __init__(self, N_features, N_layers, key, s1=1.0, s2=1.0):
        N_in, N_processor = N_features
        n_f = [N_in,] + [N_processor,]*N_layers + [1]
        keys = random.split(key, len(n_f)-1)
        self.convs = [normalize_conv(eqx.nn.Conv(1, n_in, n_out, 1, key=key), s1=s1, s2=s2) for n_in, n_out, key in zip(n_f[:-1], n_f[1:], keys)]

    def __call__(self, u, x0, x1, ind0, ind1):
        s = x1.shape[1:]
        u_ = u.reshape(u.shape[0], -1)
        x0_ = x0.reshape(x0.shape[0], -1)
        x1_ = x1.reshape(x1.shape[0], -1)
        w = jnp.concatenate([u_[:, ind0], u_[:, ind1], x0_[:, ind0], x1_[:, ind0]], axis=0)
        w = self.convs[0](w)
        for c in self.convs[1:]:
            w = gelu(w)
            w = c(w)
        output = jnp.zeros([u_.shape[0], x1_.shape[1]])
        counter = jnp.zeros([u_.shape[0], x1_.shape[1]])
        counter = counter.at[:, ind1].add(0*u_[:, ind0] + 1)
        counter = counter + (counter == 0.0)
        output = output.at[:, ind1].add(w*u_[:, ind0])
        output = output / counter
        output = output.reshape([u_.shape[0],] + list(s))
        return output
        
class custom_Identity(eqx.Module):

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, inp, x0, x1, ind0, ind1):
        return inp

class GKN_FNO(eqx.Module):
    encoder: eqx.Module
    kernel_encoder: eqx.Module
    kernel_decoder: eqx.Module
    decoder: eqx.Module
    convs1: list
    convs2: list
    A: jnp.array

    def __init__(self, N_layers, N_features, N_modes, D, J0, J1, kernel_encoder, GKN_proc, GKN_layers, key, s1=1.0, s2=1.0, s3=1.0):
        # if kernel_encoder == True, resolution changes before encoder, otherwise it changes after decoder
        n_in, n_processor, n_out = N_features
        keys = random.split(key, 4 + 2*N_layers)
        self.encoder = normalize_conv(eqx.nn.Conv(D, n_in, n_processor, 1, key=keys[-1]), s1=s1, s2=s2)
        self.decoder = normalize_conv(eqx.nn.Conv(D, n_processor, n_out, 1, key=keys[-2]), s1=s1, s2=s2)
        self.convs1 = [normalize_conv(eqx.nn.Conv(D, n_processor, n_processor, 1, key=key), s1=s1, s2=s2) for key in keys[:N_layers]]
        self.convs2 = [normalize_conv(eqx.nn.Conv(D, n_processor, n_processor, 1, key=key), s1=s1, s2=s2) for key in keys[N_layers:2*N_layers]]
        self.A = random.normal(keys[-3], [N_layers, n_processor, n_processor] + [N_modes,]*D, dtype=jnp.complex64) * s3
        if kernel_encoder:
            N_features = [2*n_in + 2*D, GKN_proc]
            self.kernel_decoder = custom_Identity()
            self.kernel_encoder = GKN(N_features, GKN_layers, keys[-4], s1=s1, s2=s2)
        else:
            N_features = [2*n_out + 2*D, GKN_proc]
            self.kernel_encoder = custom_Identity()
            self.kernel_decoder = GKN(N_features, GKN_layers, keys[-4], s1=s1, s2=s2)
    
    def __call__(self, u, x0, x1, ind0, ind1):
        u = jnp.concatenate([x0, u], 0)
        u = self.kernel_encoder(u, x0, x1, ind0, ind1)
        u = self.encoder(u)
        for i in range(self.A.shape[0]):
            u += gelu(self.convs2[i](gelu(self.convs1[i](self.spectral_conv(u, self.A[i])))))
        u = self.decoder(u)
        u = self.kernel_decoder(u, x0, x1, ind0, ind1)
        return u

    def spectral_conv(self, v, A):
        D = v.ndim - 1
        if D == 1:
            u = self.spectral_conv_d1(v, A)
        elif D == 2:
            u = self.spectral_conv_d2(v, A)
        else:
            u = self.spectral_conv_d3(v, A)
        return u

    def spectral_conv_d1(self, inp, A):
        coeff = jnp.fft.rfft(inp, norm='forward', axis=1)[:, :A.shape[2]]
        coeff = dot_general(A, coeff, (((1,), (0,)), ((2,), (1,)))).T
        out = jnp.fft.irfft(coeff, norm='forward', n=inp.shape[1], axis=1)
        return out
    
    def spectral_conv_d2(self, inp, A):
        coeff = rfft2_truncate(inp, A.shape[2])
        coeff = jnp.moveaxis(dot_general(A, coeff, (((1,), (0,)), ((2, 3), (1, 2)))), 2, 0)
        out = irfft2_pad(coeff, A.shape[2], inp.shape[1:])
        return out

    def spectral_conv_d3(self, inp, A):
        coeff = rfft3_truncate(inp, A.shape[-1])
        coeff = jnp.moveaxis(dot_general(A, coeff, (((1,), (0,)), ((2, 3, 4), (1, 2, 3)))), 3, 0)
        out = irfft3_pad(coeff, A.shape[-1], inp.shape[1:])
        return out