import jax.numpy as jnp
import equinox as eqx
import jax

def normalize_conv(A, s1=1.0, s2=1.0):
    A = eqx.tree_at(lambda x: x.weight, A, A.weight * s1)
    try:
        A = eqx.tree_at(lambda x: x.bias, A, A.bias * s2)
    except:
        pass
    return A

def rfft1_truncate(inp, M):
    inp = jnp.fft.rfft(inp, norm='forward')[:, :M]
    return inp

def irfft1_pad(inp, M, s):
    inp = jnp.pad(inp, [(0, 0),] +[(0, s_-inp.shape[i+1]) for i, s_ in enumerate(s)])
    inp = jnp.fft.irfft(inp, n=s[0], norm='forward')
    return inp

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

class AAO_UNet(eqx.Module):
    encoders: list
    decoders: list
    convs1: list
    convs2: list
    A: jnp.array

    def __init__(self, N_layers, N_features, N_processors, N_targets, N_modes, D, key, s1=1.0, s2=1.0, s3=1.0):
        N_p = sum(N_processors)
        N_levels = len(N_processors)

        keys = jax.random.split(key, N_levels+1)
        self.encoders = jax.tree.map(
            lambda key, pr: normalize_conv(eqx.nn.Conv(D, N_features, pr, 1, key=key), s1=s1, s2=s2),
            [*keys[:-1]],
            N_processors
        )
        keys = jax.random.split(keys[-1], N_levels+1)
        self.decoders = jax.tree.map(
            lambda key, pr: normalize_conv(eqx.nn.Conv(D, pr, N_targets, 1, key=key), s1=s1, s2=s2),
            [*keys[:-1]],
            N_processors
        )
        keys = jax.random.split(keys[-1], N_layers+1)
        get_conv = lambda key, pr: normalize_conv(eqx.nn.Conv(D, pr, pr, 1, key=key), s1=s1, s2=s2)
        self.convs1 = [jax.tree.map(get_conv, [*jax.random.split(key, N_levels)], N_processors) for key in keys[:-1]]
        keys = jax.random.split(keys[-1], N_layers+1)
        self.convs2 = [jax.tree.map(get_conv, [*jax.random.split(key, N_levels)], N_processors) for key in keys[:-1]]
        self.A = jax.random.normal(keys[-1], [N_layers, N_p, N_p] + [N_modes,]*D, dtype=jnp.complex64) * s3

    def __call__(self, u, x):
        f = jnp.concatenate([u, x], 0)
        f = self.get_multiscale_features(f)
        f = jax.tree.map(lambda a, b: b(a), f, self.encoders)
        for i in range(self.A.shape[0]):
            f = self.multiscale_spectral_conv(f, self.A[i], x.ndim-1)
            f = jax.tree.map(
                lambda a, b, c: a + jax.nn.gelu(c(jax.nn.gelu(b(a)))),
                f,
                self.convs1[i],
                self.convs2[i],
            )
        f = jax.tree.map(lambda a, b: b(a), f, self.decoders)
        return f

    def multiscale_spectral_conv(self, inp, W, d):
        if d == 1:
            forward, backward = rfft1_truncate, irfft1_pad
            batch_W, batch_coeff = [2,], [1,]
        elif d == 2:
            forward, backward = rfft2_truncate, irfft2_pad
            batch_W, batch_coeff = [2, 3], [1, 2]
        else:
            forward, backward = rfft3_truncate, irfft3_pad
            batch_W, batch_coeff = [2, 3, 4], [1, 2, 3]
        shapes = [0,]
        for s in jax.tree.map(lambda a: a.shape[0], inp):
            shapes.append(shapes[-1] + s)
        shapes_x = jax.tree.map(lambda a: a.shape[1:], inp)

        coeff = jnp.concatenate(jax.tree.map(lambda a: forward(a, W.shape[2]), inp), axis=0)
        
        coeff = jnp.moveaxis(jax.lax.dot_general(W, coeff, ((1, 0), (batch_W, batch_coeff))), d, 0)
        coeff = [coeff[p1:p2] for p1, p2 in zip(shapes[:-1], shapes[1:])]
        out = jax.tree.map(
            lambda a, Nx: backward(a, W.shape[2], Nx),
            coeff,
            shapes_x
        )
        return out

    def get_multiscale_features(self, features):
        levels = [*range(len(self.encoders))]
        if features.ndim - 1 == 1:
            mapper = lambda l: features[:, ::2**l]
        elif features.ndim - 1 == 2:
            mapper = lambda l: features[:, ::2**l, ::2**l]
        else:
            mapper = lambda l: features[:, ::2**l, ::2**l, ::2**l]
        return jax.tree.map(mapper, levels)