import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def kernel(x_ref, out_ref):
    x = pl.load(x_ref, (pl.dslice(16), pl.dslice(16)))
    acc = pl.dot(x, x)
    pl.store(out_ref, (pl.dslice(16), pl.dslice(16)), acc)

x = jnp.ones((16, 16), dtype=jnp.int8)
out = pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct((16, 16), jnp.int32),
    in_specs=[pl.BlockSpec((16, 16), lambda: (0, 0))],
    out_specs=pl.BlockSpec((16, 16), lambda: (0, 0))
)(x)
print("SUCCESS")
