import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def kernel(x_ref, w_ref, out_ref):
    x = pl.load(x_ref, (pl.dslice(16), pl.dslice(16)))
    for r in range(2):
        z = jnp.zeros((16, 16), dtype=jnp.int8)
        x_lhs = jnp.concatenate([x, z], axis=1)
        x_rhs = jnp.concatenate([x, z], axis=0)
        acc = pl.dot(x_lhs, x_rhs)
        x = acc.astype(jnp.int8)
    
    pl.store(out_ref, (pl.dslice(16), pl.dslice(16)), x.astype(jnp.int32))

x = jnp.ones((16, 16), dtype=jnp.int8)
w = jnp.ones((16, 16), dtype=jnp.int32)
# We use interpret mode which runs on CPU but does shape checking
out = pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct((16, 16), jnp.int32),
    in_specs=[pl.BlockSpec((16, 16), lambda: (0, 0)), pl.BlockSpec((16, 16), lambda: (0, 0))],
    out_specs=pl.BlockSpec((16, 16), lambda: (0, 0)),
    interpret=True
)(x, w)
print("SUCCESS")
