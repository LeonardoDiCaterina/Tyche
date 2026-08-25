import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import os

os.environ["JAX_PLATFORMS"] = "cpu"

def test_kernel(out_ref):
    rows = jnp.arange(16, dtype=jnp.uint32)[:, None]
    cols = jnp.arange(16, dtype=jnp.uint32)[None, :]
    val = rows + cols
    out_ref[...] = val.astype(jnp.int8)

out = pl.pallas_call(test_kernel, out_shape=jax.ShapeDtypeStruct((16, 16), jnp.int8), grid=(1,))()
print(out)
