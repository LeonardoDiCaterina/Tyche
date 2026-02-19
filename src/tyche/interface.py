"""
Registers the default Tyche PRNGImpl (block_size=4, num_rounds=16)
with JAX's internal PRNG registry so that split, fold_in, and random
operations correctly route to Tyche's quadratic FMA implementation
over GL_4(Z_256).
"""

import jax
jax.config.update("jax_enable_x64", True)

from jax._src import prng as jax_prng
from tyche.config import TycheConfig

_default_config = TycheConfig(block_size=4, num_rounds=16)
tyche_prng_impl = _default_config.build()

# Register with JAX's internal PRNG registry.
# This is required for jax.random.split / fold_in / uniform etc.
# to correctly route to Tyche's implementation rather than falling
# back to the default (Threefry) impl.
try:
    jax_prng.register_prng(tyche_prng_impl)
except ValueError:
    pass  # already registered (e.g. on module reload)