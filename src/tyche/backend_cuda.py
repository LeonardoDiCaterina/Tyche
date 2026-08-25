import jax
import jax.extend.core as core
import jax.numpy as jnp
from jax.interpreters import mlir
from jax.extend import ffi
from jax.lib import xla_client
import struct

try:
    import tyche_csrc
    for name, fn in tyche_csrc.registrations().items():
        xla_client.register_custom_call_target(name, fn, platform="gpu")
        ffi.register_ffi_target(name, fn, platform="gpu")
except ImportError:
    pass # Will be handled by the user ensuring the extension is built

tyche_dummy_p = core.Primitive("tyche_dummy")
tyche_dummy_p.def_impl(lambda size: jax.jit(lambda size: tyche_dummy_p.bind(size))(size))

@tyche_dummy_p.def_abstract_eval
def tyche_dummy_abstract_eval(size):
    return core.ShapedArray((size,), jnp.uint32)

mlir.register_lowering(
    tyche_dummy_p,
    ffi.ffi_lowering("tyche_dummy"),
    platform="gpu"
)

def dummy_fill(size: int):
    return tyche_dummy_p.bind(size)
