import jax
from jax import core
import jax.numpy as jnp
from jax.interpreters import xla, mlir
from jax.lib import xla_client
import struct

try:
    import tyche_csrc
    for name, fn in tyche_csrc.registrations().items():
        xla_client.register_custom_call_target(name, fn, platform="gpu")
except ImportError:
    pass # Will be handled by the user ensuring the extension is built

tyche_dummy_p = core.Primitive("tyche_dummy")
tyche_dummy_p.def_impl(lambda size: xla.apply_primitive(tyche_dummy_p, size))

@tyche_dummy_p.def_abstract_eval
def tyche_dummy_abstract_eval(size):
    return core.ShapedArray((size,), jnp.uint32)

def tyche_dummy_lowering(ctx, size):
    # size is known at tracing time
    opaque = struct.pack("i", size)
    
    out_type = mlir.ir.RankedTensorType.get([size], mlir.ir.IntegerType.get_signless(32))
    
    return mlir.custom_call(
        "tyche_dummy",
        result_types=[out_type],
        operands=[],
        backend_config=opaque,
        api_version=2, # xla_client.api_version_custom_call_status
    ).results
    
mlir.register_lowering(tyche_dummy_p, tyche_dummy_lowering, platform="gpu")

def dummy_fill(size: int):
    return tyche_dummy_p.bind(size)
