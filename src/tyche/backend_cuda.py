import jax
import jax.core as jax_core
import jax.extend.core as core_ext
import jax.numpy as jnp
from jax.interpreters import mlir, xla
from jax import ffi
from jax.lib import xla_client
import struct

try:
    import tyche_csrc
    for name, fn in tyche_csrc.registrations().items():
        xla_client.register_custom_call_target(name, fn, platform="gpu")
        xla_client.register_custom_call_target(name, fn, platform="CUDA")
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import tyche_csrc: {e}. Did you compile the C++ extension?")

tyche_dummy_p = core_ext.Primitive("tyche_dummy")
tyche_dummy_p.multiple_results = False
tyche_dummy_p.def_impl(lambda **kwargs: xla.apply_primitive(tyche_dummy_p, **kwargs))

@tyche_dummy_p.def_abstract_eval
def tyche_dummy_abstract_eval(*, size):
    return jax_core.ShapedArray((size,), jnp.uint32)

def tyche_dummy_lowering(ctx, *, size):
    opaque = struct.pack("i", size)
    out_type = mlir.ir.RankedTensorType.get([size], mlir.ir.IntegerType.get_unsigned(32))
    return mlir.custom_call(
        "tyche_dummy",
        result_types=[out_type],
        operands=[],
        backend_config=opaque,
        api_version=1, # original API without XlaCustomCallStatus
    ).results

mlir.register_lowering(
    tyche_dummy_p,
    tyche_dummy_lowering,
    platform="gpu"
)

def dummy_fill(size: int):
    return tyche_dummy_p.bind(size=size)
