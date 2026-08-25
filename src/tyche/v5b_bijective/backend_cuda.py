"""CUDA/C++ implementation of Tyche V5b Bijective algorithm."""
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from jax.lib import xla_client
from jax import core, dtypes
from jax.interpreters import mlir
from jax.interpreters.mlir import ir
import tyche_csrc
from .config import TycheV5bConfig

# Register the custom call targets
for name, fn in tyche_csrc.registrations().items():
    xla_client.register_custom_call_target(name, fn, platform="gpu")

_tyche_v5b_p = core.Primitive("tyche_v5b_hash")
_tyche_v5b_p.multiple_results = True

def generate_v5b_cuda(
    keys: jnp.ndarray,
    config: TycheV5bConfig,
    offset: int = 0
) -> jnp.ndarray:
    """Generate pseudo-random numbers using Tyche V5b Bijective (C++ backend)."""
    assert keys.dtype == jnp.uint32
    assert keys.shape == (8,)

    num_tiles = config.blocks * config.warps_per_block * 2 # V5b processes 2 tiles per warp
    total_elements = num_tiles * 256

    if config.R != 1 or config.T != 16:
        raise ValueError("V5b Bijective currently only supports T=16 and R=1")

    out = _tyche_v5b_p.bind(
        keys,
        config.weight_matrices,
        config.key_mix,
        offset=offset,
        num_tiles=num_tiles,
        T=config.T,
        R=config.R,
        embedding_type=config.embedding_type.value,
        total_elements=total_elements
    )
    
    return out[0].reshape((total_elements,))

def _tyche_v5b_abstract_eval(keys, weights, key_mix, **kwargs):
    total_elements = kwargs['total_elements']
    return (core.ShapedArray((total_elements,), jnp.uint32),)

def _tyche_v5b_lowering(ctx, keys, weights, key_mix, **kwargs):
    opaque = tyche_csrc.TycheV1ConfigOpaque()
    opaque.offset = kwargs['offset']
    opaque.num_tiles = kwargs['num_tiles']
    opaque.T = kwargs['T']
    opaque.R = kwargs['R']
    opaque.embedding_type = kwargs['embedding_type']
    
    opaque_bytes = bytes(opaque)
    
    out_type = ir.RankedTensorType.get(
        [kwargs['total_elements']], 
        ir.IntegerType.get_unsigned(32)
    )
    
    call = mlir.custom_call(
        "tyche_v5b_hash",
        result_types=[out_type],
        operands=[keys, weights, key_mix],
        backend_config=opaque_bytes,
        operand_layouts=[
            ir.AffineMap.get_permutation([0]),
            ir.AffineMap.get_permutation([0, 1]),
            ir.AffineMap.get_permutation([0]),
        ],
        result_layouts=[
            ir.AffineMap.get_permutation([0])
        ]
    )
    return call.results

_tyche_v5b_p.def_impl(lambda *args, **kwargs: xla_client.execute_with_python_fallback(
    _tyche_v5b_p, args, kwargs
))
_tyche_v5b_p.def_abstract_eval(_tyche_v5b_abstract_eval)
mlir.register_lowering(_tyche_v5b_p, _tyche_v5b_lowering, platform="gpu")
