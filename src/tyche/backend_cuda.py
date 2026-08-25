import jax
import jax.core as jax_core
import jax.extend.core as core_ext
import jax.numpy as jnp
from jax.interpreters import mlir, xla
from jax.lib import xla_client
import struct

from tyche import algorithm
_apply_perturbation = algorithm._apply_perturbation
make_tile = algorithm.make_tile
_mix_key_const = algorithm._mix_key_const

try:
    import tyche_csrc
    for name, fn in tyche_csrc.registrations().items():
        xla_client.register_custom_call_target(name, fn, platform="gpu")
        xla_client.register_custom_call_target(name, fn, platform="CUDA")
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import tyche_csrc: {e}. Did you compile the C++ extension?")

# ----------------- Dummy Fill Primitive -----------------
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
        api_version=1,
    ).results

mlir.register_lowering(tyche_dummy_p, tyche_dummy_lowering, platform="gpu")

def dummy_fill(size: int):
    return tyche_dummy_p.bind(size=size)

# ----------------- Tyche V1 Hash Primitive -----------------
tyche_v1_hash_p = core_ext.Primitive("tyche_v1_hash")
tyche_v1_hash_p.multiple_results = False
tyche_v1_hash_p.def_impl(lambda key, weight_matrices, key_mix, **kwargs: xla.apply_primitive(tyche_v1_hash_p, key, weight_matrices, key_mix, **kwargs))

@tyche_v1_hash_p.def_abstract_eval
def tyche_v1_hash_abstract_eval(key, weight_matrices, key_mix, *, offset, num_tiles, T, R, embedding_type):
    return jax_core.ShapedArray((num_tiles, T, T), jnp.int8)

def tyche_v1_hash_lowering(ctx, key, weight_matrices, key_mix, *, offset, num_tiles, T, R, embedding_type):
    # struct TycheV1ConfigOpaque { int offset, int num_tiles, int T, int R, int embedding_type; }
    opaque = struct.pack("iiiii", offset, num_tiles, T, R, embedding_type)
    
    out_type = mlir.ir.RankedTensorType.get(
        [num_tiles, T, T], 
        mlir.ir.IntegerType.get_signless(8)
    )
    
    return mlir.custom_call(
        "tyche_v1_hash",
        result_types=[out_type],
        operands=[key, weight_matrices, key_mix],
        backend_config=opaque,
        api_version=1,
    ).results

mlir.register_lowering(tyche_v1_hash_p, tyche_v1_hash_lowering, platform="gpu")

# ----------------- CudaBackend -----------------
class CudaBackend:
    def __init__(self, num_rounds: int, tile_size: int):
        self.R = num_rounds
        self.T = tile_size

    def hash_parallel(self, key, offset, num_tiles, weight_matrices, embedding):
        embedding_type = {"hash": 0, "diagonal": 1, "row": 2, "rank1": 3}.get(embedding, 0)
        key_mix = _mix_key_const(key)
        # weight_matrices is shaped (R, T, T) uint32. We flatten key and weight_matrices to uint32.
        flat_key = key.flatten()
        flat_weight_matrices = weight_matrices.flatten()
        
        return tyche_v1_hash_p.bind(
            flat_key,
            flat_weight_matrices,
            key_mix,
            offset=offset,
            num_tiles=num_tiles,
            T=self.T,
            R=self.R,
            embedding_type=embedding_type
        )

    def apply_perturbation(self, weight_matrices, perturbation):
        return _apply_perturbation(weight_matrices, perturbation)

    def make_tile(self, key, offset, num_tiles, tile_size, embedding):
        return make_tile(key, offset, num_tiles, tile_size, embedding)
