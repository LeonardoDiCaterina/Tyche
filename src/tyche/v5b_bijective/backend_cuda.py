import jax
import jax.core as jax_core
import jax.extend.core as core_ext
import jax.numpy as jnp
from jax.interpreters import mlir, xla
from jax.lib import xla_client
import struct

from tyche.v2 import algorithm
make_tiles = algorithm.make_tiles
_mix_key_const = algorithm._mix_key_const

try:
    import tyche_csrc
    for name, fn in tyche_csrc.registrations().items():
        xla_client.register_custom_call_target(name, fn, platform="gpu")
        xla_client.register_custom_call_target(name, fn, platform="CUDA")
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import tyche_csrc: {e}")

# ----------------- Tyche V5b Hash Primitive -----------------
tyche_v5b_hash_p = core_ext.Primitive("tyche_v5b_hash")
tyche_v5b_hash_p.multiple_results = False
tyche_v5b_hash_p.def_impl(lambda key, weight_matrices, key_mix, **kwargs: xla.apply_primitive(tyche_v5b_hash_p, key, weight_matrices, key_mix, **kwargs))

@tyche_v5b_hash_p.def_abstract_eval
def tyche_v5b_hash_abstract_eval(key, weight_matrices, key_mix, *, offset, num_tiles, T, R, embedding_type):
    # V5b returns uint32 instead of int8!
    return jax_core.ShapedArray((num_tiles, T, T), jnp.uint32)

def tyche_v5b_hash_lowering(ctx, key, weight_matrices, key_mix, *, offset, num_tiles, T, R, embedding_type):
    opaque = struct.pack("iiiii", offset, num_tiles, T, R, embedding_type)
    
    out_type = mlir.ir.RankedTensorType.get(
        [num_tiles, T, T], 
        mlir.ir.IntegerType.get_unsigned(32)
    )
    
    return mlir.custom_call(
        "tyche_v5b_hash",
        result_types=[out_type],
        operands=[key, weight_matrices, key_mix],
        backend_config=opaque,
        api_version=2,
    ).results

mlir.register_lowering(tyche_v5b_hash_p, tyche_v5b_hash_lowering, platform="gpu")

# ----------------- CudaBackendV5b -----------------
class CudaBackendV5b:
    def __init__(self, num_rounds: int, tile_size: int):
        self.R = num_rounds
        self.T = tile_size

    def hash_parallel(self, key, offset, num_tiles, weight_matrices, embedding):
        embedding_type = {"hash": 0, "diagonal": 1, "row": 2, "rank1": 3}.get(embedding, 0)
        key_mix = _mix_key_const(key)
        flat_key = key.flatten()
        flat_weight_matrices = weight_matrices.flatten()
        
        return tyche_v5b_hash_p.bind(
            flat_key,
            flat_weight_matrices,
            key_mix,
            offset=offset,
            num_tiles=num_tiles,
            T=self.T,
            R=self.R,
            embedding_type=embedding_type
        )
