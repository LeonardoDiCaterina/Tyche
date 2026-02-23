import jax.numpy as jnp
import jax
from jax.experimental import pallas as pl
import jax.experimental.pallas.triton as plgpu  # GPU path


class PallasBackend:
    def __init__(self, num_rounds: int, block_size: int):
        self.R = num_rounds
        self.B = block_size

    def hash_block(self, counter_block, weight_matrices):
        B = self.B

        def kernel(counter_ref, weights_ref, out_ref):
            x = counter_ref[...].astype(jnp.uint32)        # (B,B)
            for r in range(self.R):                         # unrolled at trace time
                W_r = weights_ref[r, :, :]                  # (B,B) uint32
                acc = pl.dot(x, x) + W_r                    # fused matmul+add
                # ALU Bridge: include odd-multiply to match JAX backend
                acc = acc * jnp.uint32(0x94D049BB)
                x = (acc ^ (acc >> jnp.uint32(16))).astype(jnp.uint16)
                x = x.astype(jnp.uint32)                   # widen for next round
            out_ref[...] = x.astype(jnp.uint16)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((B, B), jnp.uint16),
            grid=(1,),
            in_specs=[
                pl.BlockSpec((B, B), lambda i: (0, 0)),     # counter_block
                pl.BlockSpec((self.R, B, B), lambda i: (0, 0, 0)),  # weights
            ],
            out_specs=pl.BlockSpec((B, B), lambda i: (0, 0)),
        )(counter_block, weight_matrices)

    def hash_parallel(self, counter_blocks, weight_matrices):
        N = counter_blocks.shape[0]
        B = self.B

        def kernel(counters_ref, weights_ref, out_ref):
            # Each grid program handles one block
            i = pl.program_id(0)
            x = pl.load(counters_ref, (i, pl.dslice(B), pl.dslice(B))).astype(jnp.uint32)
            for r in range(self.R):
                W_r = weights_ref[r, :, :]
                acc = pl.dot(x, x) + W_r
                acc = acc * jnp.uint32(0x94D049BB)
                x = (acc ^ (acc >> jnp.uint32(16))).astype(jnp.uint16)
                x = x.astype(jnp.uint32)
            pl.store(out_ref, (i, pl.dslice(B), pl.dslice(B)), x.astype(jnp.uint16))

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((N, B, B), jnp.uint16),
            grid=(N,),
            in_specs=[
                pl.BlockSpec((1, B, B), lambda i: (i, 0, 0)),
                pl.BlockSpec((self.R, B, B), lambda i: (0, 0, 0)),
            ],
            out_specs=pl.BlockSpec((1, B, B), lambda i: (i, 0, 0)),
        )(counter_blocks, weight_matrices)

    def apply_perturbation(self, weight_matrices, perturbation):
        # Same pattern — one grid program per round
        R, B = self.R, self.B

        def kernel(w_ref, p_ref, out_ref):
            r = pl.program_id(0)
            W = pl.load(w_ref, (r, pl.dslice(B), pl.dslice(B)))
            P = p_ref[...]
            out = pl.dot(W, W) + P
            pl.store(out_ref, (r, pl.dslice(B), pl.dslice(B)), out)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((R, B, B), jnp.uint32),
            grid=(R,),
            in_specs=[
                pl.BlockSpec((1, B, B), lambda r: (r, 0, 0)),
                pl.BlockSpec((B, B), lambda r: (0, 0)),
            ],
            out_specs=pl.BlockSpec((1, B, B), lambda r: (r, 0, 0)),
        )(weight_matrices, perturbation)

    def make_counter_blocks(self, key, offset, num_blocks, block_size):
        # Counter generation is cheap & scalar-heavy — keep in JAX
        from tyche.algorithm import make_counter_blocks
        return make_counter_blocks(key, offset, num_blocks, block_size)