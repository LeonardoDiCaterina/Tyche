# ── src/tyche/backend_jax.py ─────────────────────────────────────

import jax
from tyche import algorithm

# use the current algorithm implementation; algorithm_old is deprecated
_apply_perturbation = algorithm._apply_perturbation
_hash_block = algorithm._hash_block
make_counter_blocks = algorithm.make_counter_blocks


class JaxBackend:
    def __init__(self, num_rounds: int, block_size: int):
        self.R = num_rounds
        self.B = block_size

    def hash_block(self, counter_block, weight_matrices):
        return _hash_block(counter_block, weight_matrices)

    def hash_parallel(self, counter_blocks, weight_matrices):
        return jax.vmap(_hash_block, in_axes=(0, None))(
            counter_blocks, weight_matrices
        )

    def apply_perturbation(self, weight_matrices, perturbation):
        return _apply_perturbation(weight_matrices, perturbation)

    def make_counter_blocks(self, key, offset, num_blocks, block_size):
        return make_counter_blocks(key, offset, num_blocks, block_size)