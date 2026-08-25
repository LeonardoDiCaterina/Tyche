import jax
from tyche import algorithm

_apply_perturbation = algorithm._apply_perturbation
_hash_tile = algorithm._hash_tile
make_tile = algorithm.make_tile


class JaxBackend:
    def __init__(self, num_rounds: int, tile_size: int):
        self.R = num_rounds
        self.T = tile_size

    def hash_block(self, tile, weight_matrices):
        return _hash_tile(tile, weight_matrices)

    def hash_parallel(self, tiles, weight_matrices):
        return jax.vmap(_hash_tile, in_axes=(0, None))(
            tiles, weight_matrices
        )

    def apply_perturbation(self, weight_matrices, perturbation):
        return _apply_perturbation(weight_matrices, perturbation)

    def make_tile(self, key, offset, num_tiles, tile_size, embedding):
        return make_tile(key, offset, num_tiles, tile_size, embedding)