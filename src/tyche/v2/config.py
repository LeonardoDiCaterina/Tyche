import math
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)  
try:
    from jax._src.random.prng import PRNGImpl
except ImportError:
    from jax._src.prng import PRNGImpl

from tyche.v2.algorithm import (
    key_size_uint32,
    expand_seed_to_key,
    make_hash_parallel,
    make_tiles,
    _key_to_matrices,
    derive_child_key,
)

class TycheV2Config:
    def __init__(self, tile_size: int = 16, num_rounds: int = 4, embedding: str = "hash", backend: str = "jax"):
        if backend == "pallas":
            from tyche.v2.backend_pallas import PallasBackendV2
            self._backend = PallasBackendV2(num_rounds, tile_size)
            self._hash_parallel = self._backend.hash_parallel
            self._make_tiles = self._backend.make_tiles
        else:
            self._backend = None
            _hash_p = make_hash_parallel(num_rounds)
            self._make_tiles = make_tiles
            self._hash_parallel = lambda key, offset, num_tiles, w_mat, emb: _hash_p(
                make_tiles(key, offset, num_tiles, tile_size, emb), w_mat
            )
        
        if tile_size not in (16, 32, 64):
            raise ValueError(f"tile_size must be one of 16, 32, 64 — got {tile_size}")
        if num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1 — got {num_rounds}")

        self.tile_size = tile_size
        self.num_rounds = num_rounds
        self.embedding = embedding

    @property
    def key_shape(self):
        return (key_size_uint32(self.num_rounds, self.tile_size),)

    @property
    def name(self):
        return f"tyc2_t{self.tile_size}_r{self.num_rounds}_{self.embedding}"

    def _seed(self, seed) -> jnp.ndarray:
        return expand_seed_to_key(seed, self.num_rounds, self.tile_size)

    def _split(self, key: jnp.ndarray, shape: tuple) -> jnp.ndarray:
        num = math.prod(shape)
        child_indices = jnp.arange(num, dtype=jnp.uint64)
        children = jax.vmap(
            lambda i: derive_child_key(key, i, self.num_rounds, self.tile_size)
        )(child_indices)
        return children.reshape(*shape, *self.key_shape)

    def _fold_in(self, key: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
        return derive_child_key(
            key, data.astype(jnp.uint64), self.num_rounds, self.tile_size
        )

    def _random_bits(self, key: jnp.ndarray, bit_width: int, shape: tuple) -> jnp.ndarray:
        T, R = self.tile_size, self.num_rounds

        total_out_elems = math.prod(shape)
        bytes_per_elem = bit_width // 8
        total_bytes_needed = total_out_elems * bytes_per_elem
        
        bytes_per_tile = T * T
        num_tiles = math.ceil(total_bytes_needed / bytes_per_tile)

        weight_matrices = _key_to_matrices(key, R, T)
        hashed_tiles = self._hash_parallel(key, 0, num_tiles, weight_matrices, self.embedding)

        flat_bytes = hashed_tiles.reshape(-1)[:total_bytes_needed]
        
        out_dtype = jnp.uint32 if bit_width == 32 else jnp.uint64
        packed = flat_bytes.view(out_dtype)

        return packed.reshape(shape)

    def build(self) -> PRNGImpl:
        return PRNGImpl(
            key_shape=self.key_shape,
            seed=self._seed,
            split=self._split,
            fold_in=self._fold_in,
            random_bits=self._random_bits,
            name=self.name,
            tag="tyc2",
        )
