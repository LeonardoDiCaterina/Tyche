import math
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)  
try:
    from jax._src.random.prng import PRNGImpl
except ImportError:
    from jax._src.prng import PRNGImpl

from tyche.v3_philox.algorithm import (
    key_size_uint,
    expand_seed_to_key,
    _key_to_matrices,
    derive_child_key,
    make_tiles
)
from tyche.v4_threefry.algorithm import make_hash_parallel

class TycheV4_ThreefryConfig:
    def __init__(self, tile_size: int = 16, num_rounds: int = 20, word_size: int = 32, embedding: str = "hash", backend: str = "pallas"):
        if backend == "pallas":
            from tyche.v4_threefry.backend_pallas import PallasBackendV4_Threefry
            self._backend = PallasBackendV4_Threefry(num_rounds, tile_size, word_size)
            self._hash_parallel = self._backend.hash_parallel
            self._make_tiles = self._backend.make_tiles
        else:
            self._backend = None
            self._hash_parallel = make_hash_parallel(num_rounds, word_size)
            self._make_tiles = make_tiles
        
        if word_size not in (32, 64):
            raise ValueError(f"word_size must be 32 or 64 — got {word_size}")
        if tile_size not in (2, 4, 16, 32, 64):
            raise ValueError(f"tile_size must be one of 2, 4, 16, 32, 64 — got {tile_size}")
        if num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1 — got {num_rounds}")

        self.tile_size = tile_size
        self.num_rounds = num_rounds
        self.word_size = word_size
        self.embedding = embedding

    @property
    def key_shape(self):
        return (key_size_uint(self.num_rounds, self.tile_size, self.word_size),)

    @property
    def name(self):
        return f"tyc_tf_w{self.word_size}_t{self.tile_size}_r{self.num_rounds}_{self.embedding}"

    def _seed(self, seed) -> jnp.ndarray:
        return expand_seed_to_key(seed, self.num_rounds, self.tile_size, self.word_size)

    def _split(self, key: jnp.ndarray, shape: tuple) -> jnp.ndarray:
        num = math.prod(shape)
        child_indices = jnp.arange(num, dtype=jnp.uint64)
        children = jax.vmap(
            lambda i: derive_child_key(key, i, self.num_rounds, self.tile_size, self.word_size)
        )(child_indices)
        return children.reshape(*shape, *self.key_shape)

    def _fold_in(self, key: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
        return derive_child_key(
            key, data.astype(jnp.uint64), self.num_rounds, self.tile_size, self.word_size
        )

    def _random_bits(self, key: jnp.ndarray, bit_width: int, shape: tuple) -> jnp.ndarray:
        T, R, W = self.tile_size, self.num_rounds, self.word_size

        total_out_elems = math.prod(shape)
        bytes_per_elem = bit_width // 8
        total_bytes_needed = total_out_elems * bytes_per_elem
        
        bytes_per_tile = T * T * (W // 8)
        num_tiles = math.ceil(total_bytes_needed / bytes_per_tile)

        weight_matrices = _key_to_matrices(key, R, T, W)
        tiles_in = self._make_tiles(key, 0, num_tiles, T, self.embedding, W)
        hashed_tiles = self._hash_parallel(tiles_in, weight_matrices)

        flat_bytes = hashed_tiles.reshape(-1).view(jnp.uint8)[:total_bytes_needed]
        
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
            tag=f"tyc_tf{self.word_size}",
        )
