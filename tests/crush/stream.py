"""
Generates a raw uint32 byte stream from Tyche for piping into PractRand.

Usage:
    python stream.py --n 25000000 | RNG_test stdin32 -tlmax 100MB
"""

import argparse
import sys
import os

# Suppress JAX output before import — prevents text from corrupting binary stream
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

# Redirect stderr temporarily to suppress JAX init messages
import io
_old_stderr = sys.stderr
sys.stderr = io.StringIO()
import jax
import jax.numpy as jnp
from tyche import impl
sys.stderr = _old_stderr


class TycheStream:
    """
    Infinite stream of uint32 values from Tyche.
    Manages key splitting internally so the stream never repeats a key.
    """

    def __init__(self, seed: int = 42, chunk_size: int = 100_000):
        self.chunk_size = chunk_size
        self._key = jax.random.key(seed, impl=impl)

    def write_binary(self, n: int, file=None):
        """Write n uint32 values as raw bytes to file (default: stdout)."""
        if file is None:
            file = sys.stdout.buffer
        remaining = n
        while remaining > 0:
            count = min(self.chunk_size, remaining)
            self._key, subkey = jax.random.split(self._key)
            chunk = np.array(
                jax.random.bits(subkey, shape=(count,), dtype=jnp.uint32),
                dtype=np.uint32
            )
            file.write(chunk.tobytes())
            file.flush()
            remaining -= count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream Tyche uint32s to stdout")
    parser.add_argument("--n", type=int, default=25_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stream = TycheStream(seed=args.seed)
    stream.write_binary(args.n)