import jax
import jax.numpy as jnp

def test_mul():
    a = jnp.array([0xFFFFFFFF], dtype=jnp.uint32)
    b = jnp.array([0xFFFFFFFF], dtype=jnp.uint32)
    hi, lo = jax.lax.mul_extended(a, b)
    print("32-bit hi:", hi, "lo:", lo)

    a64 = jnp.array([0xFFFFFFFFFFFFFFFF], dtype=jnp.uint64)
    b64 = jnp.array([0xFFFFFFFFFFFFFFFF], dtype=jnp.uint64)
    hi64, lo64 = jax.lax.mul_extended(a64, b64)
    print("64-bit hi:", hi64, "lo:", lo64)

test_mul()
