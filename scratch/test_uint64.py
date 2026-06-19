import jax.numpy as jnp

val = 0xD2B74407B1CE6E93
signed_val = val - (1 << 64)
print(f"Signed val: {signed_val}")

arr = jnp.uint64(signed_val)
hex_str = hex(int(arr))
print(f"Hex of uint64(signed_val): {hex_str}")

assert hex_str.upper() == "0XD2B74407B1CE6E93"
print("Match!")
