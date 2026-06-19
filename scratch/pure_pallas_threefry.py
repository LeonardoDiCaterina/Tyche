import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import time

def pallas_threefry_2x32(key, N):
    BLOCK_SIZE = 1024
    num_blocks = N // BLOCK_SIZE
    
    C240 = jnp.uint32(0x1BD11BDA)
    
    def kernel(key_ref, out0_ref, out1_ref):
        pid = pl.program_id(0)
        
        K0 = key_ref[0]
        K1 = key_ref[1]
        K2 = K0 ^ K1 ^ C240
        
        # Use broadcasted_iota to generate the counter internally
        iota = jax.lax.broadcasted_iota(jnp.uint32, (BLOCK_SIZE,), 0)
        x0 = iota + jnp.uint32(pid * BLOCK_SIZE)
        x1 = jnp.zeros((BLOCK_SIZE,), dtype=jnp.uint32)
        
        R_CONSTS = (13, 15, 26, 6, 17, 29, 16, 24)
        
        # Unroll 20 rounds
        for r in range(20):
            if r % 4 == 0:
                s = r // 4
                if s % 3 == 0:
                    x0 = x0 + K0
                    x1 = x1 + K1 + jnp.uint32(s)
                elif s % 3 == 1:
                    x0 = x0 + K1
                    x1 = x1 + K2 + jnp.uint32(s)
                elif s % 3 == 2:
                    x0 = x0 + K2
                    x1 = x1 + K0 + jnp.uint32(s)
                    
            rot = R_CONSTS[r % 8]
            x0 = x0 + x1
            x1 = (x1 << rot) | (x1 >> (32 - rot))
            x1 = x1 ^ x0
            
        # Final key injection
        s = 5
        x0 = x0 + K2
        x1 = x1 + K0 + jnp.uint32(s)
        
        out0_ref[...] = x0
        out1_ref[...] = x1

    out0, out1 = pl.pallas_call(
        kernel,
        out_shape=[
            jax.ShapeDtypeStruct((num_blocks * BLOCK_SIZE,), jnp.uint32),
            jax.ShapeDtypeStruct((num_blocks * BLOCK_SIZE,), jnp.uint32)
        ],
        grid=(num_blocks,),
        in_specs=[
            pl.BlockSpec((2,), lambda i: (0,)),
        ],
        out_specs=[
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,))
        ]
    )(key)
    
    # Return as a single array of shape (N*2,)
    # By stacking and flattening, we emulate generating N*2 random uint32s.
    return jnp.concatenate([out0, out1])

def benchmark(name, func, N, num_warmups=5, num_steps=20):
    # Warmup
    for _ in range(num_warmups):
        res = func()
        res.block_until_ready()
    
    t0 = time.perf_counter()
    for _ in range(num_steps):
        res = func()
        res.block_until_ready()
    t1 = time.perf_counter()
    
    avg_time = (t1 - t0) / num_steps
    bytes_generated = (N * 2 * 4) # N pairs of uint32
    gbps = (bytes_generated / 1e9) / avg_time
    
    print(f"{name:<25}: {gbps:>8.2f} GB/s")
    return gbps

if __name__ == "__main__":
    N = 100000000 # 100M pairs = 200M random numbers
    key = jax.random.key(42)
    
    print(f"Benchmarking Pure Pallas Threefry vs Native JAX")
    print(f"Generating {N*2} random uint32s ({(N*2*4)/1e9:.2f} GB) per step...\n")
    
    # 1. Native JAX
    # jax.random.uniform generates random numbers internally using Threefry
    @jax.jit
    def native_jax():
        return jax.random.uniform(key, shape=(N*2,), dtype=jnp.float32)
        
    benchmark("Native JAX (Threefry)", native_jax, N)
    
    # 2. Pure Pallas
    @jax.jit
    def pallas_jax():
        return pallas_threefry_2x32(key, N)
        
    benchmark("Pure Pallas (Threefry)", pallas_jax, N)
