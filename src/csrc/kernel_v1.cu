#include <stdint.h>
#include <cuda_runtime.h>
#include <iostream>

#define ODD_MULT 0x94D049BBu
#define FAST_MUL1 0xBF58476Du
#define FAST_MUL2 0x94D049BBu

__device__ uint32_t fast_mix_u32(uint32_t x) {
    x = (x ^ (x >> 16)) * FAST_MUL1;
    x = (x ^ (x >> 13)) * FAST_MUL2;
    x = x ^ (x >> 16);
    return x;
}

__global__ void tyche_v1_kernel(
    int8_t* out,
    const uint32_t* key,
    const uint32_t* weight_matrices,
    int offset,
    int num_tiles,
    int T,
    int R,
    int embedding_type,
    uint32_t key_mix
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tiles) return;
    
    uint32_t tile_idx = (uint32_t)offset + (uint32_t)idx;
    
    // Allocate tile in local memory (max T=64 means 4096 bytes, which might spill to local memory but that's fine for MVP)
    int8_t x[4096];
    
    // make_tile logic
    for (int r = 0; r < T; ++r) {
        for (int c = 0; c < T; ++c) {
            uint32_t v = 0;
            if (embedding_type == 1) { // diagonal
                v = (r == c) ? fast_mix_u32(key_mix ^ tile_idx) : 0;
            } else if (embedding_type == 2) { // row
                v = fast_mix_u32(key_mix ^ tile_idx ^ ((uint32_t)r * 1234567891u));
            } else if (embedding_type == 3) { // rank1
                uint32_t v1 = fast_mix_u32(key_mix ^ tile_idx ^ (uint32_t)r);
                uint32_t v2 = fast_mix_u32(key_mix ^ tile_idx ^ (uint32_t)c);
                v = v1 * v2;
            } else { // hash (0)
                v = key_mix ^ (tile_idx * 2654435761u);
                v = v ^ ((uint32_t)r * 1234567891u);
                v = v ^ ((uint32_t)c * 987654321u);
                v = fast_mix_u32(v);
            }
            x[r * T + c] = (int8_t)v;
        }
    }
    
    // _hash_tile logic
    int32_t acc[4096];
    for (int round = 0; round < R; ++round) {
        const uint32_t* W_r = weight_matrices + round * T * T;
        
        // Matrix multiplication: acc = x * x + W_r
        for (int i = 0; i < T; ++i) {
            for (int j = 0; j < T; ++j) {
                int32_t sum = (int32_t)W_r[i * T + j];
                for (int k = 0; k < T; ++k) {
                    sum += (int32_t)x[i * T + k] * (int32_t)x[k * T + j];
                }
                acc[i * T + j] = sum;
            }
        }
        
        // ALU Fold
        for (int i = 0; i < T * T; ++i) {
            uint32_t acc_u32 = (uint32_t)acc[i];
            acc_u32 = acc_u32 * ODD_MULT;
            uint32_t alu_mixed = acc_u32 ^ (acc_u32 >> 16);
            x[i] = (int8_t)alu_mixed;
        }
    }
    
    // Write out
    for (int i = 0; i < T * T; ++i) {
        out[idx * T * T + i] = x[i];
    }
}

void tyche_v1_kernel_launch(
    cudaStream_t stream,
    int8_t* out,
    const uint32_t* key,
    const uint32_t* weight_matrices,
    int offset,
    int num_tiles,
    int T,
    int R,
    int embedding_type,
    uint32_t key_mix
) {
    int threads = 256;
    int blocks = (num_tiles + threads - 1) / threads;
    if (blocks > 0) {
        tyche_v1_kernel<<<blocks, threads, 0, stream>>>(
            out, key, weight_matrices, offset, num_tiles, T, R, embedding_type, key_mix
        );
    }
}
