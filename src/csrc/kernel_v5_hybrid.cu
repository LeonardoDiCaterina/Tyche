#include <stdint.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>

using namespace nvcuda;

#define FAST_MUL1 0xBF58476Du
#define FAST_MUL2 0x94D049BBu

__device__ uint32_t fast_mix_u32_v5(uint32_t x) {
    x = (x ^ (x >> 16)) * FAST_MUL1;
    x = (x ^ (x >> 13)) * FAST_MUL2;
    x = x ^ (x >> 16);
    return x;
}

// Each warp processes 1 tile (T=16).
// Threads per block should be a multiple of 32 (e.g., 256 -> 8 warps)
__global__ void tyche_v5_hybrid_kernel(
    uint32_t* out,
    const uint32_t* key,
    const uint32_t* weight_matrices,
    int offset,
    int num_tiles,
    int T,
    int R,
    int embedding_type,
    const uint32_t* key_mix_ptr
) {
    // Only support T=16 for now
    if (T != 16) return;

    uint32_t key_mix = *key_mix_ptr;

    int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    
    if (warp_idx >= num_tiles) return;
    
    uint32_t tile_idx = (uint32_t)offset + (uint32_t)warp_idx;
    
    // Shared memory ONLY for initial int8 input to wmma
    extern __shared__ int8_t smem_base_v5[];
    int warp_in_block = threadIdx.x / 32;
    int8_t* my_smem_i8 = smem_base_v5 + warp_in_block * 256;
    
    // 1. Initial make_tile
    for (int i = 0; i < 8; ++i) {
        int idx = i * 32 + lane;
        int r = idx / 16;
        int c = idx % 16;
        
        uint32_t v = 0;
        if (embedding_type == 1) { // diagonal
            v = (r == c) ? fast_mix_u32_v5(key_mix ^ tile_idx) : 0;
        } else if (embedding_type == 2) { // row
            v = fast_mix_u32_v5(key_mix ^ tile_idx ^ ((uint32_t)r * 1234567891u));
        } else if (embedding_type == 3) { // rank1
            uint32_t v1 = fast_mix_u32_v5(key_mix ^ tile_idx ^ (uint32_t)r);
            uint32_t v2 = fast_mix_u32_v5(key_mix ^ tile_idx ^ (uint32_t)c);
            v = v1 * v2;
        } else { // hash (0)
            v = key_mix ^ (tile_idx * 2654435761u);
            v = v ^ ((uint32_t)r * 1234567891u);
            v = v ^ ((uint32_t)c * 987654321u);
            v = fast_mix_u32_v5(v);
        }
        my_smem_i8[idx] = (int8_t)v;
    }
    
    __syncwarp();
    
    // 2. WMMA
    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::row_major> b_frag; 
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> acc_frag;
    
    const uint32_t* W_0 = weight_matrices; // Only use W_0 for R=1 pass
    
    wmma::load_matrix_sync(a_frag, my_smem_i8, 16);
    wmma::load_matrix_sync(b_frag, my_smem_i8, 16);
    wmma::load_matrix_sync(acc_frag, (const int32_t*)W_0, 16, wmma::mem_row_major);
    
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    
    // 3. The 5-stage Butterfly Network in ALUs
    for (int i = 0; i < 8; ++i) {
        uint32_t v = (uint32_t)acc_frag.x[i];
        
        // Initial non-linear mix
        v = fast_mix_u32_v5(v);
        
        // Stage 1 (swap distance 1)
        v = fast_mix_u32_v5(v ^ __shfl_xor_sync(0xffffffff, v, 1));
        
        // Stage 2 (swap distance 2)
        v = fast_mix_u32_v5(v ^ __shfl_xor_sync(0xffffffff, v, 2));
        
        // Stage 3 (swap distance 4)
        v = fast_mix_u32_v5(v ^ __shfl_xor_sync(0xffffffff, v, 4));
        
        // Stage 4 (swap distance 8)
        v = fast_mix_u32_v5(v ^ __shfl_xor_sync(0xffffffff, v, 8));
        
        // Stage 5 (swap distance 16)
        v = fast_mix_u32_v5(v ^ __shfl_xor_sync(0xffffffff, v, 16));
        
        // Write out directly to global memory as uint32_t
        // Out is dimensioned [num_tiles, 256]. Each thread writes 8 elements.
        int out_idx = i * 32 + lane;
        out[warp_idx * 256 + out_idx] = v;
    }
}

void tyche_v5_hybrid_kernel_launch(
    cudaStream_t stream,
    uint32_t* out,
    const uint32_t* key,
    const uint32_t* weight_matrices,
    int offset,
    int num_tiles,
    int T,
    int R,
    int embedding_type,
    const uint32_t* key_mix_ptr
) {
    if (T != 16) return; 
    
    int threads = 256;
    int warps_per_block = threads / 32; // 8
    int blocks = (num_tiles + warps_per_block - 1) / warps_per_block;
    
    // Shared memory: only 256 bytes for int8, per warp
    size_t smem_size = warps_per_block * 256;
    
    if (blocks > 0) {
        tyche_v5_hybrid_kernel<<<blocks, threads, smem_size, stream>>>(
            out, key, weight_matrices, offset, num_tiles, T, R, embedding_type, key_mix_ptr
        );
    }
}
