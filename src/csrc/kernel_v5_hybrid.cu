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
    
    // Allocate shared memory for BOTH 32-bit (for acc_frag) and 8-bit (for a_frag/b_frag)
    extern __shared__ uint32_t smem_base_v5_32[];
    int warp_in_block = threadIdx.x / 32;
    uint32_t* my_smem_u32 = smem_base_v5_32 + warp_in_block * 256;
    // Put the 8-bit array after the 32-bit arrays of ALL warps in the block
    int num_warps = blockDim.x / 32;
    int8_t* my_smem_i8 = (int8_t*)(smem_base_v5_32 + num_warps * 256) + warp_in_block * 256;
    
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
        
        // Store full 32-bit entropy to load into the accumulator
        my_smem_u32[idx] = v;
        // Truncate to 8-bit for the A/B matrices
        my_smem_i8[idx] = (int8_t)v;
    }
    
    __syncwarp();
    
    // 2. WMMA
    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::row_major> b_frag; 
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> acc_frag;
    
    wmma::load_matrix_sync(a_frag, my_smem_i8, 16);
    wmma::load_matrix_sync(b_frag, my_smem_i8, 16);
    // Load the full 32-bit state into the accumulator to preserve entropy!
    wmma::load_matrix_sync(acc_frag, my_smem_u32, 16, wmma::mem_row_major);
    
    // Compute A * B + acc_frag (which holds the full 32-bit entropy)
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    
    // 3. The 5-stage Butterfly Network in ALUs
    for (int i = 0; i < 8; ++i) {
        uint32_t v = (uint32_t)acc_frag.x[i];
        
        // Initial non-linear mix with tile context
        v = fast_mix_u32_v5(v ^ tile_idx);
        uint32_t v_other;
        
        // Asymmetric Butterfly: We use addition instead of XOR to prevent identical thread states,
        // and we inject the thread lane ID to explicitly break symmetry across the warp!
        
        // Stage 1 (swap distance 1)
        v_other = __shfl_xor_sync(0xffffffff, v, 1);
        v = fast_mix_u32_v5(v + (v_other ^ 0x9E3779B9u) + lane);
        
        // Stage 2 (swap distance 2)
        v_other = __shfl_xor_sync(0xffffffff, v, 2);
        v = fast_mix_u32_v5(v + (v_other ^ 0x85EBCA6Bu) + lane);
        
        // Stage 3 (swap distance 4)
        v_other = __shfl_xor_sync(0xffffffff, v, 4);
        v = fast_mix_u32_v5(v + (v_other ^ 0xC2B2AE35u) + lane);
        
        // Stage 4 (swap distance 8)
        v_other = __shfl_xor_sync(0xffffffff, v, 8);
        v = fast_mix_u32_v5(v + (v_other ^ 0x27D4EB2Fu) + lane);
        
        // Stage 5 (swap distance 16)
        v_other = __shfl_xor_sync(0xffffffff, v, 16);
        v = fast_mix_u32_v5(v + (v_other ^ 0x165667B1u) + lane);
        
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
    
    // Shared memory: 1024 bytes for uint32 + 256 bytes for int8 = 1280 bytes per warp
    size_t smem_size = warps_per_block * 1280;
    
    if (blocks > 0) {
        tyche_v5_hybrid_kernel<<<blocks, threads, smem_size, stream>>>(
            out, key, weight_matrices, offset, num_tiles, T, R, embedding_type, key_mix_ptr
        );
    }
}
