#include <stdint.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>

using namespace nvcuda;

#define ODD_MULT 0x94D049BBu
#define FAST_MUL1 0xBF58476Du
#define FAST_MUL2 0x94D049BBu

__device__ uint32_t fast_mix_u32_v2(uint32_t x) {
    x = (x ^ (x >> 16)) * FAST_MUL1;
    x = (x ^ (x >> 13)) * FAST_MUL2;
    x = x ^ (x >> 16);
    return x;
}

// Each warp processes 1 tile (T=16).
// Threads per block should be a multiple of 32 (e.g., 256 -> 8 warps)
__global__ void tyche_v2_wmma_kernel(
    int8_t* out,
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
    
    // Shared memory for TC output and next round input
    // Each warp needs 16x16x4 bytes = 1024 bytes for int32, plus 256 bytes for int8
    // Max threads per block = 256 (8 warps).
    // 8 * 1024 = 8192 bytes (8KB).
    extern __shared__ int8_t smem_base[];
    
    int warp_in_block = threadIdx.x / 32;
    // int32 buffer takes first 8KB (if 8 warps)
    int32_t* my_smem_i32 = (int32_t*)(smem_base) + warp_in_block * 256;
    // int8 buffer takes next 2KB (if 8 warps)
    int8_t* my_smem_i8 = smem_base + (blockDim.x / 32) * 1024 + warp_in_block * 256;
    
    // 1. Initial make_tile (each thread initializes 8 elements of the 256 element tile)
    for (int i = 0; i < 8; ++i) {
        int idx = i * 32 + lane;
        int r = idx / 16;
        int c = idx % 16;
        
        uint32_t v = 0;
        if (embedding_type == 1) { // diagonal
            v = (r == c) ? fast_mix_u32_v2(key_mix ^ tile_idx) : 0;
        } else if (embedding_type == 2) { // row
            v = fast_mix_u32_v2(key_mix ^ tile_idx ^ ((uint32_t)r * 1234567891u));
        } else if (embedding_type == 3) { // rank1
            uint32_t v1 = fast_mix_u32_v2(key_mix ^ tile_idx ^ (uint32_t)r);
            uint32_t v2 = fast_mix_u32_v2(key_mix ^ tile_idx ^ (uint32_t)c);
            v = v1 * v2;
        } else { // hash (0)
            v = key_mix ^ (tile_idx * 2654435761u);
            v = v ^ ((uint32_t)r * 1234567891u);
            v = v ^ ((uint32_t)c * 987654321u);
            v = fast_mix_u32_v2(v);
        }
        my_smem_i8[idx] = (int8_t)v;
    }
    
    __syncwarp();
    
    // 2. WMMA loops
    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::row_major> b_frag; 
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> acc_frag;
    
    for (int round = 0; round < R; ++round) {
        const uint32_t* W_r = weight_matrices + round * 256; 
        
        wmma::load_matrix_sync(a_frag, my_smem_i8, 16);
        wmma::load_matrix_sync(b_frag, my_smem_i8, 16);
        
        // Load W_r directly into accumulator
        wmma::load_matrix_sync(acc_frag, (const int32_t*)W_r, 16, wmma::mem_row_major);
        
        // Tensor Core Matmul! acc_frag = a_frag * b_frag + acc_frag
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        
        // Store to shared memory
        wmma::store_matrix_sync(my_smem_i32, acc_frag, 16, wmma::mem_row_major);
        __syncwarp();
        
        // ALU Fold in registers and write downcast to smem_i8
        for (int i = 0; i < 8; ++i) {
            int idx = i * 32 + lane;
            uint32_t acc_u32 = (uint32_t)my_smem_i32[idx];
            acc_u32 = acc_u32 * ODD_MULT;
            uint32_t alu_mixed = acc_u32 ^ (acc_u32 >> 16);
            my_smem_i8[idx] = (int8_t)alu_mixed;
        }
        __syncwarp();
    }
    
    // 3. Write out
    for (int i = 0; i < 8; ++i) {
        int idx = i * 32 + lane;
        out[warp_idx * 256 + idx] = my_smem_i8[idx];
    }
}

void tyche_v2_wmma_kernel_launch(
    cudaStream_t stream,
    int8_t* out,
    const uint32_t* key,
    const uint32_t* weight_matrices,
    int offset,
    int num_tiles,
    int T,
    int R,
    int embedding_type,
    const uint32_t* key_mix_ptr
) {
    if (T != 16) return; // Only T=16 supported for WMMA in this MVP
    
    int threads = 256;
    int warps_per_block = threads / 32; // 8
    int blocks = (num_tiles + warps_per_block - 1) / warps_per_block;
    
    // Shared memory: 1024 bytes for int32 + 256 bytes for int8, per warp
    size_t smem_size = warps_per_block * (1024 + 256);
    
    if (blocks > 0) {
        tyche_v2_wmma_kernel<<<blocks, threads, smem_size, stream>>>(
            out, key, weight_matrices, offset, num_tiles, T, R, embedding_type, key_mix_ptr
        );
    }
}
