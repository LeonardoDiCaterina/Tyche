#include <stdint.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>

using namespace nvcuda;

#define FAST_MUL1 0xBF58476Du
#define FAST_MUL2 0x94D049BBu

__device__ uint32_t fast_mix_u32_v5b(uint32_t x) {
    x = (x ^ (x >> 16)) * FAST_MUL1;
    x = (x ^ (x >> 13)) * FAST_MUL2;
    x = x ^ (x >> 16);
    return x;
}

// Each warp processes 2 tiles (T=16).
// Threads per block should be a multiple of 32 (e.g., 256 -> 8 warps)
__global__ void tyche_v5b_bijective_kernel(
    uint32_t* out,
    const uint32_t* key,
    const uint32_t* weight_matrices,
    int offset,
    int num_tiles, // total tiles
    int T,
    int R,
    int embedding_type,
    const uint32_t* key_mix_ptr,
    unsigned long long* debug_cycles
) {
    if (T != 16) return;

    uint32_t key_mix = *key_mix_ptr;

    // We process 2 tiles per warp.
    int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    
    // Each warp computes tile pair: (warp_idx * 2, warp_idx * 2 + 1)
    int tile_idx_base = warp_idx * 2;
    if (tile_idx_base >= num_tiles) return;
    
    uint32_t tile_L_idx = (uint32_t)offset + (uint32_t)tile_idx_base;
    uint32_t tile_R_idx = (uint32_t)offset + (uint32_t)tile_idx_base + 1;
    
    // Shared memory for 32-bit (for acc_frag) and 8-bit (for a_frag/b_frag)
    extern __shared__ uint32_t smem_base_v5b_32[];
    int warp_in_block = threadIdx.x / 32;
    // Each warp needs 2 x 256 uint32s = 512 uint32s
    uint32_t* my_smem_u32_L = smem_base_v5b_32 + warp_in_block * 512;
    uint32_t* my_smem_u32_R = my_smem_u32_L + 256;
    
    // Put the 8-bit array after the 32-bit arrays of ALL warps in the block
    int num_warps = blockDim.x / 32;
    // Each warp needs 1 x 256 int8s for loading into A/B
    int8_t* my_smem_i8 = (int8_t*)(smem_base_v5b_32 + num_warps * 512) + warp_in_block * 256;
    
    // 1. Initial make_tile (Left and Right)
    for (int i = 0; i < 8; ++i) {
        int idx = i * 32 + lane;
        int r = idx / 16;
        int c = idx % 16;
        
        uint32_t vL = 0;
        uint32_t vR = 0;
        if (embedding_type == 1) { // diagonal
            vL = (r == c) ? fast_mix_u32_v5b(key_mix ^ tile_L_idx) : 0;
            vR = (r == c) ? fast_mix_u32_v5b(key_mix ^ tile_R_idx) : 0;
        } else if (embedding_type == 2) { // row
            vL = fast_mix_u32_v5b(key_mix ^ tile_L_idx ^ ((uint32_t)r * 1234567891u));
            vR = fast_mix_u32_v5b(key_mix ^ tile_R_idx ^ ((uint32_t)r * 1234567891u));
        } else if (embedding_type == 3) { // rank1
            uint32_t v1L = fast_mix_u32_v5b(key_mix ^ tile_L_idx ^ (uint32_t)r);
            uint32_t v2L = fast_mix_u32_v5b(key_mix ^ tile_L_idx ^ (uint32_t)c);
            vL = v1L * v2L;
            
            uint32_t v1R = fast_mix_u32_v5b(key_mix ^ tile_R_idx ^ (uint32_t)r);
            uint32_t v2R = fast_mix_u32_v5b(key_mix ^ tile_R_idx ^ (uint32_t)c);
            vR = v1R * v2R;
        } else { // hash (0)
            vL = key_mix ^ (tile_L_idx * 2654435761u);
            vL = vL ^ ((uint32_t)r * 1234567891u);
            vL = vL ^ ((uint32_t)c * 987654321u);
            vL = fast_mix_u32_v5b(vL);
            
            vR = key_mix ^ (tile_R_idx * 2654435761u);
            vR = vR ^ ((uint32_t)r * 1234567891u);
            vR = vR ^ ((uint32_t)c * 987654321u);
            vR = fast_mix_u32_v5b(vR);
        }
        
        my_smem_u32_L[idx] = vL;
        my_smem_u32_R[idx] = vR;
    }
    
    __syncwarp();
    
    // 2. Feistel Tensor Core
    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::row_major> b_frag; 
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> acc_frag_L;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> acc_frag_R;
    
    bool is_debug_thread = (blockIdx.x == 0 && threadIdx.x == 0);
    long long t0, t1, t2;
    unsigned long long smem_cycles = 0;
    unsigned long long wmma_cycles = 0;
    unsigned long long butterfly_cycles = 0;
    
    if (is_debug_thread) t0 = clock64();
    // --- ROUND 1: L' = L + (R * R) ---
    // Vectorized Cast: 32 threads pack 256 elements in chunks of 4.
    // 2 iterations * 32 threads * 4 bytes = 256 bytes total.
    for (int i = 0; i < 2; ++i) {
        int base = i * 128 + lane * 4;
        uint32_t v0 = my_smem_u32_R[base + 0];
        uint32_t v1 = my_smem_u32_R[base + 1];
        uint32_t v2 = my_smem_u32_R[base + 2];
        uint32_t v3 = my_smem_u32_R[base + 3];
        uint32_t packed = (v0 & 0xFF) | ((v1 & 0xFF) << 8) | ((v2 & 0xFF) << 16) | ((v3 & 0xFF) << 24);
        ((uint32_t*)my_smem_i8)[i * 32 + lane] = packed;
    }
    __syncwarp();
    
    wmma::load_matrix_sync(a_frag, my_smem_i8, 16);
    wmma::load_matrix_sync(b_frag, my_smem_i8, 16);
    wmma::load_matrix_sync(acc_frag_L, (const int32_t*)my_smem_u32_L, 16, wmma::mem_row_major);
    
    if (is_debug_thread) t1 = clock64();
    // Compute L' = L + R * R
    wmma::mma_sync(acc_frag_L, a_frag, b_frag, acc_frag_L);
    if (is_debug_thread) { t2 = clock64(); smem_cycles += (t1 - t0); wmma_cycles += (t2 - t1); }
    
    // Store L' back to shared memory so we can truncate it for Round 2
    wmma::store_matrix_sync((int32_t*)my_smem_u32_L, acc_frag_L, 16, wmma::mem_row_major);
    __syncwarp();
    
    if (is_debug_thread) t0 = clock64();
    // --- ROUND 2: R' = R + (L' * L') ---
    // Vectorized Cast: Truncate L' to 8-bit and store in shared memory
    for (int i = 0; i < 2; ++i) {
        int base = i * 128 + lane * 4;
        uint32_t v0 = my_smem_u32_L[base + 0];
        uint32_t v1 = my_smem_u32_L[base + 1];
        uint32_t v2 = my_smem_u32_L[base + 2];
        uint32_t v3 = my_smem_u32_L[base + 3];
        uint32_t packed = (v0 & 0xFF) | ((v1 & 0xFF) << 8) | ((v2 & 0xFF) << 16) | ((v3 & 0xFF) << 24);
        ((uint32_t*)my_smem_i8)[i * 32 + lane] = packed;
    }
    __syncwarp();
    
    wmma::load_matrix_sync(a_frag, my_smem_i8, 16);
    wmma::load_matrix_sync(b_frag, my_smem_i8, 16);
    wmma::load_matrix_sync(acc_frag_R, (const int32_t*)my_smem_u32_R, 16, wmma::mem_row_major);
    
    if (is_debug_thread) t1 = clock64();
    // Compute R' = R + L' * L'
    wmma::mma_sync(acc_frag_R, a_frag, b_frag, acc_frag_R);
    if (is_debug_thread) { t2 = clock64(); smem_cycles += (t1 - t0); wmma_cycles += (t2 - t1); }
    
    if (is_debug_thread) t0 = clock64();
    // 3. Feistel Butterfly Network
    for (int i = 0; i < 8; ++i) {
        uint32_t vL = (uint32_t)acc_frag_L.x[i];
        uint32_t vR = (uint32_t)acc_frag_R.x[i];
        
        uint32_t v_other;
        
        // Stage 1 (swap distance 1) - LINEAR ONLY
        v_other = __shfl_xor_sync(0xffffffff, vR, 1);
        vL = vL + (v_other ^ 0x9E3779B9u) + lane;
        v_other = __shfl_xor_sync(0xffffffff, vL, 1);
        vR = vR + (v_other ^ 0x9E3779B9u) + lane;
        
        // Stage 2 (swap distance 2) - LINEAR ONLY
        v_other = __shfl_xor_sync(0xffffffff, vR, 2);
        vL = vL + (v_other ^ 0x85EBCA6Bu) + lane;
        v_other = __shfl_xor_sync(0xffffffff, vL, 2);
        vR = vR + (v_other ^ 0x85EBCA6Bu) + lane;
        
        // Stage 3 (swap distance 4) - NON-LINEAR MIX
        v_other = __shfl_xor_sync(0xffffffff, vR, 4);
        vL = fast_mix_u32_v5b(vL + (v_other ^ 0xC2B2AE35u) + lane);
        v_other = __shfl_xor_sync(0xffffffff, vL, 4);
        vR = fast_mix_u32_v5b(vR + (v_other ^ 0xC2B2AE35u) + lane);
        
        // Stage 4 (swap distance 8) - LINEAR ONLY
        v_other = __shfl_xor_sync(0xffffffff, vR, 8);
        vL = vL + (v_other ^ 0x27D4EB2Fu) + lane;
        v_other = __shfl_xor_sync(0xffffffff, vL, 8);
        vR = vR + (v_other ^ 0x27D4EB2Fu) + lane;
        
        // Stage 5 (swap distance 16) - NON-LINEAR MIX
        v_other = __shfl_xor_sync(0xffffffff, vR, 16);
        vL = fast_mix_u32_v5b(vL + (v_other ^ 0x165667B1u) + lane);
        v_other = __shfl_xor_sync(0xffffffff, vL, 16);
        vR = fast_mix_u32_v5b(vR + (v_other ^ 0x165667B1u) + lane);
        
        // Write out
        int out_idx = i * 32 + lane;
        // L tile goes to tile_idx_base
        out[tile_idx_base * 256 + out_idx] = vL;
        // R tile goes to tile_idx_base + 1
        if (tile_idx_base + 1 < num_tiles) {
            out[(tile_idx_base + 1) * 256 + out_idx] = vR;
        }
    }
    
    if (is_debug_thread) { 
        t1 = clock64(); 
        butterfly_cycles = (t1 - t0); 
        debug_cycles[0] = smem_cycles;
        debug_cycles[1] = wmma_cycles;
        debug_cycles[2] = butterfly_cycles;
    }
}

void tyche_v5b_bijective_kernel_launch(
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
    // Each warp processes 2 tiles!
    int tiles_per_block = warps_per_block * 2; // 16
    int blocks = (num_tiles + tiles_per_block - 1) / tiles_per_block;
    
    // Shared memory: 2048 bytes for uint32 + 256 bytes for int8 = 2304 bytes per warp
    size_t smem_size = warps_per_block * 2304;
    
    unsigned long long* d_debug_cycles = nullptr;
    cudaMalloc(&d_debug_cycles, 3 * sizeof(unsigned long long));
    cudaMemset(d_debug_cycles, 0, 3 * sizeof(unsigned long long));
    
    if (blocks > 0) {
        tyche_v5b_bijective_kernel<<<blocks, threads, smem_size, stream>>>(
            out, key, weight_matrices, offset, num_tiles, T, R, embedding_type, key_mix_ptr, d_debug_cycles
        );
    }
    
    // Only print debug cycles if this is the default stream (or just synchronize and print anyway for profiling)
    // We synchronize stream so we can accurately read back the memory. This breaks async execution but it's an MVP profiling.
    cudaStreamSynchronize(stream);
    
    unsigned long long h_debug[3];
    cudaMemcpy(h_debug, d_debug_cycles, 3 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_debug_cycles);
    
    std::cout << "\n=== PROFILING KERNEL V5B (clock64) ===" << std::endl;
    std::cout << "SMEM Cast & Load Latency: " << h_debug[0] << " cycles" << std::endl;
    std::cout << "WMMA mma_sync Latency:    " << h_debug[1] << " cycles" << std::endl;
    std::cout << "Butterfly ALU Latency:    " << h_debug[2] << " cycles" << std::endl;
    std::cout << "======================================" << std::endl;
}
