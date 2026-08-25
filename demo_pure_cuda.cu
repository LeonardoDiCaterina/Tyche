#include <iostream>
#include <cuda_runtime.h>
#include <stdint.h>
#include <mma.h>

using namespace nvcuda;

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// ============================================================================
// PHASE 0: ALU BASELINE (Simplified Threefry)
// ============================================================================
__device__ void threefry2x32_4(uint32_t k0, uint32_t k1, uint32_t p0, uint32_t p1, uint32_t& out0, uint32_t& out1) {
    uint32_t v0 = p0 + k0;
    uint32_t v1 = p1 + k1;
    
    // Minimal 4-round Threefry for raw speed demonstration
    for (int i = 0; i < 4; ++i) {
        v0 += v1;
        v1 = (v1 << 16) | (v1 >> 16);
        v1 ^= v0;
        
        v0 += v1;
        v1 = (v1 << 17) | (v1 >> 15);
        v1 ^= v0;
    }
    
    out0 = v0;
    out1 = v1;
}

__global__ void threefry_pi_kernel(unsigned long long* out_hits, unsigned long long num_points) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread processes 256 points to heavily amortize grid overhead
    unsigned long long points_per_thread = 256;
    if (idx * points_per_thread >= num_points) return;
    
    unsigned int hits = 0;
    uint32_t out0, out1;
    
    for(int i = 0; i < points_per_thread; ++i) {
        threefry2x32_4(0x12345678, 0x9ABCDEF0, idx, i, out0, out1);
        
        // Extract 8-bit values to stay perfectly comparable with Tensor Core int8 output
        int8_t x = (int8_t)out0;
        int8_t y = (int8_t)out1;
        
        // Calculate X^2 + Y^2 <= 127^2 (16129)
        int r2 = (int)x * x + (int)y * y;
        if (r2 <= 16129) {
            hits++;
        }
    }
    
    atomicAdd(out_hits, (unsigned long long)hits);
}

void run_phase0() {
    // Phase 0 Baseline uses 1 Billion points
    unsigned long long total_points = 1000000000ULL; 
    int threads_per_block = 256;
    int points_per_thread = 256;
    unsigned long long total_threads = (total_points + points_per_thread - 1) / points_per_thread;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    unsigned long long* d_hits;
    CHECK_CUDA(cudaMalloc(&d_hits, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemset(d_hits, 0, sizeof(unsigned long long)));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::cout << "Launching Threefry ALU Kernel with " << total_points << " points...\n";
    
    // Warmup
    threefry_pi_kernel<<<blocks, threads_per_block>>>(d_hits, total_points);
    CHECK_CUDA(cudaMemset(d_hits, 0, sizeof(unsigned long long)));
    
    cudaEventRecord(start);
    threefry_pi_kernel<<<blocks, threads_per_block>>>(d_hits, total_points);
    cudaEventRecord(stop);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    unsigned long long h_hits;
    CHECK_CUDA(cudaMemcpy(&h_hits, d_hits, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    
    double pi_estimate = 4.0 * (double)h_hits / (double)total_points;
    
    std::cout << "  Execution Time: " << milliseconds << " ms\n";
    std::cout << "  Pi Estimate:    " << pi_estimate << "\n";
    
    cudaFree(d_hits);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// PHASE 1: TENSOR CORE WMMA SCAFFOLDING
// ============================================================================
__global__ void tyche_wmma_pi_kernel(unsigned long long* out_hits, unsigned long long num_points) {
    // Each warp computes a 16x16 matrix (256 points)
    int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    
    unsigned long long points_per_warp = 256;
    if ((unsigned long long)warp_idx * points_per_warp >= num_points) return;
    
    // Allocate shared memory for 16x16 int8 matrices A and B for this warp
    // 256 bytes per matrix * 2 = 512 bytes per warp
    extern __shared__ int8_t smem_base[];
    int warp_in_block = threadIdx.x / 32;
    int8_t* my_smem_A = smem_base + warp_in_block * 512;
    int8_t* my_smem_B = my_smem_A + 256;
    
    // Fill with dummy data
    for (int i = 0; i < 8; ++i) {
        my_smem_A[i * 32 + lane] = 1;
        my_smem_B[i * 32 + lane] = 2;
    }
    __syncwarp();
    
    // Declare WMMA fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::row_major> b_frag; 
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> acc_frag;
    
    wmma::fill_fragment(acc_frag, 0);
    wmma::load_matrix_sync(a_frag, my_smem_A, 16);
    wmma::load_matrix_sync(b_frag, my_smem_B, 16);
    
    // Dummy mma execution
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    
    // Tally hits to avoid compiler optimizing away the math
    unsigned int hits = 0;
    for (int i = 0; i < 8; ++i) {
        if (acc_frag.x[i] == 32) { // 16 * 1 * 2 = 32
            hits++;
        }
    }
    
    atomicAdd(out_hits, (unsigned long long)hits);
}

void run_phase1() {
    unsigned long long total_points = 1000000000ULL; 
    int threads_per_block = 256; // 8 warps per block
    int warps_per_block = threads_per_block / 32;
    unsigned long long points_per_warp = 256; // 16x16 output block
    
    unsigned long long total_warps = (total_points + points_per_warp - 1) / points_per_warp;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    
    size_t smem_size = warps_per_block * 512; // 512 bytes per warp
    
    unsigned long long* d_hits;
    CHECK_CUDA(cudaMalloc(&d_hits, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemset(d_hits, 0, sizeof(unsigned long long)));
    
    std::cout << "Launching Tensor Core WMMA Dummy Kernel...\n";
    
    tyche_wmma_pi_kernel<<<blocks, threads_per_block, smem_size>>>(d_hits, total_points);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "  Tensor Core launch successful! No memory crashes.\n";
    
    cudaFree(d_hits);
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "PHASE 0: PURE CUDA ALU BASELINE\n";
    std::cout << "========================================================\n";
    run_phase0();
    
    std::cout << "\n========================================================\n";
    std::cout << "PHASE 1: TENSOR CORE WMMA SCAFFOLDING\n";
    std::cout << "========================================================\n";
    run_phase1();
    
    return 0;
}
