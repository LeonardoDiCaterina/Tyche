#include <iostream>
#include <cuda_runtime.h>
#include <stdint.h>

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

int main() {
    std::cout << "========================================================\n";
    std::cout << "PHASE 0: PURE CUDA ALU BASELINE\n";
    std::cout << "========================================================\n";
    run_phase0();
    
    return 0;
}
