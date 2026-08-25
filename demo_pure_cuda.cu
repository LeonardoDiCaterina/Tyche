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

#define FAST_MUL1 0xBF58476Du
#define FAST_MUL2 0x94D049BBu

__device__ uint32_t fast_mix_u32(uint32_t x) {
    x = (x ^ (x >> 16)) * FAST_MUL1;
    x = (x ^ (x >> 13)) * FAST_MUL2;
    x = x ^ (x >> 16);
    return x;
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
    extern __shared__ char smem_base[];
    int warp_in_block = threadIdx.x / 32;
    int8_t* my_smem_A = (int8_t*)(smem_base + warp_in_block * 512);
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

// ============================================================================
// PHASE 2: FULL TYCHE PRNG & 8-BIT FUSION
// ============================================================================
__global__ void tyche_wmma_pi_kernel_full(unsigned long long* out_hits, unsigned long long num_points, uint32_t key_mix) {
    int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    
    unsigned long long points_per_warp = 256;
    if ((unsigned long long)warp_idx * points_per_warp >= num_points) return;
    
    extern __shared__ char smem_base[];
    int warp_in_block = threadIdx.x / 32;
    
    // Allocate 512 uint32s per warp (2048 bytes) for L and R
    uint32_t* my_smem_vL = (uint32_t*)(smem_base + warp_in_block * 2048);
    uint32_t* my_smem_vR = my_smem_vL + 256;
    
    // Put int8 shared memory after ALL uint32 shared memory for the block
    int num_warps = blockDim.x / 32;
    int8_t* my_smem_i8 = (int8_t*)(smem_base + num_warps * 2048 + warp_in_block * 256);
    
    uint32_t tile_L_idx = warp_idx * 2;
    uint32_t tile_R_idx = warp_idx * 2 + 1;
    
    for (int i = 0; i < 8; ++i) {
        int idx = i * 32 + lane;
        int r = idx / 16;
        int c = idx % 16;
        
        uint32_t vL = key_mix ^ (tile_L_idx * 2654435761u) ^ (r * 1234567891u) ^ (c * 987654321u);
        uint32_t vR = key_mix ^ (tile_R_idx * 2654435761u) ^ (r * 1234567891u) ^ (c * 987654321u);
        
        my_smem_vL[idx] = fast_mix_u32(vL);
        my_smem_vR[idx] = fast_mix_u32(vR);
    }
    __syncwarp();
    
    // Cast vR to int8
    for (int i = 0; i < 8; ++i) {
        int idx = i * 32 + lane;
        my_smem_i8[idx] = (int8_t)my_smem_vR[idx];
    }
    __syncwarp();
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::row_major> b_frag; 
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> acc_frag_L;
    
    wmma::load_matrix_sync(a_frag, my_smem_i8, 16);
    wmma::load_matrix_sync(b_frag, my_smem_i8, 16);
    wmma::load_matrix_sync(acc_frag_L, (const int32_t*)my_smem_vL, 16, wmma::mem_row_major);
    
    // Tensor Core Math
    wmma::mma_sync(acc_frag_L, a_frag, b_frag, acc_frag_L);
    
    // Extract 8-bit pieces directly from int32 accumulator registers
    unsigned int hits = 0;
    for (int i = 0; i < 8; ++i) {
        int32_t val = acc_frag_L.x[i];
        
        int8_t x = (int8_t)(val & 0xFF);
        int8_t y = (int8_t)((val >> 8) & 0xFF);
        
        int r2 = (int)x * x + (int)y * y;
        if (r2 <= 16129) {
            hits++;
        }
    }
    
    atomicAdd(out_hits, (unsigned long long)hits);
}

void run_phase2() {
    unsigned long long total_points = 10000000000ULL; // 10 BILLION POINTS
    std::cout << "Workload: " << total_points << " points.\n";
    
    // --- 1. RUN THREEFRY ALU ---
    int threads_per_block = 256;
    int points_per_thread = 256;
    unsigned long long total_threads = (total_points + points_per_thread - 1) / points_per_thread;
    int blocks_alu = (total_threads + threads_per_block - 1) / threads_per_block;
    
    unsigned long long* d_hits_alu;
    CHECK_CUDA(cudaMalloc(&d_hits_alu, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemset(d_hits_alu, 0, sizeof(unsigned long long)));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup ALU
    threefry_pi_kernel<<<blocks_alu, threads_per_block>>>(d_hits_alu, total_points);
    CHECK_CUDA(cudaMemset(d_hits_alu, 0, sizeof(unsigned long long)));
    
    cudaEventRecord(start);
    threefry_pi_kernel<<<blocks_alu, threads_per_block>>>(d_hits_alu, total_points);
    cudaEventRecord(stop);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    float ms_alu = 0;
    cudaEventElapsedTime(&ms_alu, start, stop);
    
    // --- 2. RUN TYCHE WMMA ---
    int warps_per_block = threads_per_block / 32;
    unsigned long long points_per_warp = 256; 
    unsigned long long total_warps = (total_points + points_per_warp - 1) / points_per_warp;
    int blocks_wmma = (total_warps + warps_per_block - 1) / warps_per_block;
    
    // 2048 bytes for uint32 + 256 bytes for int8 = 2304 bytes per warp
    size_t smem_size = warps_per_block * 2304; 
    
    unsigned long long* d_hits_wmma;
    CHECK_CUDA(cudaMalloc(&d_hits_wmma, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemset(d_hits_wmma, 0, sizeof(unsigned long long)));
    
    uint32_t key_mix = 123456789;
    
    // Warmup WMMA
    tyche_wmma_pi_kernel_full<<<blocks_wmma, threads_per_block, smem_size>>>(d_hits_wmma, total_points, key_mix);
    CHECK_CUDA(cudaMemset(d_hits_wmma, 0, sizeof(unsigned long long)));
    
    cudaEventRecord(start);
    tyche_wmma_pi_kernel_full<<<blocks_wmma, threads_per_block, smem_size>>>(d_hits_wmma, total_points, key_mix);
    cudaEventRecord(stop);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    float ms_wmma = 0;
    cudaEventElapsedTime(&ms_wmma, start, stop);
    
    std::cout << "\nRESULTS:\n";
    std::cout << "  Threefry (ALU):         " << ms_alu << " ms\n";
    std::cout << "  Tyche (Tensor Cores):   " << ms_wmma << " ms\n";
    std::cout << "  Speedup:                " << ms_alu / ms_wmma << "x\n";
    
    if (ms_alu / ms_wmma > 1.0) {
        std::cout << "\nVICTORY! By avoiding int8 casting overhead and keeping the state in 8-bit registers, Tyche's Tensor Cores successfully crushed Threefry!\n";
    } else {
        std::cout << "\nDEFEAT! Even without casting overhead, Tensor Core setup latency and shared memory movement is too high. Threefry remains the undisputed champion.\n";
    }
    
    cudaFree(d_hits_alu);
    cudaFree(d_hits_wmma);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// PHASE 3: 1-BIT BINARY TENSOR CORE "HAIL MARY"
// ============================================================================
__global__ void tyche_b1_pi_kernel(unsigned long long* out_hits, unsigned long long num_points, uint32_t key_mix) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    
    // b1 mode output is 8x8 = 64 points per warp
    unsigned long long points_per_warp = 64;
    if ((unsigned long long)warp_idx * points_per_warp >= num_points) return;
    
    extern __shared__ char smem_base[];
    int warp_in_block = threadIdx.x / 32;
    
    // We need an 8x128 matrix of bits. 128 bits = 4 uint32s.
    // 8 rows * 4 uint32s = 32 uint32s = 128 bytes.
    // Two matrices (A and B) = 256 bytes per warp.
    uint32_t* my_smem_A = (uint32_t*)(smem_base + warp_in_block * 256);
    uint32_t* my_smem_B = my_smem_A + 32;
    
    uint32_t tile_L_idx = warp_idx * 2;
    uint32_t tile_R_idx = warp_idx * 2 + 1;
    
    // 32 threads exactly fill the 32 uint32s for A and B. No loops!
    uint32_t r = lane / 4; 
    uint32_t c = lane % 4; 
    
    uint32_t vL = key_mix ^ (tile_L_idx * 2654435761u) ^ (r * 1234567891u) ^ (c * 987654321u);
    uint32_t vR = key_mix ^ (tile_R_idx * 2654435761u) ^ (r * 1234567891u) ^ (c * 987654321u);
    
    // Zero casting. Raw 32-bit registers directly to shared memory.
    my_smem_A[lane] = fast_mix_u32(vL);
    my_smem_B[lane] = fast_mix_u32(vR);
    
    __syncwarp();
    
    wmma::fragment<wmma::matrix_a, 8, 8, 128, wmma::experimental::precision::b1, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 128, wmma::experimental::precision::b1, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 128, int32_t> acc_frag;
    
    wmma::fill_fragment(acc_frag, 0);
    wmma::load_matrix_sync(a_frag, my_smem_A, 128);
    wmma::load_matrix_sync(b_frag, my_smem_B, 128);
    
    // Binary Tensor Core execution: C = POPC(A XOR B)
    wmma::bmma_sync(acc_frag, a_frag, b_frag, acc_frag, wmma::bmmaBitOpXOR, wmma::bmmaAccumulateOpPOPC);
    
    // Extract 8-bit pieces directly from int32 accumulator registers
    unsigned int hits = 0;
    // Each thread holds exactly 2 accumulator elements for an 8x8 matrix (64 total / 32 threads = 2)
    for (int i = 0; i < 2; ++i) {
        int32_t val = acc_frag.x[i];
        
        int8_t x = (int8_t)(val & 0xFF);
        int8_t y = (int8_t)((val >> 8) & 0xFF);
        
        int r2 = (int)x * x + (int)y * y;
        if (r2 <= 16129) {
            hits++;
        }
    }
    
    atomicAdd(out_hits, (unsigned long long)hits);
#else
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("ERROR: b1 Tensor Cores require Compute Capability >= 7.5!\n");
    }
#endif
}

void run_phase3() {
    unsigned long long total_points = 10000000000ULL; // 10 BILLION POINTS
    std::cout << "Workload: " << total_points << " points.\n";
    
    // --- 1. RUN THREEFRY ALU ---
    int threads_per_block = 256;
    int points_per_thread = 256;
    unsigned long long total_threads = (total_points + points_per_thread - 1) / points_per_thread;
    int blocks_alu = (total_threads + threads_per_block - 1) / threads_per_block;
    
    unsigned long long* d_hits_alu;
    CHECK_CUDA(cudaMalloc(&d_hits_alu, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemset(d_hits_alu, 0, sizeof(unsigned long long)));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup ALU
    threefry_pi_kernel<<<blocks_alu, threads_per_block>>>(d_hits_alu, total_points);
    CHECK_CUDA(cudaMemset(d_hits_alu, 0, sizeof(unsigned long long)));
    
    cudaEventRecord(start);
    threefry_pi_kernel<<<blocks_alu, threads_per_block>>>(d_hits_alu, total_points);
    cudaEventRecord(stop);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    float ms_alu = 0;
    cudaEventElapsedTime(&ms_alu, start, stop);
    
    // --- 2. RUN TYCHE BINARY WMMA ---
    int warps_per_block = threads_per_block / 32;
    unsigned long long points_per_warp = 64; 
    unsigned long long total_warps = (total_points + points_per_warp - 1) / points_per_warp;
    int blocks_wmma = (total_warps + warps_per_block - 1) / warps_per_block;
    
    size_t smem_size = warps_per_block * 256; 
    
    unsigned long long* d_hits_wmma;
    CHECK_CUDA(cudaMalloc(&d_hits_wmma, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemset(d_hits_wmma, 0, sizeof(unsigned long long)));
    
    uint32_t key_mix = 123456789;
    
    // Warmup WMMA
    tyche_b1_pi_kernel<<<blocks_wmma, threads_per_block, smem_size>>>(d_hits_wmma, total_points, key_mix);
    CHECK_CUDA(cudaMemset(d_hits_wmma, 0, sizeof(unsigned long long)));
    
    cudaEventRecord(start);
    tyche_b1_pi_kernel<<<blocks_wmma, threads_per_block, smem_size>>>(d_hits_wmma, total_points, key_mix);
    cudaEventRecord(stop);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    float ms_wmma = 0;
    cudaEventElapsedTime(&ms_wmma, start, stop);
    
    std::cout << "\nRESULTS:\n";
    std::cout << "  Threefry (ALU):         " << ms_alu << " ms\n";
    std::cout << "  Tyche (b1 Tensor Core): " << ms_wmma << " ms\n";
    std::cout << "  Speedup:                " << ms_alu / ms_wmma << "x\n";
    
    cudaFree(d_hits_alu);
    cudaFree(d_hits_wmma);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "PHASE 0: PURE CUDA ALU BASELINE\n";
    std::cout << "========================================================\n";
    run_phase0();
    
    std::cout << "\n========================================================\n";
    std::cout << "PHASE 2: THE ULTIMATE HARDWARE SHOWDOWN (10 BILLION)\n";
    std::cout << "========================================================\n";
    run_phase2();
    
    std::cout << "\n========================================================\n";
    std::cout << "PHASE 3: THE BINARY TENSOR CORE HAIL MARY (10 BILLION)\n";
    std::cout << "========================================================\n";
    run_phase3();
    
    return 0;
}
