#include "kernel.h"
#include <cstdint>

__global__ void dummy_fill_kernel(uint32_t* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = 42;
    }
}

void dummy_fill(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len) {
    uint32_t* out = reinterpret_cast<uint32_t*>(buffers[0]);
    int size = *reinterpret_cast<const int*>(opaque);
    
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    dummy_fill_kernel<<<blocks, threads, 0, stream>>>(out, size);
}
