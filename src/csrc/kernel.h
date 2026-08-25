#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

void dummy_fill_kernel(cudaStream_t stream, uint32_t* out, int size);

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
    const uint32_t* key_mix_ptr
);

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
);

void dummy_fill(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len);
