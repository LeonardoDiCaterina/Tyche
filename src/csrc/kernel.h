#pragma once
#include <cuda_runtime.h>

void dummy_fill(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len);
