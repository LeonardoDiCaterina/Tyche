#include <pybind11/pybind11.h>
#include "kernel.h"

namespace py = pybind11;

template <typename T>
py::capsule EncapsulateFunction(T* fn) {
  return py::capsule(reinterpret_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}

struct TycheV1ConfigOpaque {
    int offset;
    int num_tiles;
    int T;
    int R;
    int embedding_type;
};

void tyche_v1_hash(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len, void* status) {
    if (opaque_len != sizeof(TycheV1ConfigOpaque)) return;
    const TycheV1ConfigOpaque* config = reinterpret_cast<const TycheV1ConfigOpaque*>(opaque);
    
    const uint32_t* key = reinterpret_cast<const uint32_t*>(buffers[0]);
    const uint32_t* weight_matrices = reinterpret_cast<const uint32_t*>(buffers[1]);
    const uint32_t* key_mix_ptr = reinterpret_cast<const uint32_t*>(buffers[2]);
    int8_t* out = reinterpret_cast<int8_t*>(buffers[3]);
    
    tyche_v1_kernel_launch(
        stream, out, key, weight_matrices,
        config->offset, config->num_tiles, config->T, config->R, config->embedding_type, key_mix_ptr
    );
}

void tyche_v2_hash(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len, void* status) {
    if (opaque_len != sizeof(TycheV1ConfigOpaque)) return;
    const TycheV1ConfigOpaque* config = reinterpret_cast<const TycheV1ConfigOpaque*>(opaque);
    
    // Inputs come first in legacy custom call API
    const uint32_t* key = reinterpret_cast<const uint32_t*>(buffers[0]);
    const uint32_t* weight_matrices = reinterpret_cast<const uint32_t*>(buffers[1]);
    const uint32_t* key_mix_ptr = reinterpret_cast<const uint32_t*>(buffers[2]);
    int8_t* out = reinterpret_cast<int8_t*>(buffers[3]);
    
    tyche_v2_wmma_kernel_launch(
        stream, out, key, weight_matrices,
        config->offset, config->num_tiles, config->T, config->R, config->embedding_type, key_mix_ptr
    );
}

void tyche_v5_hash(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len, void* status) {
    if (opaque_len != sizeof(TycheV1ConfigOpaque)) return;
    const TycheV1ConfigOpaque* config = reinterpret_cast<const TycheV1ConfigOpaque*>(opaque);
    
    // Inputs come first in legacy custom call API
    const uint32_t* key = reinterpret_cast<const uint32_t*>(buffers[0]);
    const uint32_t* weight_matrices = reinterpret_cast<const uint32_t*>(buffers[1]);
    const uint32_t* key_mix_ptr = reinterpret_cast<const uint32_t*>(buffers[2]);
    uint32_t* out = reinterpret_cast<uint32_t*>(buffers[3]); // V5 writes uint32_t!
    
    tyche_v5_hybrid_kernel_launch(
        stream, out, key, weight_matrices,
        config->offset, config->num_tiles, config->T, config->R, config->embedding_type, key_mix_ptr
    );
}

void tyche_v5b_hash(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len, void* status) {
    if (opaque_len != sizeof(TycheV1ConfigOpaque)) return;
    const TycheV1ConfigOpaque* config = reinterpret_cast<const TycheV1ConfigOpaque*>(opaque);
    
    const uint32_t* key = reinterpret_cast<const uint32_t*>(buffers[0]);
    const uint32_t* weight_matrices = reinterpret_cast<const uint32_t*>(buffers[1]);
    const uint32_t* key_mix_ptr = reinterpret_cast<const uint32_t*>(buffers[2]);
    uint32_t* out = reinterpret_cast<uint32_t*>(buffers[3]);
    
    tyche_v5b_bijective_kernel_launch(
        stream, out, key, weight_matrices,
        config->offset, config->num_tiles, config->T, config->R, config->embedding_type, key_mix_ptr
    );
}

PYBIND11_MODULE(tyche_csrc, m) {
    m.def("registrations", []() {
        py::dict dict;
        dict["tyche_dummy"] = EncapsulateFunction(dummy_fill);
        dict["tyche_v1_hash"] = EncapsulateFunction(tyche_v1_hash);
        dict["tyche_v2_hash"] = EncapsulateFunction(tyche_v2_hash);
        dict["tyche_v5_hash"] = EncapsulateFunction(tyche_v5_hash);
        dict["tyche_v5b_hash"] = EncapsulateFunction(tyche_v5b_hash);
        return dict;
    });
}
