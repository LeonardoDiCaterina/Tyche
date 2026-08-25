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
    uint32_t key_mix;
};

void tyche_v1_hash(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len) {
    if (opaque_len != sizeof(TycheV1ConfigOpaque)) return;
    const TycheV1ConfigOpaque* config = reinterpret_cast<const TycheV1ConfigOpaque*>(opaque);
    
    const uint32_t* key = reinterpret_cast<const uint32_t*>(buffers[0]);
    const uint32_t* weight_matrices = reinterpret_cast<const uint32_t*>(buffers[1]);
    int8_t* out = reinterpret_cast<int8_t*>(buffers[2]);
    
    tyche_v1_kernel_launch(
        stream, out, key, weight_matrices,
        config->offset, config->num_tiles, config->T, config->R, config->embedding_type, config->key_mix
    );
}

PYBIND11_MODULE(tyche_csrc, m) {
    m.def("registrations", []() {
        py::dict dict;
        dict["tyche_dummy"] = EncapsulateFunction(dummy_fill);
        dict["tyche_v1_hash"] = EncapsulateFunction(tyche_v1_hash);
        return dict;
    });
}
