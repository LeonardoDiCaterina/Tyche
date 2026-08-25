#include <pybind11/pybind11.h>
#include "kernel.h"

namespace py = pybind11;

template <typename T>
py::capsule EncapsulateFunction(T* fn) {
  return py::capsule(reinterpret_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}

PYBIND11_MODULE(tyche_csrc, m) {
    m.def("registrations", []() {
        py::dict dict;
        dict["tyche_dummy"] = EncapsulateFunction(dummy_fill);
        return dict;
    });
}
