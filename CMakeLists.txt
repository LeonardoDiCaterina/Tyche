cmake_minimum_required(VERSION 3.25...3.28)
project(tyche_csrc LANGUAGES CXX CUDA)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(tyche_csrc MODULE
    src/csrc/ffi.cc
    src/csrc/kernel_v1.cu
    src/csrc/kernel_wmma.cu
    src/csrc/kernel_v5_hybrid.cu
    src/csrc/kernel_v5b_bijective.cu
)

set_target_properties(tyche_csrc PROPERTIES
    CUDA_ARCHITECTURES "90"
    CXX_STANDARD 17
)

target_include_directories(tyche_csrc PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

install(TARGETS tyche_csrc DESTINATION .)
