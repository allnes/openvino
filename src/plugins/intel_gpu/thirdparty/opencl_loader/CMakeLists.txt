# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

add_library(openvino_portable_opencl_loader STATIC opencl_loader.cpp)
add_library(openvino::portable_opencl_loader ALIAS openvino_portable_opencl_loader)

target_link_libraries(openvino_portable_opencl_loader
    PUBLIC OpenCL::NewHeaders
    PRIVATE ${CMAKE_DL_LIBS})
target_compile_definitions(openvino_portable_opencl_loader INTERFACE
    OV_GPU_OPENCL_HPP_HAS_UUID
    OV_GPU_OPENCL_HPP_HAS_BUS_INFO)

set_target_properties(openvino_portable_opencl_loader PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
