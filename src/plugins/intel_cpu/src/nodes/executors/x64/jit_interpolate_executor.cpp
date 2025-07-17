// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_interpolate_executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "openvino/core/parallel.hpp"
#include "utils/bfloat16.hpp"
#include "utils/general_utils.h"

#if defined(OPENVINO_ARCH_X86_64)
#include <cpu/x64/cpu_isa_traits.hpp>
#include "cpu_types.h"
#endif

namespace ov::intel_cpu {

JitInterpolateExecutor::JitInterpolateExecutor(ExecutorContext::CPtr context)
    : InterpolateExecutor(context) {}

bool JitInterpolateExecutor::init(const InterpolateAttrs& interpolateAttrs,
                                  const std::vector<MemoryDescPtr>& srcDescs,
                                  const std::vector<MemoryDescPtr>& dstDescs,
                                  const dnnl::primitive_attr& attr) {
    if (!InterpolateExecutor::init(interpolateAttrs, srcDescs, dstDescs, attr)) {
        return false;
    }

#if defined(OPENVINO_ARCH_X86_64)
    // Initialize JIT kernel configuration
    using namespace ov::intel_cpu::node;
    
    jit_interpolate_config_params jcp;
    jcp.mode = interpolateAttrs.mode;
    jcp.layout = interpolateAttrs.layout;
    jcp.src_prc = interpolateAttrs.inPrc;
    jcp.dst_prc = interpolateAttrs.outPrc;
    jcp.src_data_size = interpolateAttrs.inPrc.size();
    jcp.dst_data_size = interpolateAttrs.outPrc.size();
    jcp.spatial_dim_size = spatialDimSize;
    
    // Set dimensions
    jcp.C = srcDimPad5d[1];
    jcp.ID = srcDimPad5d[2];
    jcp.IH = srcDimPad5d[3];
    jcp.IW = srcDimPad5d[4];
    jcp.OD = dstDim5d[2];
    jcp.OH = dstDim5d[3];
    jcp.OW = dstDim5d[4];

    // Create JIT kernel based on available ISA
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
        interpolateKernel = std::make_shared<jit_uni_interpolate_kernel_f32<dnnl::impl::cpu::x64::avx512_core>>(jcp, attr);
    } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
        interpolateKernel = std::make_shared<jit_uni_interpolate_kernel_f32<dnnl::impl::cpu::x64::avx2>>(jcp, attr);
    } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41)) {
        interpolateKernel = std::make_shared<jit_uni_interpolate_kernel_f32<dnnl::impl::cpu::x64::sse41>>(jcp, attr);
    } else {
        return false; // No supported ISA
    }
    
    if (interpolateKernel) {
        interpolateKernel->create_ker();
    }
#endif

    return true;
}

void JitInterpolateExecutor::exec(const std::vector<MemoryCPtr>& src,
                                  const std::vector<MemoryPtr>& dst,
                                  const void* post_ops_data_) {
    const uint8_t* in_ptr_ = padPreprocess(src, dst);
    uint8_t* out_ptr_ = dst[0]->getData<uint8_t>();

    const auto& srcDim5d = srcDimPad5d;
    const auto& dstDim5d = this->dstDim5d;
    const int B = srcDim5d[0];
    const int C = srcDim5d[1];
    const int ID = srcDim5d[2];
    const int IH = srcDim5d[3];
    const int IW = srcDim5d[4];
    const int OD = dstDim5d[2];
    const int OH = dstDim5d[3];
    const int OW = dstDim5d[4];

    switch (interpAttrs.mode) {
        case InterpolateMode::nearest:
            switch (interpAttrs.layout) {
                case InterpolateLayoutType::planar:
                    NNPlanar(in_ptr_, out_ptr_, post_ops_data_, B, C, ID, IH, IW, OD, OH, OW);
                    break;
                case InterpolateLayoutType::block:
                case InterpolateLayoutType::by_channel:
                    NNCGathered(in_ptr_, out_ptr_, post_ops_data_, B, C, ID, IH, IW, OD, OH, OW);
                    break;
            }
            break;
        case InterpolateMode::linear_onnx:
            switch (interpAttrs.layout) {
                case InterpolateLayoutType::planar:
                    linearOnnxPlanar(in_ptr_, out_ptr_, post_ops_data_, B, C, ID, IH, IW, OD, OH, OW);
                    break;
                case InterpolateLayoutType::block:
                case InterpolateLayoutType::by_channel:
                    linearOnnxCGathered(in_ptr_, out_ptr_, post_ops_data_, B, C, ID, IH, IW, OD, OH, OW);
                    break;
            }
            break;
        case InterpolateMode::cubic:
            switch (interpAttrs.layout) {
                case InterpolateLayoutType::planar:
                    cubicPlanar(in_ptr_, out_ptr_, post_ops_data_, B, C, IH, IW, OH, OW);
                    break;
                case InterpolateLayoutType::block:
                case InterpolateLayoutType::by_channel:
                    cubicCGathered(in_ptr_, out_ptr_, post_ops_data_, B, C, IH, IW, OH, OW);
                    break;
            }
            break;
        case InterpolateMode::bilinear_pillow:
        case InterpolateMode::bicubic_pillow:
            switch (interpAttrs.layout) {
                case InterpolateLayoutType::by_channel:
                    pillowCGathered(in_ptr_, out_ptr_, post_ops_data_, B, C, IH, IW, OH, OW);
                    break;
                default:
                    // Unsupported layout for pillow modes
                    break;
            }
            break;
        default:
            // Unsupported mode
            break;
    }
}

impl_desc_type JitInterpolateExecutor::getImplType() const {
    return impl_desc_type::jit;
}

void JitInterpolateExecutor::NNPlanar(const uint8_t* in_ptr_, uint8_t* out_ptr_, const void* post_ops_data_, int B, int C, int ID, int IH, int IW, int OD, int OH, int OW) {
    // TODO: Move actual implementation from InterpolateJitExecutor in interpolate.cpp
    // This is a placeholder - the actual implementation will be copied from the existing code
}

void JitInterpolateExecutor::NNCGathered(const uint8_t* in_ptr_, uint8_t* out_ptr_, const void* post_ops_data_, int B, int C, int ID, int IH, int IW, int OD, int OH, int OW) {
    // TODO: Move actual implementation from InterpolateJitExecutor in interpolate.cpp
    // This is a placeholder - the actual implementation will be copied from the existing code
}

void JitInterpolateExecutor::linearOnnxPlanar(const uint8_t* in_ptr_, uint8_t* out_ptr_, const void* post_ops_data_, int B, int C, int ID, int IH, int IW, int OD, int OH, int OW) {
    // TODO: Move actual implementation from InterpolateJitExecutor in interpolate.cpp
    // This is a placeholder - the actual implementation will be copied from the existing code
}

void JitInterpolateExecutor::linearOnnxCGathered(const uint8_t* in_ptr_, uint8_t* out_ptr_, const void* post_ops_data_, int B, int C, int ID, int IH, int IW, int OD, int OH, int OW) {
    // TODO: Move actual implementation from InterpolateJitExecutor in interpolate.cpp
    // This is a placeholder - the actual implementation will be copied from the existing code
}

void JitInterpolateExecutor::cubicPlanar(const uint8_t* in_ptr_, uint8_t* out_ptr_, const void* post_ops_data_, int B, int C, int IH, int IW, int OH, int OW) {
    // TODO: Move actual implementation from InterpolateJitExecutor in interpolate.cpp
    // This is a placeholder - the actual implementation will be copied from the existing code
}

void JitInterpolateExecutor::cubicCGathered(const uint8_t* in_ptr_, uint8_t* out_ptr_, const void* post_ops_data_, int B, int C, int IH, int IW, int OH, int OW) {
    // TODO: Move actual implementation from InterpolateJitExecutor in interpolate.cpp
    // This is a placeholder - the actual implementation will be copied from the existing code
}

void JitInterpolateExecutor::pillowCGathered(const uint8_t* in_ptr_, uint8_t* out_ptr_, const void* post_ops_data_, int B, int C, int IH, int IW, int OH, int OW) {
    // TODO: Move actual implementation from InterpolateJitExecutor in interpolate.cpp
    // This is a placeholder - the actual implementation will be copied from the existing code
}

bool JitInterpolateExecutorBuilder::isSupported(const InterpolateAttrs& interpolateAttrs,
                                                 const std::vector<MemoryDescPtr>& srcDescs,
                                                 const std::vector<MemoryDescPtr>& dstDescs) const {
#if defined(OPENVINO_ARCH_X86_64)
    // Check if we have at least SSE4.1 support
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41)) {
        return false;
    }

    // Check if mode is supported by JIT
    switch (interpolateAttrs.mode) {
        case InterpolateMode::nearest:
        case InterpolateMode::linear_onnx:
        case InterpolateMode::cubic:
        case InterpolateMode::bilinear_pillow:
        case InterpolateMode::bicubic_pillow:
            return true;
        default:
            return false;
    }
#else
    return false;
#endif
}

InterpolateExecutorPtr JitInterpolateExecutorBuilder::makeExecutor(ExecutorContext::CPtr context) const {
    return std::make_shared<JitInterpolateExecutor>(context);
}

} // namespace ov::intel_cpu