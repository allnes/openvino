// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_interpolate.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/NEON/functions/NEScale.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include "acl_utils.hpp"
#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/interpolate_config.hpp"
#include "openvino/core/except.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"
#include "utils/cpu_utils.hpp"
#include "common/primitive_hashing_utils.hpp"
#include "cpu_types.h"

namespace ov {
namespace intel_cpu {

ACLInterpolateExecutor::ACLInterpolateExecutor(const InterpolateAttrs& attrs,
                                               const PostOpsPtr& postOps,
                                               const MemoryArgs& memory,
                                               const ExecutorContext::CPtr context)
    : m_attrs(attrs) {
    // Initialize ACL components based on attrs and memory descriptors
    const auto& srcDesc = memory.at(ARG_SRC)->getDescPtr();
    const auto& dstDesc = memory.at(ARG_DST)->getDescPtr();
    
    // Always use generic ACL type - test framework will append precision
    
    const auto& srcDims = srcDesc->getShape().getStaticDims();
    const auto& dstDims = dstDesc->getShape().getStaticDims();
    
    // Configure ACL sampling policy based on coordinate transformation mode
    acl_coord = arm_compute::SamplingPolicy::TOP_LEFT;
    static const size_t index_h = 2;
    static const size_t index_w = 3;
    
    if ((m_attrs.coordTransMode == InterpolateCoordTransMode::pytorch_half_pixel &&
         dstDims[index_h] > 1 && dstDims[index_w] > 1) ||
        m_attrs.coordTransMode == InterpolateCoordTransMode::half_pixel) {
        acl_coord = arm_compute::SamplingPolicy::CENTER;
    }
    
    // Set interpolation policy based on mode
    switch (m_attrs.mode) {
    case InterpolateMode::linear:
    case InterpolateMode::linear_onnx:
        acl_policy = arm_compute::InterpolationPolicy::BILINEAR;
        break;
    case InterpolateMode::nearest:
        acl_policy = arm_compute::InterpolationPolicy::NEAREST_NEIGHBOR;
        break;
    default:
        OPENVINO_THROW("ACL Interpolate: unsupported mode ", static_cast<int>(m_attrs.mode));
    }
    
    auto aclSrcDims = shapeCast(srcDims);
    auto aclDstDims = shapeCast(dstDims);
    
    if (srcDesc->hasLayoutType(LayoutType::nspc) && dstDesc->hasLayoutType(LayoutType::nspc)) {
        changeLayoutToNH_C({&aclSrcDims, &aclDstDims});
    }
    
    auto srcTensorInfo = arm_compute::TensorInfo(aclSrcDims,
                                                 1,
                                                 precisionToAclDataType(srcDesc->getPrecision()),
                                                 getAclDataLayoutByMemoryDesc(srcDesc));
    auto dstTensorInfo = arm_compute::TensorInfo(aclDstDims,
                                                 1,
                                                 precisionToAclDataType(dstDesc->getPrecision()),
                                                 getAclDataLayoutByMemoryDesc(dstDesc));
    
    // Validate configuration
    arm_compute::Status status = arm_compute::NEScale::validate(
        &srcTensorInfo,
        &dstTensorInfo,
        arm_compute::ScaleKernelInfo(acl_policy,
                                     arm_compute::BorderMode::REPLICATE,
                                     arm_compute::PixelValue(),
                                     acl_coord,
                                     false,
                                     m_attrs.coordTransMode == InterpolateCoordTransMode::align_corners,
                                     getAclDataLayoutByMemoryDesc(srcDesc)));
    
    if (!status) {
        OPENVINO_THROW("ACL NEScale validation failed: ", status.error_description());
    }
    
    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);
    
    acl_scale = std::make_unique<arm_compute::NEScale>();
    configureThreadSafe([&] {
        acl_scale->configure(
            &srcTensor,
            &dstTensor,
            arm_compute::ScaleKernelInfo(acl_policy,
                                         arm_compute::BorderMode::REPLICATE,
                                         arm_compute::PixelValue(),
                                         acl_coord,
                                         false,
                                         m_attrs.coordTransMode == InterpolateCoordTransMode::align_corners,
                                         getAclDataLayoutByMemoryDesc(srcDesc)));
    });
}

bool ACLInterpolateExecutor::update(const MemoryArgs& memory) {
    // Update logic if needed when memory changes
    return true;
}

void ACLInterpolateExecutor::execute(const MemoryArgs& memory) {
    const auto& srcMem = memory.at(ARG_SRC);
    const auto& dstMem = memory.at(ARG_DST);
    
    const uint8_t* src_data = srcMem->getDataAs<const uint8_t>();
    uint8_t* dst_data = dstMem->getDataAs<uint8_t>();
    
    // Handle padding if needed
    const uint8_t* src_data_ptr = src_data;
    if (m_attrs.hasPad) {
        src_data_ptr = padPreprocess(srcMem, dstMem);
    }
    
    // Import memory and run ACL scale
    srcTensor.allocator()->import_memory(const_cast<void*>(reinterpret_cast<const void*>(src_data_ptr)));
    dstTensor.allocator()->import_memory(dst_data);
    
    acl_scale->run();
    
    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}

const uint8_t* ACLInterpolateExecutor::padPreprocess(const MemoryCPtr& src, const MemoryPtr& dst) {
    const auto& srcDims = src->getStaticDims();
    const auto& srcPrec = src->getPrecision();
    const uint8_t* src_data = src->getDataAs<const uint8_t>();
    
    VectorDims inDimsPad = getPaddedInputShape(srcDims, m_attrs.padBegin, m_attrs.padEnd);
    
    size_t padded_size = std::accumulate(inDimsPad.begin(), inDimsPad.end(), 
                                        srcPrec.size(), std::multiplies<size_t>());
    
    m_padded_input.resize(padded_size, 0);
    
    // Convert to 5D for unified processing
    const auto srcDim5d = to5Dim(srcDims);
    const auto srcDimPad5d = to5Dim(inDimsPad);
    const auto srcDataSize = srcPrec.size();
    size_t dimSize = srcDims.size();
    
    int padB0 = (dimSize > 2) ? m_attrs.padBegin[0] : 0;
    int padB1 = (dimSize > 2) ? m_attrs.padBegin[1] : 0;
    int padB2 = (dimSize == 5) ? m_attrs.padBegin[dimSize - 3] : 0;
    int padB3 = m_attrs.padBegin[dimSize - 2];
    int padB4 = m_attrs.padBegin[dimSize - 1];
    
    // Calculate strides for source and padded tensors
    std::vector<size_t> srcStrides(5), padStrides(5);
    srcStrides[4] = 1;
    padStrides[4] = 1;
    for (int i = 3; i >= 0; i--) {
        srcStrides[i] = srcStrides[i + 1] * srcDim5d[i + 1];
        padStrides[i] = padStrides[i + 1] * srcDimPad5d[i + 1];
    }
    
    // Copy data with padding
    for (int n = 0; n < srcDim5d[0]; n++) {
        for (int c = 0; c < srcDim5d[1]; c++) {
            for (int d = 0; d < srcDim5d[2]; d++) {
                for (int h = 0; h < srcDim5d[3]; h++) {
                    const uint8_t* src_ptr = src_data + (n * srcStrides[0] + c * srcStrides[1] + 
                                                        d * srcStrides[2] + h * srcStrides[3]) * srcDataSize;
                    uint8_t* dst_ptr = m_padded_input.data() + ((n + padB0) * padStrides[0] + (c + padB1) * padStrides[1] +
                                                                (d + padB2) * padStrides[2] + (h + padB3) * padStrides[3] + 
                                                                padB4) * srcDataSize;
                    std::memcpy(dst_ptr, src_ptr, srcDim5d[4] * srcDataSize);
                }
            }
        }
    }
    
    return m_padded_input.data();
}

}  // namespace intel_cpu
}  // namespace ov