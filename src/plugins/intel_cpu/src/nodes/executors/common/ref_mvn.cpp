// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ref_mvn.hpp"

#include <cmath>
#include <vector>

#include "openvino/core/parallel.hpp"
#include "utils/bfloat16.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

MVNRefExecutor::MVNRefExecutor(const MVNAttrs& mvnAttrs,
                               const MemoryArgs& memory,
                               const ExecutorContext::CPtr& context)
    : MVNExecutor(mvnAttrs, memory, context) {
    // Initialize the reference implementation
    auto srcDesc = memory.at(ARG_SRC_0)->getDescPtr();
    auto dstDesc = memory.at(ARG_DST)->getDescPtr();
    std::vector<MemoryDescPtr> srcDescs = {srcDesc};
    std::vector<MemoryDescPtr> dstDescs = {dstDesc};
    
    dnnl::primitive_attr attr;
    init(mvnAttrs, srcDescs, dstDescs, attr);
}

bool MVNRefExecutor::supports(const MVNAttrs& attrs,
                              const std::vector<MemoryDescPtr>& srcDescs,
                              const std::vector<MemoryDescPtr>& dstDescs) {
    // Reference implementation supports all configurations
    return true;
}

bool MVNRefExecutor::init(const MVNAttrs& mvnAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const dnnl::primitive_attr& attr) {
    attrs = mvnAttrs;
    
    auto srcDesc = srcDescs[0];
    auto dstDesc = dstDescs[0];
    
    if (!srcDesc || !dstDesc)
        return false;

    // Use the transformed 5D shape from attrs
    shape5D = attrs.shape5D;
    if (shape5D.size() != 5 || shape5D.empty())
        return false;
        
    src_data_size = srcDesc->getPrecision().size();
    dst_data_size = dstDesc->getPrecision().size();
    
    return true;
}

void MVNRefExecutor::executeImpl(const MemoryArgs& memory) {
    const auto src = memory.at(ARG_SRC_0);
    const auto dst = memory.at(ARG_DST);
    
    const uint8_t* src_data = src->getDataAs<const uint8_t>();
    uint8_t* dst_data = dst->getDataAs<uint8_t>();
    
    mvn_ref(src_data, dst_data, attrs.shape5D);
}

void MVNRefExecutor::mvn_ref(const uint8_t* src_data, uint8_t* dst_data, const VectorDims& shape5d) {
    const float* src_data_ptr = reinterpret_cast<const float*>(src_data);
    float* dst_data_ptr = reinterpret_cast<float*>(dst_data);

    const size_t N = shape5d[0];
    const size_t C = shape5d[1];
    const size_t D = shape5d[2];
    const size_t H = shape5d[3];
    const size_t W = shape5d[4];

    if (attrs.execAcrossChannels_) {
        parallel_for(N, [&](int b) {
            const size_t data_size = C * D * H * W;
            const float* src_data_ptr_b = src_data_ptr + b * data_size;
            float* dst_data_ptr_b = dst_data_ptr + b * data_size;

            // Calculate mean
            double mean = 0;
            for (size_t i = 0; i < data_size; i++) {
                mean += src_data_ptr_b[i];
            }
            mean /= data_size;

            // Calculate variance (if needed) and normalize
            if (attrs.normalizeVariance_) {
                double variance = 0;
                for (size_t i = 0; i < data_size; i++) {
                    double diff = src_data_ptr_b[i] - mean;
                    variance += diff * diff;
                }
                variance /= data_size;

                double sigma = attrs.epsMode_ == INSIDE_SQRT
                    ? std::sqrt(variance + attrs.epsValue_)
                    : std::sqrt(variance) + attrs.epsValue_;

                for (size_t i = 0; i < data_size; i++) {
                    dst_data_ptr_b[i] = (src_data_ptr_b[i] - mean) / sigma;
                }
            } else {
                for (size_t i = 0; i < data_size; i++) {
                    dst_data_ptr_b[i] = src_data_ptr_b[i] - mean;
                }
            }
        });
    } else {
        parallel_for2d(N, C, [&](int b, int c) {
            const size_t data_size = D * H * W;
            const size_t offset = (b * C + c) * data_size;
            const float* src_data_ptr_c = src_data_ptr + offset;
            float* dst_data_ptr_c = dst_data_ptr + offset;

            // Calculate mean
            double mean = 0;
            for (size_t i = 0; i < data_size; i++) {
                mean += src_data_ptr_c[i];
            }
            mean /= data_size;

            // Calculate variance (if needed) and normalize
            if (attrs.normalizeVariance_) {
                double variance = 0;
                for (size_t i = 0; i < data_size; i++) {
                    double diff = src_data_ptr_c[i] - mean;
                    variance += diff * diff;
                }
                variance /= data_size;

                double sigma = attrs.epsMode_ == INSIDE_SQRT
                    ? std::sqrt(variance + attrs.epsValue_)
                    : std::sqrt(variance) + attrs.epsValue_;

                for (size_t i = 0; i < data_size; i++) {
                    dst_data_ptr_c[i] = (src_data_ptr_c[i] - mean) / sigma;
                }
            } else {
                for (size_t i = 0; i < data_size; i++) {
                    dst_data_ptr_c[i] = src_data_ptr_c[i] - mean;
                }
            }
        });
    }
}

}  // namespace ov::intel_cpu