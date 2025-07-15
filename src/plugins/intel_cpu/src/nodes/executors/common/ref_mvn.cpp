// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ref_mvn.hpp"

#include <cmath>
#include <vector>

#include "openvino/core/parallel.hpp"
#include "utils/bfloat16.hpp"
#include "utils/general_utils.h"
#include "nodes/common/cpu_convert.h"

namespace ov::intel_cpu {

MVNRefExecutor::MVNRefExecutor(const MVNAttrs& mvnAttrs,
                               const MemoryArgs& memory,
                               const ExecutorContext::CPtr& context)
    : attrs(mvnAttrs), memoryArgs(memory), context(context) {
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
    const size_t N = shape5d[0];
    const size_t C = shape5d[1];
    const size_t D = shape5d[2];
    const size_t H = shape5d[3];
    const size_t W = shape5d[4];
    const size_t total_size = N * C * D * H * W;
    
    // Check if we need conversion or can work directly with the data
    const float* src_data_ptr = nullptr;
    float* dst_data_ptr = nullptr;
    std::vector<float> src_float;
    std::vector<float> dst_float;
    
    if (attrs.src_prc == ov::element::f32) {
        // No conversion needed for input
        src_data_ptr = reinterpret_cast<const float*>(src_data);
    } else {
        // Convert input to float for intermediate calculations
        src_float.resize(total_size);
        cpu_convert(src_data, src_float.data(), attrs.src_prc, ov::element::f32, total_size);
        src_data_ptr = src_float.data();
    }
    
    if (attrs.dst_prc == ov::element::f32) {
        // No conversion needed for output
        dst_data_ptr = reinterpret_cast<float*>(dst_data);
    } else {
        // We'll need to convert output later
        dst_float.resize(total_size);
        dst_data_ptr = dst_float.data();
    }

    if (attrs.execAcrossChannels_) {
        parallel_for(N, [&](int b) {
            const size_t data_size = C * D * H * W;
            
            // Pre-calculate strides for better optimization
            const size_t spatial_size = D * H * W;
            const size_t HW = H * W;
            
            // Calculate mean
            double mean = 0;
            if (attrs.layout == mvn_planar) {
                // NCDHW/NCHW layout - planar format
                for (size_t c = 0; c < C; c++) {
                    const size_t c_offset = b * C * spatial_size + c * spatial_size;
                    for (size_t d = 0; d < D; d++) {
                        const size_t d_offset = c_offset + d * HW;
                        for (size_t h = 0; h < H; h++) {
                            const size_t h_offset = d_offset + h * W;
                            for (size_t w = 0; w < W; w++) {
                                mean += src_data_ptr[h_offset + w];
                            }
                        }
                    }
                }
            } else {
                // NDHWC/NHWC layout - channel last format
                const size_t b_offset = b * spatial_size * C;
                for (size_t d = 0; d < D; d++) {
                    const size_t d_offset = b_offset + d * HW * C;
                    for (size_t h = 0; h < H; h++) {
                        const size_t h_offset = d_offset + h * W * C;
                        for (size_t w = 0; w < W; w++) {
                            const size_t w_offset = h_offset + w * C;
                            for (size_t c = 0; c < C; c++) {
                                mean += src_data_ptr[w_offset + c];
                            }
                        }
                    }
                }
            }
            mean /= data_size;

            // Calculate variance (if needed) and normalize
            if (attrs.normalizeVariance_) {
                double variance = 0;
                if (attrs.layout == mvn_planar) {
                    for (size_t c = 0; c < C; c++) {
                        const size_t c_offset = b * C * spatial_size + c * spatial_size;
                        for (size_t d = 0; d < D; d++) {
                            const size_t d_offset = c_offset + d * HW;
                            for (size_t h = 0; h < H; h++) {
                                const size_t h_offset = d_offset + h * W;
                                for (size_t w = 0; w < W; w++) {
                                    double diff = src_data_ptr[h_offset + w] - mean;
                                    variance += diff * diff;
                                }
                            }
                        }
                    }
                } else {
                    const size_t b_offset = b * spatial_size * C;
                    for (size_t d = 0; d < D; d++) {
                        const size_t d_offset = b_offset + d * HW * C;
                        for (size_t h = 0; h < H; h++) {
                            const size_t h_offset = d_offset + h * W * C;
                            for (size_t w = 0; w < W; w++) {
                                const size_t w_offset = h_offset + w * C;
                                for (size_t c = 0; c < C; c++) {
                                    double diff = src_data_ptr[w_offset + c] - mean;
                                    variance += diff * diff;
                                }
                            }
                        }
                    }
                }
                variance /= data_size;

                double sigma = attrs.epsMode_ == INSIDE_SQRT
                    ? std::sqrt(variance + attrs.epsValue_)
                    : std::sqrt(variance) + attrs.epsValue_;
                const double inv_sigma = 1.0 / sigma;

                // Normalize
                if (attrs.layout == mvn_planar) {
                    for (size_t c = 0; c < C; c++) {
                        const size_t c_offset = b * C * spatial_size + c * spatial_size;
                        for (size_t d = 0; d < D; d++) {
                            const size_t d_offset = c_offset + d * HW;
                            for (size_t h = 0; h < H; h++) {
                                const size_t h_offset = d_offset + h * W;
                                for (size_t w = 0; w < W; w++) {
                                    const size_t idx = h_offset + w;
                                    dst_data_ptr[idx] = (src_data_ptr[idx] - mean) * inv_sigma;
                                }
                            }
                        }
                    }
                } else {
                    const size_t b_offset = b * spatial_size * C;
                    for (size_t d = 0; d < D; d++) {
                        const size_t d_offset = b_offset + d * HW * C;
                        for (size_t h = 0; h < H; h++) {
                            const size_t h_offset = d_offset + h * W * C;
                            for (size_t w = 0; w < W; w++) {
                                const size_t w_offset = h_offset + w * C;
                                for (size_t c = 0; c < C; c++) {
                                    const size_t idx = w_offset + c;
                                    dst_data_ptr[idx] = (src_data_ptr[idx] - mean) * inv_sigma;
                                }
                            }
                        }
                    }
                }
            } else {
                // Just subtract mean
                if (attrs.layout == mvn_planar) {
                    for (size_t c = 0; c < C; c++) {
                        const size_t c_offset = b * C * spatial_size + c * spatial_size;
                        for (size_t d = 0; d < D; d++) {
                            const size_t d_offset = c_offset + d * HW;
                            for (size_t h = 0; h < H; h++) {
                                const size_t h_offset = d_offset + h * W;
                                for (size_t w = 0; w < W; w++) {
                                    const size_t idx = h_offset + w;
                                    dst_data_ptr[idx] = src_data_ptr[idx] - mean;
                                }
                            }
                        }
                    }
                } else {
                    const size_t b_offset = b * spatial_size * C;
                    for (size_t d = 0; d < D; d++) {
                        const size_t d_offset = b_offset + d * HW * C;
                        for (size_t h = 0; h < H; h++) {
                            const size_t h_offset = d_offset + h * W * C;
                            for (size_t w = 0; w < W; w++) {
                                const size_t w_offset = h_offset + w * C;
                                for (size_t c = 0; c < C; c++) {
                                    const size_t idx = w_offset + c;
                                    dst_data_ptr[idx] = src_data_ptr[idx] - mean;
                                }
                            }
                        }
                    }
                }
            }
        });
    } else {
        parallel_for2d(N, C, [&](int b, int c) {
            const size_t data_size = D * H * W;
            const size_t HW = H * W;
            
            // Calculate mean
            double mean = 0;
            if (attrs.layout == mvn_planar) {
                // NCDHW/NCHW layout - planar format
                const size_t c_offset = b * C * data_size + c * data_size;
                for (size_t d = 0; d < D; d++) {
                    const size_t d_offset = c_offset + d * HW;
                    for (size_t h = 0; h < H; h++) {
                        const size_t h_offset = d_offset + h * W;
                        for (size_t w = 0; w < W; w++) {
                            mean += src_data_ptr[h_offset + w];
                        }
                    }
                }
            } else {
                // NDHWC/NHWC layout - channel last format
                const size_t b_offset = b * data_size * C;
                for (size_t d = 0; d < D; d++) {
                    const size_t d_offset = b_offset + d * HW * C;
                    for (size_t h = 0; h < H; h++) {
                        const size_t h_offset = d_offset + h * W * C;
                        for (size_t w = 0; w < W; w++) {
                            mean += src_data_ptr[h_offset + w * C + c];
                        }
                    }
                }
            }
            mean /= data_size;

            // Calculate variance (if needed) and normalize
            if (attrs.normalizeVariance_) {
                double variance = 0;
                if (attrs.layout == mvn_planar) {
                    const size_t c_offset = b * C * data_size + c * data_size;
                    for (size_t d = 0; d < D; d++) {
                        const size_t d_offset = c_offset + d * HW;
                        for (size_t h = 0; h < H; h++) {
                            const size_t h_offset = d_offset + h * W;
                            for (size_t w = 0; w < W; w++) {
                                double diff = src_data_ptr[h_offset + w] - mean;
                                variance += diff * diff;
                            }
                        }
                    }
                } else {
                    const size_t b_offset = b * data_size * C;
                    for (size_t d = 0; d < D; d++) {
                        const size_t d_offset = b_offset + d * HW * C;
                        for (size_t h = 0; h < H; h++) {
                            const size_t h_offset = d_offset + h * W * C;
                            for (size_t w = 0; w < W; w++) {
                                double diff = src_data_ptr[h_offset + w * C + c] - mean;
                                variance += diff * diff;
                            }
                        }
                    }
                }
                variance /= data_size;

                double sigma = attrs.epsMode_ == INSIDE_SQRT
                    ? std::sqrt(variance + attrs.epsValue_)
                    : std::sqrt(variance) + attrs.epsValue_;
                const double inv_sigma = 1.0 / sigma;

                // Normalize
                if (attrs.layout == mvn_planar) {
                    const size_t c_offset = b * C * data_size + c * data_size;
                    for (size_t d = 0; d < D; d++) {
                        const size_t d_offset = c_offset + d * HW;
                        for (size_t h = 0; h < H; h++) {
                            const size_t h_offset = d_offset + h * W;
                            for (size_t w = 0; w < W; w++) {
                                const size_t idx = h_offset + w;
                                dst_data_ptr[idx] = (src_data_ptr[idx] - mean) * inv_sigma;
                            }
                        }
                    }
                } else {
                    const size_t b_offset = b * data_size * C;
                    for (size_t d = 0; d < D; d++) {
                        const size_t d_offset = b_offset + d * HW * C;
                        for (size_t h = 0; h < H; h++) {
                            const size_t h_offset = d_offset + h * W * C;
                            for (size_t w = 0; w < W; w++) {
                                const size_t idx = h_offset + w * C + c;
                                dst_data_ptr[idx] = (src_data_ptr[idx] - mean) * inv_sigma;
                            }
                        }
                    }
                }
            } else {
                // Just subtract mean
                if (attrs.layout == mvn_planar) {
                    const size_t c_offset = b * C * data_size + c * data_size;
                    for (size_t d = 0; d < D; d++) {
                        const size_t d_offset = c_offset + d * HW;
                        for (size_t h = 0; h < H; h++) {
                            const size_t h_offset = d_offset + h * W;
                            for (size_t w = 0; w < W; w++) {
                                const size_t idx = h_offset + w;
                                dst_data_ptr[idx] = src_data_ptr[idx] - mean;
                            }
                        }
                    }
                } else {
                    const size_t b_offset = b * data_size * C;
                    for (size_t d = 0; d < D; d++) {
                        const size_t d_offset = b_offset + d * HW * C;
                        for (size_t h = 0; h < H; h++) {
                            const size_t h_offset = d_offset + h * W * C;
                            for (size_t w = 0; w < W; w++) {
                                const size_t idx = h_offset + w * C + c;
                                dst_data_ptr[idx] = src_data_ptr[idx] - mean;
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Convert output back to destination precision if needed
    if (attrs.dst_prc != ov::element::f32) {
        cpu_convert(dst_float.data(), dst_data, ov::element::f32, attrs.dst_prc, total_size);
    }
}

}  // namespace ov::intel_cpu