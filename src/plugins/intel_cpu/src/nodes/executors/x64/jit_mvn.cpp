// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_mvn.hpp"
#include "openvino/core/parallel.hpp"
#include "utils/cpu_utils.hpp"
#include "nodes/mvn.h"
#include <memory>

using namespace dnnl;
using namespace dnnl::impl::cpu::x64;
using namespace ov;

namespace ov {
namespace intel_cpu {

MVNJitExecutor::MVNJitExecutor(const MVNAttrs& mvnAttrs,
                               const MemoryArgs& memory, 
                               const ExecutorContext::CPtr& context) 
    : MVNExecutor(mvnAttrs, memory, context) {
}

bool MVNJitExecutor::init(const MVNAttrs& mvnAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const dnnl::primitive_attr& attr) {
    attr_ = attr;
    shape5D = mvnAttrs.shape5D;
    
    auto jcp = node::jit_mvn_config_params();
    jcp.src_prc = attrs.src_prc;
    jcp.dst_prc = attrs.dst_prc;
    jcp.src_data_size = attrs.src_prc.size();
    jcp.dst_data_size = attrs.dst_prc.size();
    jcp.layout = attrs.layout;
    jcp.normalize_variance = attrs.normalizeVariance_;
    jcp.across_channels = attrs.execAcrossChannels_;

    // TODO: JIT kernel template implementations need to be added
    // The original JIT kernel implementations (jit_uni_mvn_kernel_f32 and jit_uni_mvn_mean_variance_kernel_f32)
    // were removed during refactoring. They need to be reimplemented following the new executor pattern.
    
#if defined(OPENVINO_ARCH_X86_64)
    // Temporarily disabled until JIT kernels are reimplemented
    // if (mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
    //     mvn_kernel.reset(new jit_uni_mvn_kernel_f32<dnnl::impl::cpu::x64::avx512_core>(jcp, *attr.get()));
    //     jcp.normalize_variance = false;
    //     mvn_mean_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<dnnl::impl::cpu::x64::avx512_core>(jcp));
    //     if (attrs.normalizeVariance_) {
    //         jcp.normalize_variance = true;
    //         mvn_variance_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<dnnl::impl::cpu::x64::avx512_core>(jcp));
    //     }
    // } else if (mayiuse(dnnl::impl::cpu::x64::avx2)) {
    //     mvn_kernel.reset(new jit_uni_mvn_kernel_f32<dnnl::impl::cpu::x64::avx2>(jcp, *attr.get()));
    //     jcp.normalize_variance = false;
    //     mvn_mean_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<dnnl::impl::cpu::x64::avx2>(jcp));
    //     if (attrs.normalizeVariance_) {
    //         jcp.normalize_variance = true;
    //         mvn_variance_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<dnnl::impl::cpu::x64::avx2>(jcp));
    //     }
    // } else if (mayiuse(dnnl::impl::cpu::x64::sse41)) {
    //     mvn_kernel.reset(new jit_uni_mvn_kernel_f32<dnnl::impl::cpu::x64::sse41>(jcp, *attr.get()));
    //     jcp.normalize_variance = false;
    //     mvn_mean_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<dnnl::impl::cpu::x64::sse41>(jcp));
    //     if (attrs.normalizeVariance_) {
    //         jcp.normalize_variance = true;
    //         mvn_variance_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<dnnl::impl::cpu::x64::sse41>(jcp));
    //     }
    // }

    // if (mvn_mean_kernel)
    //     mvn_mean_kernel->create_ker();
    // if (attrs.normalizeVariance_ && mvn_variance_kernel)
    //     mvn_variance_kernel->create_ker();
    // if (mvn_kernel)
    //     mvn_kernel->create_ker();
        
    // return mvn_kernel && mvn_mean_kernel;
    
    return false;  // JIT implementation not yet available
#else
    return false;
#endif
}

void MVNJitExecutor::executeImpl(const MemoryArgs& memory) {
    if (!mvn_mean_kernel || (attrs.normalizeVariance_ && !mvn_variance_kernel) || !mvn_kernel) {
        OPENVINO_THROW("MVN layer doesn't create kernel to execute on sse41 above platform.");
    }

    const uint8_t* src_data = memory.at(ARG_SRC_0)->getDataAs<const uint8_t>();
    uint8_t* dst_data = memory.at(ARG_DST)->getDataAs<uint8_t>();
    const void* post_ops_data_ = nullptr;

    if (attrs.layout == MVNLayoutType::mvn_planar) {
        mvn_pln(src_data, dst_data, post_ops_data_, shape5D);
    } else if (attrs.layout == MVNLayoutType::mvn_by_channel) {
        mvn_nspc(src_data, dst_data, post_ops_data_, shape5D);
    } else {
        mvn_blk(src_data, dst_data, post_ops_data_, shape5D);
    }
}

void MVNJitExecutor::mvn_pln(const uint8_t* src_data, uint8_t* dst_data, const void *post_ops_data_, const VectorDims& shape5d) {
    size_t blk_size = 1;  // blk size in vmm
    if (mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
        blk_size = 16;
    } else if (mayiuse(dnnl::impl::cpu::x64::avx2)) {
        blk_size = 8;
    } else if (mayiuse(dnnl::impl::cpu::x64::sse41)) {
        blk_size = 4;
    }

    const size_t N = shape5d[0];
    const size_t C = shape5d[1];
    const size_t D = shape5d[2];
    const size_t H = shape5d[3];
    const size_t W = shape5d[4];

    size_t C1 = H * W;
    size_t C2 = C1 * D;
    size_t C3 = C2 * C;

    if (attrs.execAcrossChannels_) {
        parallel_for(N, [&](size_t b) {
            size_t b_offset = b * C3;
            float mean = 0.0f;
            float variance = 0.0f;

            // mean for one instance in batch
            auto arg = node::jit_mvn_call_args();
            arg.src = src_data + b_offset * attrs.src_prc.size();
            arg.sum = &mean;
            arg.work_amount = static_cast<size_t>(C3);
            arg.rt_shape_size = static_cast<size_t>(0);
            (*mvn_mean_kernel)(&arg);
            mean /= static_cast<float>(C3);

            if (attrs.normalizeVariance_) {
                // variance for one instance in batch
                auto arg = node::jit_mvn_call_args();
                arg.src = src_data + b_offset * attrs.src_prc.size();
                arg.mean = &mean;
                arg.variance = &variance;
                arg.work_amount = static_cast<size_t>(C3);
                arg.rt_shape_size = static_cast<size_t>(0);
                (*mvn_variance_kernel)(&arg);

                variance = variance / static_cast<float>(C3);
                float variance_inv = 1.f;
                if (attrs.epsMode_ == MVNEpsMode::INSIDE_SQRT)
                    variance_inv /= sqrtf(variance + attrs.epsValue_);
                else if (attrs.epsMode_ == MVNEpsMode::OUTSIDE_SQRT)
                    variance_inv /= sqrtf(variance) + attrs.epsValue_;
                variance = variance_inv;
            }

            // mvn for one instance in batch
            arg = node::jit_mvn_call_args();
            arg.src = src_data + b_offset * attrs.src_prc.size();
            arg.dst = dst_data + b_offset * attrs.dst_prc.size();
            arg.mean = &mean;
            arg.variance = &variance;
            arg.work_amount = static_cast<size_t>(C3);
            arg.rt_shape_size = static_cast<size_t>(0);
            arg.oc_off = 0;
            arg.post_op_data = post_ops_data_;
            (*mvn_kernel)(&arg);
        });
    } else {  // per channel
        size_t threads_num = parallel_get_num_threads();
        size_t buffer_size = rnd_up(C, blk_size);
        std::vector<float> mean_buffer(buffer_size * threads_num);
        std::vector<float> variance_buffer(attrs.normalizeVariance_ ? buffer_size * threads_num : 0);

        parallel_for(N, [&](size_t b) {
            size_t b_offset = b * C3;
            auto mean_buffer_ptr = &mean_buffer[buffer_size * parallel_get_thread_num()];

            for (size_t c = 0; c < buffer_size; c++) {
                mean_buffer_ptr[c] = 0.f;
            }

            parallel_for3d(C, D, H, [&](size_t c, size_t d, size_t h) {
                size_t src_offset = b_offset + c * C2 + d * C1 + h * W;
                auto arg = node::jit_mvn_call_args();
                arg.src = src_data + src_offset * attrs.src_prc.size();
                arg.sum = &mean_buffer_ptr[c];
                arg.work_amount = static_cast<size_t>(W);
                arg.rt_shape_size = static_cast<size_t>(1);
                (*mvn_mean_kernel)(&arg);
            });

            // size_t src_stride = W * attrs.src_prc.size();
            // size_t dst_stride = W * attrs.dst_prc.size();

            parallel_for(buffer_size, [&](size_t c) {
                mean_buffer_ptr[c] /= static_cast<float>(D * H * W);
            });

            if (attrs.normalizeVariance_) {
                auto variance_buffer_ptr = &variance_buffer[buffer_size * parallel_get_thread_num()];
                for (size_t c = 0; c < buffer_size; c++) {
                    variance_buffer_ptr[c] = 0.f;
                }

                parallel_for3d(C, D, H, [&](size_t c, size_t d, size_t h) {
                    size_t src_offset = b_offset + c * C2 + d * C1 + h * W;
                    auto arg = node::jit_mvn_call_args();
                    arg.src = src_data + src_offset * attrs.src_prc.size();
                    arg.mean = &mean_buffer_ptr[c];
                    arg.variance = &variance_buffer_ptr[c];
                    arg.work_amount = static_cast<size_t>(W);
                    arg.rt_shape_size = static_cast<size_t>(1);
                    (*mvn_variance_kernel)(&arg);
                });

                parallel_for(buffer_size, [&](size_t c) {
                    variance_buffer_ptr[c] /= static_cast<float>(D * H * W);
                    if (attrs.epsMode_ == MVNEpsMode::INSIDE_SQRT)
                        variance_buffer_ptr[c] = 1.f / sqrtf(variance_buffer_ptr[c] + attrs.epsValue_);
                    else if (attrs.epsMode_ == MVNEpsMode::OUTSIDE_SQRT)
                        variance_buffer_ptr[c] = 1.f / (sqrtf(variance_buffer_ptr[c]) + attrs.epsValue_);
                });

                parallel_for3d(C, D, H, [&](size_t c, size_t d, size_t h) {
                    size_t src_offset = b_offset + c * C2 + d * C1 + h * W;
                    size_t dst_offset = b_offset + c * C2 + d * C1 + h * W;

                    auto arg = node::jit_mvn_call_args();
                    arg.src = src_data + src_offset * attrs.src_prc.size();
                    arg.dst = dst_data + dst_offset * attrs.dst_prc.size();
                    arg.mean = &mean_buffer_ptr[c];
                    arg.variance = &variance_buffer_ptr[c];
                    arg.work_amount = static_cast<size_t>(W);
                    arg.rt_shape_size = static_cast<size_t>(1);
                    arg.oc_off = c * sizeof(float);
                    arg.post_op_data = post_ops_data_;
                    (*mvn_kernel)(&arg);
                });
            } else {
                parallel_for3d(C, D, H, [&](size_t c, size_t d, size_t h) {
                    size_t src_offset = b_offset + c * C2 + d * C1 + h * W;
                    size_t dst_offset = b_offset + c * C2 + d * C1 + h * W;

                    auto arg = node::jit_mvn_call_args();
                    arg.src = src_data + src_offset * attrs.src_prc.size();
                    arg.dst = dst_data + dst_offset * attrs.dst_prc.size();
                    arg.mean = &mean_buffer_ptr[c];
                    arg.variance = 0;
                    arg.work_amount = static_cast<size_t>(W);
                    arg.rt_shape_size = static_cast<size_t>(1);
                    arg.oc_off = c * sizeof(float);
                    arg.post_op_data = post_ops_data_;
                    (*mvn_kernel)(&arg);
                });
            }
        });
    }
}

void MVNJitExecutor::mvn_nspc(const uint8_t* src_data, uint8_t* dst_data, const void *post_ops_data_, const VectorDims& shape5d) {
    size_t blk_size = 1;  // channel blk for memory layout
    if (mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
        blk_size = 16;
    } else if (mayiuse(dnnl::impl::cpu::x64::avx2)) {
        blk_size = 8;
    } else {
        blk_size = 4;
    }

    const size_t N = shape5d[0];
    const size_t C = shape5d[1];
    const size_t D = shape5d[2];
    const size_t H = shape5d[3];
    const size_t W = shape5d[4];

    size_t src_stride = C * attrs.src_prc.size();
    size_t dst_stride = C * attrs.dst_prc.size();

    if (attrs.execAcrossChannels_) {
        size_t C5 = C * D * H * W;
        parallel_for(N, [&](size_t b) {
            size_t b_offset = b * C5;
            float mean = 0.f;
            // mean for one instance in batch
            auto arg = node::jit_mvn_call_args();
            arg.src = src_data + b_offset * attrs.src_prc.size();
            arg.sum = &mean;
            arg.work_amount = static_cast<size_t>(D * H * W);
            arg.rt_shape_size = static_cast<size_t>(C);
            arg.oc_off = 0;
            (*mvn_mean_kernel)(&arg);
            mean /= C5;

            if (attrs.normalizeVariance_) {
                float variance = 0.f;
                // variance for one instance in batch
                auto arg = node::jit_mvn_call_args();
                arg.src = src_data + b_offset * attrs.src_prc.size();
                arg.mean = &mean;
                arg.variance = &variance;
                arg.work_amount = static_cast<size_t>(D * H * W);
                arg.rt_shape_size = static_cast<size_t>(C);
                arg.oc_off = 0;
                (*mvn_variance_kernel)(&arg);

                variance = variance / C5;
                float variance_inv = 1.f;
                if (attrs.epsMode_ == MVNEpsMode::INSIDE_SQRT)
                    variance_inv /= sqrtf(variance + attrs.epsValue_);
                else if (attrs.epsMode_ == MVNEpsMode::OUTSIDE_SQRT)
                    variance_inv /= sqrtf(variance) + attrs.epsValue_;
                variance = variance_inv;

                // mvn for one instance in batch
                arg = node::jit_mvn_call_args();
                arg.src = src_data + b_offset * attrs.src_prc.size();
                arg.dst = dst_data + b_offset * attrs.dst_prc.size();
                arg.mean = &mean;
                arg.variance = &variance;
                arg.work_amount = static_cast<size_t>(D * H * W);
                arg.rt_shape_size = static_cast<size_t>(C);
                arg.oc_off = 0;
                arg.post_op_data = post_ops_data_;
                (*mvn_kernel)(&arg);
            } else {
                // mvn for one instance in batch
                arg = node::jit_mvn_call_args();
                arg.src = src_data + b_offset * attrs.src_prc.size();
                arg.dst = dst_data + b_offset * attrs.dst_prc.size();
                arg.mean = &mean;
                arg.work_amount = static_cast<size_t>(D * H * W);
                arg.rt_shape_size = static_cast<size_t>(C);
                arg.oc_off = 0;
                arg.post_op_data = post_ops_data_;
                (*mvn_kernel)(&arg);
            }
        });
    } else {  // per channel
        size_t threads_num = parallel_get_num_threads();
        // buffer_size needs to be set to rnd(C, blk) + blk if tails to avoid wor with protected memory
        size_t buffer_size = attrs.execAcrossChannels_ ? blk_size : rnd_up(C, blk_size) + blk_size;
        std::vector<float> mean_buffer(buffer_size * threads_num);
        std::vector<float> variance_buffer(buffer_size * threads_num);

        for (size_t b = 0; b < N; b++) {
            size_t b_offset = b * C * D * H * W;

            for (size_t i = 0; i < mean_buffer.size(); i++)
                mean_buffer[i] = 0.f;

            // sum for mean per channel for one instance in batch
            parallel_for2d(D, H, [&](size_t thr_idx, size_t d, size_t h) {
                for (size_t w = 0; w < W; w++) {
                    size_t src_offset = b_offset + d * C * H * W + h * C * W + w * C;
                    auto mean_buffer_ptr = &mean_buffer[buffer_size * thr_idx];

                    auto arg = node::jit_mvn_call_args();
                    arg.src = src_data + src_offset * attrs.src_prc.size();
                    arg.sum = mean_buffer_ptr;
                    arg.work_amount = static_cast<size_t>(1);
                    arg.rt_shape_size = static_cast<size_t>(C);
                    arg.oc_off = 0;
                    (*mvn_mean_kernel)(&arg);
                }
            });

            // aggregate sum for mean per channel for one instance in batch
            for (size_t i = 1; i < threads_num; i++) {
                for (size_t c = 0; c < C; c++)
                    mean_buffer[c] += mean_buffer[c + buffer_size * i];
            }

            // mean per channel for one instance in batch
            for (size_t c = 0; c < C; c++)
                mean_buffer[c] /= static_cast<float>(D * H * W);

            if (attrs.normalizeVariance_) {
                for (size_t i = 0; i < variance_buffer.size(); i++)
                    variance_buffer[i] = 0.f;

                // sum for variance per channel for one instance in batch
                parallel_for2d(D, H, [&](size_t thr_idx, size_t d, size_t h) {
                    for (size_t w = 0; w < W; w++) {
                        size_t src_offset = b_offset + d * C * H * W + h * C * W + w * C;
                        auto mean_buffer_ptr = &mean_buffer[0];
                        auto variance_buffer_ptr = &variance_buffer[buffer_size * thr_idx];

                        auto arg = node::jit_mvn_call_args();
                        arg.src = src_data + src_offset * attrs.src_prc.size();
                        arg.mean = mean_buffer_ptr;
                        arg.variance = variance_buffer_ptr;
                        arg.work_amount = static_cast<size_t>(1);
                        arg.rt_shape_size = static_cast<size_t>(C);
                        arg.oc_off = 0;
                        (*mvn_variance_kernel)(&arg);
                    }
                });

                // aggregate sum for variance per channel for one instance in batch
                for (size_t i = 1; i < threads_num; i++) {
                    for (size_t c = 0; c < C; c++)
                        variance_buffer[c] += variance_buffer[c + buffer_size * i];
                }

                // variance per channel for one instance in batch
                for (size_t c = 0; c < C; c++) {
                    variance_buffer[c] /= static_cast<float>(D * H * W);
                    if (attrs.epsMode_ == MVNEpsMode::INSIDE_SQRT)
                        variance_buffer[c] = 1.f / sqrtf(variance_buffer[c] + attrs.epsValue_);
                    else if (attrs.epsMode_ == MVNEpsMode::OUTSIDE_SQRT)
                        variance_buffer[c] = 1.f / (sqrtf(variance_buffer[c]) + attrs.epsValue_);
                }

                // mvn for one instance in batch
                parallel_for2d(D, H, [&](size_t d, size_t h) {
                    for (size_t w = 0; w < W; w++) {
                        size_t src_offset = b_offset + d * C * H * W + h * C * W + w * C;
                        size_t dst_offset = b_offset + d * C * H * W + h * C * W + w * C;
                        auto mean_buffer_ptr = &mean_buffer[0];
                        auto variance_buffer_ptr = &variance_buffer[0];

                        auto arg = node::jit_mvn_call_args();
                        arg.src = src_data + src_offset * attrs.src_prc.size();
                        arg.dst = dst_data + dst_offset * attrs.dst_prc.size();
                        arg.mean = mean_buffer_ptr;
                        arg.variance = variance_buffer_ptr;
                        arg.work_amount = static_cast<size_t>(1);
                        arg.rt_shape_size = static_cast<size_t>(C);
                        arg.oc_off = 0;
                        arg.post_op_data = post_ops_data_;
                        (*mvn_kernel)(&arg);
                    }
                });
            } else {
                // mvn for one instance in batch
                parallel_for2d(D, H, [&](size_t d, size_t h) {
                    for (size_t w = 0; w < W; w++) {
                        size_t src_offset = b_offset + d * C * H * W + h * C * W + w * C;
                        size_t dst_offset = b_offset + d * C * H * W + h * C * W + w * C;
                        auto mean_buffer_ptr = &mean_buffer[0];

                        auto arg = node::jit_mvn_call_args();
                        arg.src = src_data + src_offset * attrs.src_prc.size();
                        arg.dst = dst_data + dst_offset * attrs.dst_prc.size();
                        arg.mean = mean_buffer_ptr;
                        arg.work_amount = static_cast<size_t>(1);
                        arg.rt_shape_size = static_cast<size_t>(C);
                        arg.oc_off = 0;
                        arg.post_op_data = post_ops_data_;
                        (*mvn_kernel)(&arg);
                    }
                });
            }
        }
    }
}

void MVNJitExecutor::mvn_blk(const uint8_t* src_data, uint8_t* dst_data, const void *post_ops_data_, const VectorDims& shape5d) {
    size_t blk_size = 1;  // channel blk for memory layout
    if (mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
        blk_size = 16;
    } else {
        blk_size = 8;
    }

    const size_t N = shape5d[0];
    const size_t C = shape5d[1];
    const size_t D = shape5d[2];
    const size_t H = shape5d[3];
    const size_t W = shape5d[4];

    size_t CB = div_up(C, blk_size);

    size_t C0 = W * blk_size;
    size_t C1 = C0 * H;
    size_t C2 = C1 * D;
    size_t C3 = C2 * CB;
    size_t C5 = C * D * H * W;

    size_t threads_num = parallel_get_num_threads();
    size_t aux_buffer_size = attrs.execAcrossChannels_ ? blk_size : rnd_up(C, blk_size);
    aux_buffer_size += blk_size;
    std::vector<float> mean_buffer(aux_buffer_size * threads_num);
    std::vector<float> variance_buffer(aux_buffer_size * threads_num);

    for (size_t b = 0lu; b < N; b++) {
        size_t b_offset = b * C3;
        if (attrs.execAcrossChannels_) {
            // mean for this instance in batch
            float C5inv = 1.f / static_cast<float>(C5);
            float mean_temp = 0.0f;
            mean_temp = parallel_sum3d(CB, D, H, mean_temp, [&](size_t cb, size_t d, size_t h)->float {
                size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;

                float mean_internal = 0.0f;
                auto mean_buffer_ptr = &mean_buffer[aux_buffer_size * parallel_get_thread_num()];
                for (size_t i = 0; i < blk_size; i++)
                    mean_buffer_ptr[i] = 0.f;

                auto arg = node::jit_mvn_call_args();
                arg.src = src_data + src_offset * attrs.src_prc.size();
                arg.sum = mean_buffer_ptr;
                arg.work_amount = static_cast<size_t>(W);
                // real tail number or tail is 0(for full vector block).
                arg.rt_shape_size = (C - cb * blk_size) < blk_size ? static_cast<size_t>(C % blk_size) : 0;
                arg.oc_off = static_cast<size_t>(cb * blk_size * sizeof(float));  // for tail process
                (*mvn_mean_kernel)(&arg); // for W * blk

                size_t min_cb = (std::min)(blk_size, C - cb * blk_size);
                for (size_t i = 0; i < min_cb; i++)
                    mean_internal += mean_buffer_ptr[i];
                return mean_internal;
            });
            float mean = mean_temp * C5inv;

            if (attrs.normalizeVariance_) {
                // variance: sum((x-mean)*(x-mean)) for one instance in batch
                float variance_temp = 0.0f;
                variance_temp = parallel_sum3d(CB, D, H, variance_temp, [&](size_t cb, size_t d, size_t h)->float {
                    size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;

                    float variance_internal = 0.0f;
                    auto variance_buffer_ptr = &variance_buffer[aux_buffer_size * parallel_get_thread_num()];
                    for (size_t i = 0; i < blk_size; i++)
                        variance_buffer_ptr[i] = 0.f;

                    auto arg = node::jit_mvn_call_args();
                    arg.src = src_data + src_offset * attrs.src_prc.size();
                    arg.mean = static_cast<float*>(&mean);
                    arg.variance = variance_buffer_ptr;
                    arg.work_amount = static_cast<size_t>(W);
                    arg.rt_shape_size = (C - cb * blk_size) < blk_size ? static_cast<size_t>(C % blk_size) : 0;
                    arg.oc_off = cb * blk_size * sizeof(float);
                    arg.post_op_data = post_ops_data_;
                    (*mvn_variance_kernel)(&arg);

                    size_t min_cb = (std::min)(blk_size, C - cb * blk_size);
                    for (size_t i = 0; i < min_cb; i++)
                        variance_internal += variance_buffer_ptr[i];
                    return variance_internal;
                });

                float variance = 1.f;
                if (attrs.epsMode_ == MVNEpsMode::INSIDE_SQRT)
                    variance /= sqrtf(variance_temp * C5inv + attrs.epsValue_);
                else if (attrs.epsMode_ == MVNEpsMode::OUTSIDE_SQRT)
                    variance /= sqrtf(variance_temp * C5inv) + attrs.epsValue_;

                // mvn for one instance in batch
                parallel_for3d(CB, D, H, [&](size_t cb, size_t d, size_t h) {
                    size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;
                    auto arg = node::jit_mvn_call_args();
                    arg.src = src_data + src_offset * attrs.src_prc.size();
                    arg.dst = dst_data + src_offset * attrs.dst_prc.size();
                    arg.mean = static_cast<float*>(&mean);
                    arg.variance = static_cast<float*>(&variance);
                    arg.work_amount = static_cast<size_t>(W);
                    arg.rt_shape_size = (C - cb * blk_size) < blk_size ? static_cast<size_t>(C % blk_size) : 0;
                    arg.oc_off = cb * blk_size * sizeof(float);
                    arg.post_op_data = post_ops_data_;
                    (*mvn_kernel)(&arg);
                });
            } else {
                // mvn for one instance in batch
                parallel_for3d(CB, D, H, [&](size_t cb, size_t d, size_t h) {
                    size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;
                    auto arg = node::jit_mvn_call_args();
                    arg.src = src_data + src_offset * attrs.src_prc.size();
                    arg.dst = dst_data + src_offset * attrs.dst_prc.size();
                    arg.mean = static_cast<float*>(&mean);
                    arg.work_amount = static_cast<size_t>(W);
                    arg.rt_shape_size = (C - cb * blk_size) < blk_size ? static_cast<size_t>(C % blk_size) : 0;
                    arg.oc_off = cb * blk_size * sizeof(float);
                    arg.post_op_data = post_ops_data_;
                    (*mvn_kernel)(&arg);
                });
            }
        } else {  // for per_channel
            float size_inv = 1.f / static_cast<float>(D * H * W);
            for (size_t i = 0; i < mean_buffer.size(); i++)
                mean_buffer[i] = 0.f;

            // one thread for one C*W size(the same H) to get C size result for the same H, added to last group result
            // keep the compute order the same as planar
            parallel_for2d(D, H, [&](size_t thr_idx, size_t d, size_t h) {
                for (size_t cb = 0; cb < CB; cb++) {
                    size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;
                    auto mean_buffer_ptr = &mean_buffer[blk_size * cb + aux_buffer_size * thr_idx];

                    auto arg = node::jit_mvn_call_args();
                    arg.src = src_data + src_offset * attrs.src_prc.size();
                    arg.sum = mean_buffer_ptr;
                    arg.work_amount = static_cast<size_t>(W);
                    arg.rt_shape_size = (C - cb * blk_size) < blk_size ? static_cast<size_t>(C % blk_size) : 0;
                    arg.oc_off = cb * blk_size * sizeof(float);
                    arg.post_op_data = post_ops_data_;
                    (*mvn_mean_kernel)(&arg);
                }
            });

            for (size_t i = 1; i < threads_num; i++) {
                for (size_t c = 0; c < C; c++)
                    mean_buffer[c] += mean_buffer[c + aux_buffer_size * i];
            }
            for (size_t c = 0; c < C; c++)
                mean_buffer[c] *= size_inv;

            if (attrs.normalizeVariance_) {
                for (size_t i = 0; i < variance_buffer.size(); i++)
                    variance_buffer[i] = 0.f;

                parallel_for2d(D, H, [&](size_t thr_idx, size_t d, size_t h) {
                    for (size_t cb = 0; cb < CB; cb++) {
                        size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;
                        auto mean_buffer_ptr = &mean_buffer[blk_size * cb];
                        auto variance_buffer_ptr = &variance_buffer[blk_size * cb + aux_buffer_size * thr_idx];

                        auto arg = node::jit_mvn_call_args();
                        arg.src = src_data + src_offset * attrs.src_prc.size();
                        arg.mean = mean_buffer_ptr;
                        arg.variance = variance_buffer_ptr;
                        arg.work_amount = static_cast<size_t>(W);
                        arg.rt_shape_size = (C - cb * blk_size) < blk_size ? static_cast<size_t>(C % blk_size) : 0;
                        arg.oc_off = cb * blk_size * sizeof(float);
                        arg.post_op_data = post_ops_data_;
                        (*mvn_variance_kernel)(&arg);
                    }
                });
                for (size_t i = 1; i < threads_num; i++) {
                    for (size_t c = 0; c < C; c++)
                        variance_buffer[c] += variance_buffer[c + aux_buffer_size * i];
                }
                for (size_t c = 0; c < C; c++) {
                    if (attrs.epsMode_ == MVNEpsMode::INSIDE_SQRT)
                        variance_buffer[c] = 1.f / sqrtf(variance_buffer[c] * size_inv + attrs.epsValue_);
                    else if (attrs.epsMode_ == MVNEpsMode::OUTSIDE_SQRT)
                        variance_buffer[c] = 1.f / (sqrtf(variance_buffer[c] * size_inv) + attrs.epsValue_);
                }

                parallel_for2d(D, H, [&](size_t d, size_t h) {
                    for (size_t cb = 0; cb < CB; cb++) {
                        size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;
                        auto mean_buffer_ptr = &mean_buffer[blk_size * cb];
                        auto variance_buffer_ptr = &variance_buffer[blk_size * cb];

                        auto arg = node::jit_mvn_call_args();
                        arg.src = src_data + src_offset * attrs.src_prc.size();
                        arg.dst = dst_data + src_offset * attrs.dst_prc.size();
                        arg.mean = mean_buffer_ptr;
                        arg.variance = variance_buffer_ptr;
                        arg.work_amount = static_cast<size_t>(W);
                        arg.rt_shape_size = (C - cb * blk_size) < blk_size ? static_cast<size_t>(C % blk_size) : 0;
                        arg.oc_off = cb * blk_size * sizeof(float);
                        arg.post_op_data = post_ops_data_;
                        (*mvn_kernel)(&arg);
                    }
                });
            } else {
                // normalizeVariance_ == false
                parallel_for2d(D, H, [&](size_t d, size_t h) {
                    for (size_t cb = 0; cb < CB; cb++) {
                        size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;
                        auto mean_buffer_ptr = &mean_buffer[blk_size * cb];

                        auto arg = node::jit_mvn_call_args();
                        arg.src = src_data + src_offset * attrs.src_prc.size();
                        arg.dst = dst_data + src_offset * attrs.dst_prc.size();
                        arg.mean = mean_buffer_ptr;
                        arg.work_amount = static_cast<size_t>(W);
                        arg.rt_shape_size = (C - cb * blk_size) < blk_size ? static_cast<size_t>(C % blk_size) : 0;
                        arg.oc_off = cb * blk_size * sizeof(float);
                        arg.post_op_data = post_ops_data_;
                        (*mvn_kernel)(&arg);
                    }
                });
            }
        }
    }
}

bool MVNJitExecutor::supports(const MVNAttrs& attrs,
                              const std::vector<MemoryDescPtr>& srcDescs,
                              const std::vector<MemoryDescPtr>& dstDescs) {
    // Check if x86_64 and at least SSE4.1 is available
    if (!mayiuse(dnnl::impl::cpu::x64::sse41))
        return false;
    
    // Check supported precisions
    if (!one_of(attrs.src_prc, ov::element::f32, ov::element::bf16, ov::element::f16))
        return false;
    if (!one_of(attrs.dst_prc, ov::element::f32, ov::element::bf16, ov::element::f16))
        return false;
    
    // Check layout
    if (!one_of(attrs.layout, MVNLayoutType::mvn_planar, MVNLayoutType::mvn_by_channel, MVNLayoutType::mvn_block))
        return false;
    
    // For now, JIT kernels are not implemented, so return false
    // TODO: Enable once JIT kernels are implemented
    return false;
}

}   // namespace intel_cpu
}   // namespace ov