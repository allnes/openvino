// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include "../interpolate.hpp"
#include "jit_interpolate.hpp"

namespace ov::intel_cpu {

class JitInterpolateExecutor : public InterpolateExecutor {
public:
    JitInterpolateExecutor(ExecutorContext::CPtr context);

    bool init(const InterpolateAttrs& interpolateAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr& attr) override;

    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void* post_ops_data_) override;

    impl_desc_type getImplType() const override;

private:
    // Helper methods - these will be moved from the InterpolateJitExecutor class
    void NNPlanar(const uint8_t* in_ptr_, uint8_t* out_ptr_, const void* post_ops_data_, int B, int C, int ID, int IH, int IW, int OD, int OH, int OW);
    void NNCGathered(const uint8_t* in_ptr_, uint8_t* out_ptr_, const void* post_ops_data_, int B, int C, int ID, int IH, int IW, int OD, int OH, int OW);
    void linearOnnxPlanar(const uint8_t* in_ptr_, uint8_t* out_ptr_, const void* post_ops_data_, int B, int C, int ID, int IH, int IW, int OD, int OH, int OW);
    void linearOnnxCGathered(const uint8_t* in_ptr_, uint8_t* out_ptr_, const void* post_ops_data_, int B, int C, int ID, int IH, int IW, int OD, int OH, int OW);
    void cubicPlanar(const uint8_t* in_ptr_, uint8_t* out_ptr_, const void* post_ops_data_, int B, int C, int IH, int IW, int OH, int OW);
    void cubicCGathered(const uint8_t* in_ptr_, uint8_t* out_ptr_, const void* post_ops_data_, int B, int C, int IH, int IW, int OH, int OW);
    void pillowCGathered(const uint8_t* in_ptr_, uint8_t* out_ptr_, const void* post_ops_data_, int B, int C, int IH, int IW, int OH, int OW);

#if defined(OPENVINO_ARCH_X86_64)
    std::shared_ptr<ov::intel_cpu::node::jit_uni_interpolate_kernel> interpolateKernel = nullptr;
#endif
};

class JitInterpolateExecutorBuilder : public InterpolateExecutorBuilder {
public:
    bool isSupported(const InterpolateAttrs& interpolateAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override;

    InterpolateExecutorPtr makeExecutor(ExecutorContext::CPtr context) const override;
};

} // namespace ov::intel_cpu