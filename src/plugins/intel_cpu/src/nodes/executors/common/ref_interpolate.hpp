// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../interpolate.hpp"

namespace ov {
namespace intel_cpu {

class RefInterpolateExecutor : public InterpolateExecutor {
public:
    RefInterpolateExecutor(const ExecutorContext::CPtr context) : InterpolateExecutor(context) {}

    bool init(const InterpolateAttrs& interpolateAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;

    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) override;

    impl_desc_type getImplType() const override {
        return implType;
    }

private:
    void NNRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW, int OD, int OH, int OW);
    void linearOnnxRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW, int OD, int OH, int OW);

    void cubicRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int IH, int IW, int OH, int OW);
    void linearInterpolation(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                             float fx, float fy, float fz, int OD, int OH, int OW, int kernel_width, bool antialias);

    static float getValue(const uint8_t *base, size_t offset, InferenceEngine::Precision prec);
    static void setValue(uint8_t *base, size_t offset, float value, InferenceEngine::Precision prec);

private:
    impl_desc_type implType = impl_desc_type::ref;
    bool antialias;
    std::vector<float> dataScales;
};

class RefInterpolateExecutorBuilder : public InterpolateExecutorBuilder {
public:
    bool isSupported(const InterpolateAttrs& interpolateAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        return true;
    }

    InterpolateExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<RefInterpolateExecutor>(context);
    }
};
} // namespace intel_cpu
} // namespace ov