// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "nodes/executors/executor.hpp"
#include "nodes/executors/interpolate.hpp"

namespace ov::intel_cpu {

class InterpolateRefExecutor : public InterpolateExecutor {
public:
    explicit InterpolateRefExecutor(const ExecutorContext::CPtr& context) : InterpolateExecutor(context) {}

    bool init(const InterpolateAttrs& interpolateAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr& attr) override;

    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void* post_ops_data_) override;

    [[nodiscard]] impl_desc_type getImplType() const override { return impl_desc_type::ref; }

private:
    // Helpers copied from node ref implementation
    static float getValue(const uint8_t* base, size_t offset, ov::element::Type prec);
    static void setValue(uint8_t* base, size_t offset, float value, ov::element::Type prec);

    // Reference kernels
    void NNRef(const uint8_t* in_ptr_,
               uint8_t* out_ptr_,
               int B,
               int C,
               int ID,
               int IH,
               int IW,
               int OD,
               int OH,
               int OW);

    void linearOnnxRef(const uint8_t* in_ptr_,
                       uint8_t* out_ptr_,
                       int B,
                       int C,
                       int ID,
                       int IH,
                       int IW,
                       int OD,
                       int OH,
                       int OW);

    void linearRef(const uint8_t* in_ptr_,
                   uint8_t* out_ptr_,
                   int B,
                   int C,
                   int ID,
                   int IH,
                   int IW,
                   int OD,
                   int OH,
                   int OW);

    void pillowRef(const uint8_t* in_ptr_,
                   uint8_t* out_ptr_,
                   int B,
                   int C,
                   int IH,
                   int IW,
                   int OH,
                   int OW);

    void pillowRefNCHWAsNHWC(const uint8_t* in_ptr_,
                             uint8_t* out_ptr_,
                             int B,
                             int C,
                             int IH,
                             int IW,
                             int OH,
                             int OW);
};

}  // namespace ov::intel_cpu
