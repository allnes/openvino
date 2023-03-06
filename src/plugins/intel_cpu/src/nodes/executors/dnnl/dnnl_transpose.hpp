// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/transpose.hpp"

namespace ov {
namespace intel_cpu {

class TransposeJitExecutor : public TransposeExecutor {
public:
    explicit TransposeJitExecutor(const ExecutorContext::CPtr context);
    bool init(const PermuteParams& permuteParams,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const int MB) override;
protected:
    std::shared_ptr<PermuteKernel> pKernel;
};

} // namespace intel_cpu
} // namespace ov