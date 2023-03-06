// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "onednn/iml_type_mapper.h"
#include "executor.hpp"
#include "nodes/common/permute_kernel.h"

namespace ov {
namespace intel_cpu {

class TransposeExecutor {
public:
    TransposeExecutor() = default;
    virtual bool init(const PermuteParams& permuteParams,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr &attr) = 0;
    virtual void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const int MB) = 0;
    virtual ~TransposeExecutor() = default;
};
using executorPtr = std::shared_ptr<TransposeExecutor>;

} // namespace intel_cpu
} // namespace ov