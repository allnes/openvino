// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "onednn/iml_type_mapper.h"
#include "executor.hpp"

namespace ov {
namespace intel_cpu {

class TransposeExecutor {
public:
    TransposeExecutor() = default;
    virtual void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const int MB) = 0;
    virtual ~TransposeExecutor() = default;
};
using executorPtr = std::shared_ptr<TransposeExecutor>;

} // namespace intel_cpu
} // namespace ov