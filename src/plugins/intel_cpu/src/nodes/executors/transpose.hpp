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
    virtual void exec(MemoryPtr& srcMemPtr, MemoryPtr& dstMemPtr, const int MB) = 0;
    virtual ~TransposeExecutor() = default;
    void setNode(Node* _node) { curr_node = _node; }
protected:
    Node *curr_node;
};
using executorPtr = std::shared_ptr<TransposeExecutor>;

} // namespace intel_cpu
} // namespace ov