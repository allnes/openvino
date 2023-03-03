// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose.hpp"

void ov::intel_cpu::TransposeExecutor::setNode(ov::intel_cpu::Node *_node) {
    curr_node = _node;
}

ov::intel_cpu::TransposeExecutor::TransposeExecutor(const ov::intel_cpu::ExecutorContext::CPtr context) : context(context) {}
