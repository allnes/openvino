// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_transpose.hpp"

ov::intel_cpu::ACLTransposeExecutor::ACLTransposeExecutor(const ExecutorContext::CPtr context) : TransposeExecutor(context) {}

bool ov::intel_cpu::ACLTransposeExecutor::init(const PermuteParams &permuteParams,
                                               const std::vector <MemoryDescPtr> &srcDescs,
                                               const std::vector <MemoryDescPtr> &dstDescs,
                                               const dnnl::primitive_attr &attr) {
    return true;
}

void ov::intel_cpu::ACLTransposeExecutor::exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst,
                                               const int MB) {
}
