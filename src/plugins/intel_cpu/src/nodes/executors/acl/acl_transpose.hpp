// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../transpose.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace ov {
namespace intel_cpu {

class ACLTransposeExecutor : public TransposeExecutor {
public:
    ACLTransposeExecutor(const ExecutorContext::CPtr context);
    bool init(const PermuteParams& permuteParams,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const int MB) override;
    impl_desc_type getImplType() const {
        return implType;
    }
private:
    const ExecutorContext::CPtr acl_context;
    impl_desc_type implType = impl_desc_type::acl;
    PermuteParams aclPermuteParams;
};

class ACLTransposeExecutorBuilder : public TransposeExecutorBuilder {
public:
    bool isSupported(const PermuteParams& permuteParams,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        return true;
    }

    TransposeExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<ACLTransposeExecutor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov