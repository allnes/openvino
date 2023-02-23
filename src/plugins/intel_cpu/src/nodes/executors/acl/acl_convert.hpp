// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/convert.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace ov {
namespace intel_cpu {

class ACLConvertExecutor : public ConvertExecutor {
public:
    ACLConvertExecutor(const ExecutorContext::CPtr context) : ConvertExecutor(context) {}

    bool init(const ConvertAttrs& convertAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;

    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              std::unordered_map<int, MemoryPtr> postOpsArgs) override;

    impl_desc_type getImplType() const override {
        return implType;
    }

private:
    impl_desc_type implType = impl_desc_type::acl;
    ConvertAttrs aclConvertAttrs;
    arm_compute::Tensor srcTensor;
    arm_compute::Tensor dstTensor;
    std::shared_ptr<arm_compute::NECopy> acl_copy;
    std::shared_ptr<arm_compute::NECast> acl_cast;
};

class ACLConvertExecutorBuilder : public ConvertExecutorBuilder {
public:
    bool isSupported(const ConvertAttrs& convertAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        return true;
    }

    ConvertExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<ACLConvertExecutor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov
