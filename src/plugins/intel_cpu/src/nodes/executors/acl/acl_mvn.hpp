// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "acl_utils.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "nodes/executors/mvn.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

class AclMVNExecutor : public MVNExecutor {
public:
    AclMVNExecutor(const MVNAttrs& mvnAttrs,
                   const MemoryArgs& memory,
                   const ExecutorContext::CPtr& context);

    bool init(const MVNAttrs& mvnAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr& attr) override;
    void executeImpl(const MemoryArgs& memory) override;

    [[nodiscard]] impl_desc_type getImplType() const override {
        return implType;
    }
    
    static bool supports(const MVNAttrs& mvnAttrs,
                        const std::vector<MemoryDescPtr>& srcDescs,
                        const std::vector<MemoryDescPtr>& dstDescs);

private:
    impl_desc_type implType = impl_desc_type::acl;

    arm_compute::Tensor srcTensor;
    arm_compute::Tensor dstTensor;
    std::unique_ptr<arm_compute::NEMeanStdDevNormalizationLayer> mvn = nullptr;
};

// Builder is now in mvn_implementations.cpp using ExecutorFactoryBuilder template

}  // namespace ov::intel_cpu
