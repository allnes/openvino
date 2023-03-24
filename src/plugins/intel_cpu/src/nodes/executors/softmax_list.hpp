// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor.hpp"

#include "softmax.hpp"
#if defined(OV_CPU_WITH_ACL)
//#include "acl/acl_softmax.hpp"
#endif
#include "dnnl/dnnl_softmax.hpp"

#include "onednn/iml_type_mapper.h"
#include "common/primitive_cache.hpp"

namespace ov {
namespace intel_cpu {

struct SoftMaxExecutorDesc {
    ExecutorType executorType;
    SoftMaxExecutorBuilderCPtr builder;
};

const std::vector<SoftMaxExecutorDesc>& getSoftMaxExecutorsList();

class SoftMaxExecutorFactory : public ExecutorFactory {
public:
    SoftMaxExecutorFactory(const SoftMaxAttrs& softMaxAttrs,
                       const std::vector<MemoryDescPtr>& srcDescs,
                       const std::vector<MemoryDescPtr>& dstDescs,
                       const ExecutorContext::CPtr context) : ExecutorFactory(context) {
        for (auto& desc : getSoftMaxExecutorsList()) {
            if (desc.builder->isSupported(softMaxAttrs, srcDescs, dstDescs)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~SoftMaxExecutorFactory() = default;
    virtual SoftMaxExecutorPtr makeExecutor(const SoftMaxAttrs& softMaxAttrs,
                                        const std::vector<MemoryDescPtr>& srcDescs,
                                        const std::vector<MemoryDescPtr>& dstDescs,
                                        const dnnl::primitive_attr &attr) {
        auto build = [&](const SoftMaxExecutorDesc* desc) {
            //TODO: enable exeuctor cache for JIT executor
            switch (desc->executorType) {
                default: {
                    auto executor = desc->builder->makeExecutor(context);
                    if (executor->init(softMaxAttrs, srcDescs, dstDescs, attr)) {
                        return executor;
                    }
                } break;
            }

            SoftMaxExecutorPtr ptr = nullptr;
            return ptr;
        };


        if (chosenDesc) {
            if (auto executor = build(chosenDesc)) {
                return executor;
            }
        }

        for (const auto& sd : supportedDescs) {
            if (auto executor = build(&sd)) {
                chosenDesc = &sd;
                return executor;
            }
        }

        IE_THROW() << "Supported executor is not found";
    }

private:
    std::vector<SoftMaxExecutorDesc> supportedDescs;
    const SoftMaxExecutorDesc* chosenDesc = nullptr;
};

using SoftMaxExecutorFactoryPtr = std::shared_ptr<SoftMaxExecutorFactory>;
using SoftMaxExecutorFactoryCPtr = std::shared_ptr<const SoftMaxExecutorFactory>;

}   // namespace intel_cpu
}   // namespace ov