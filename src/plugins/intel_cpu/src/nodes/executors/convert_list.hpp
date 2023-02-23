// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor.hpp"

#include "convert.hpp"

#if defined(OV_CPU_WITH_ACL)
#include "acl/acl_convert.hpp"
#endif

#include "onednn/iml_type_mapper.h"
#include "common/primitive_cache.hpp"

namespace ov {
namespace intel_cpu {

struct ConvertExecutorDesc {
    ExecutorType executorType;
    ConvertExecutorBuilderCPtr builder;
};

const std::vector<ConvertExecutorDesc>& getConvertExecutorsList();

class ConvertExecutorFactory : public ExecutorFactory {
public:
    ConvertExecutorFactory(const ConvertAttrs& convertAttrs,
                           const std::vector<MemoryDescPtr>& srcDescs,
                           const std::vector<MemoryDescPtr>& dstDescs,
                           const ExecutorContext::CPtr context) : ExecutorFactory(context) {
        for (auto& desc : getConvertExecutorsList()) {
            if (desc.builder->isSupported(convertAttrs, srcDescs, dstDescs)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~ConvertExecutorFactory() = default;
    virtual ConvertExecutorPtr makeExecutor(const ConvertAttrs& convertAttrs,
                                            const std::vector<MemoryDescPtr>& srcDescs,
                                            const std::vector<MemoryDescPtr>& dstDescs,
                                            const dnnl::primitive_attr &attr) {
        auto build = [&](const ConvertExecutorDesc* desc) {
            switch (desc->executorType) {
                // case impl_desc_type::x64: {
                //     auto builder = [&](const JitConvertExecutor::Key& key) -> ConvertExecutorPtr {
                //         auto executor = desc->builder->makeExecutor();
                //         if (executor->init(ConvertAttrs, srcDescs, dstDescs, attr)) {
                //             return executor;
                //         } else {
                //             return nullptr;
                //         }
                //     };

                //     auto key = JitConvertExecutor::Key(ConvertAttrs, srcDescs, dstDescs, attr);
                //     auto res = runtimeCache->getOrCreate(key, builder);
                //     return res.first;
                // } break;
                default: {
                    auto executor = desc->builder->makeExecutor(context);
                    if (executor->init(convertAttrs, srcDescs, dstDescs, attr)) {
                        return executor;
                    }
                } break;
            }

            ConvertExecutorPtr ptr = nullptr;
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
    std::vector<ConvertExecutorDesc> supportedDescs;
    const ConvertExecutorDesc* chosenDesc = nullptr;
};

using ConvertExecutorFactoryPtr = std::shared_ptr<ConvertExecutorFactory>;
using ConvertExecutorFactoryCPtr = std::shared_ptr<const ConvertExecutorFactory>;

}   // namespace intel_cpu
}   // namespace ov