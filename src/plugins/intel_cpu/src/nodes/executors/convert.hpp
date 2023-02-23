// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "onednn/iml_type_mapper.h"
#include "executor.hpp"

namespace ov {
namespace intel_cpu {

struct ConvertAttrs {
    InferenceEngine::Precision srcPrc;
    InferenceEngine::Precision dstPrc;
    Shape srcShape;
    Shape dstShape;
};

class ConvertExecutor {
public:
    ConvertExecutor(const ExecutorContext::CPtr context);
    virtual bool init(const ConvertAttrs& mvnAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr &attr) = 0;

    virtual void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, std::unordered_map<int, MemoryPtr> postOpsArgs) = 0;
    virtual ~ConvertExecutor() = default;

    virtual impl_desc_type getImplType() const = 0;

protected:
    ConvertAttrs convertAttrs;
    const ExecutorContext::CPtr context;
};

using ConvertExecutorPtr = std::shared_ptr<ConvertExecutor>;
using ConvertExecutorCPtr = std::shared_ptr<const ConvertExecutor>;

class ConvertExecutorBuilder {
public:
    ~ConvertExecutorBuilder() = default;
    virtual bool isSupported(const ConvertAttrs& convertAttrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    virtual ConvertExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const = 0;
};

using ConvertExecutorBuilderPtr = std::shared_ptr<ConvertExecutorBuilder>;
using ConvertExecutorBuilderCPtr = std::shared_ptr<const ConvertExecutorBuilder>;

}   // namespace intel_cpu
}   // namespace ov