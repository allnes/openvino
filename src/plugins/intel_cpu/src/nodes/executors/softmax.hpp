// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "onednn/iml_type_mapper.h"
#include "executor.hpp"

namespace ov {
namespace intel_cpu {

struct SoftmaxKey {
    DnnlMemoryDescCPtr inp0;
    impl_desc_type implType;
    size_t axis;

    size_t hash() const;
    bool operator==(const SoftmaxKey& rhs) const;
};

struct SoftMaxAttrs {
    DnnlMemoryDescCPtr inp0;
    impl_desc_type implDescType;
    size_t axis = 0;
};

class SoftMaxExecutor {
public:
    explicit SoftMaxExecutor(const ExecutorContext::CPtr context);
    virtual bool init(const SoftMaxAttrs& softMaxAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr &attr) = 0;
    virtual void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void **post_ops_data_) = 0;
    virtual ~SoftMaxExecutor() = default;
    virtual impl_desc_type getImplType() const = 0;

protected:
    SoftMaxAttrs softMaxAttrs;
    const ExecutorContext::CPtr softMaxContext;
};
using SoftMaxExecutorPtr = std::shared_ptr<SoftMaxExecutor>;
using SoftMaxExecutorCPtr = std::shared_ptr<const SoftMaxExecutor>;

class SoftMaxExecutorBuilder {
public:
    ~SoftMaxExecutorBuilder() = default;
    virtual bool isSupported(const SoftMaxAttrs& softMaxAttrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    virtual SoftMaxExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const = 0;
};

using SoftMaxExecutorBuilderPtr = std::shared_ptr<SoftMaxExecutorBuilder>;
using SoftMaxExecutorBuilderCPtr = std::shared_ptr<const SoftMaxExecutorBuilder>;

}   // namespace intel_cpu
}   // namespace ov
