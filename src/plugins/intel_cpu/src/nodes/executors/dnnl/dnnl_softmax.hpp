// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/softmax.hpp"

namespace ov {
namespace intel_cpu {

class DNNLSoftMaxExecutor : public SoftMaxExecutor {
public:
    explicit DNNLSoftMaxExecutor(const ExecutorContext::CPtr context);
    bool init(const SoftMaxAttrs& softMaxAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void **post_ops_data_) override;
    impl_desc_type getImplType() const override { return impl_desc_type::unknown; }
private:
    std::shared_ptr<dnnl::softmax_forward> softMaxPrim;
    dnnl::stream dnnlStream;
    MemoryPtr scratchpadMemory;
};

class DNNLSoftMaxExecutorBuilder : public SoftMaxExecutorBuilder {
public:
    bool isSupported(const SoftMaxAttrs& SoftMaxAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        return true;
    }

    SoftMaxExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<DNNLSoftMaxExecutor>(context);
    }
};
}   // namespace intel_cpu
}   // namespace ov
