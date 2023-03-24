// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include "executors/softmax_list.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class SoftMax : public Node {
public:
    SoftMax(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    void initOptimalPrimitiveDescriptor() override;
    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;
    void getSupportedDescriptors() override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

    void prepareParams() override;
    void executeDynamicImpl(dnnl::stream strm) override;
    void execute(dnnl::stream strm) override;

private:
    std::unordered_map<int, dnnl::memory> softMaxPrimArgs;
    Primitive softMaxPrim;
    SoftMaxAttrs softMaxAttrs;
    std::shared_ptr<SoftMaxExecutor> execPtr = nullptr;
    NodeConfig config;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
