// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>
#include <memory>
#include <vector>

#include "executors/reduce_list.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class Reduce : public Node {
public:
    Reduce(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void prepareParams() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;
    int getFusingAxis() const override;
    bool canFuse(const NodePtr& node) const override;
    bool canBeInPlace() const override {
        return false;
    }

    bool isExecutable() const override;
    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    void setPostOps(dnnl::primitive_attr &attr, const VectorDims &postOpDims, bool initWeights = false);

    static const size_t REDUCE_DATA = 0;
    static const size_t REDUCE_INDEXES = 1;

    InferenceEngine::Precision input_prec, output_prec;

    dnnl::primitive_attr attr;
    std::vector<const void*> postOpsDataPtrs;

    static const std::map<const ngraph::DiscreteTypeInfo, std::function<void(const std::shared_ptr<ngraph::Node>& op, Reduce& node)>> initializers;

    std::string errorPrefix;

    ReduceAttrs reduceAttrs;

    std::shared_ptr<ReduceExecutor> execPtr = nullptr;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
