// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>
#include <vector>
#include <memory>
#include <caseless.hpp>
#include "executors/eltwise_list.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class Eltwise : public Node {
public:
    Eltwise(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void selectOptimalPrimitiveDescriptor() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override;
    bool canFuse(const NodePtr& node) const override;
    void appendPostOps(dnnl::post_ops& ops, const VectorDims &postOpDims, std::unordered_map<int, MemoryPtr>& postOpsMem, const int channelAxis = 1) override;
    void appendPostOps(dnnl::post_ops& ops, const VectorDims &postOpDims, std::vector<const void*>& postOpsMem, const int channelAxis = 1) override;
    bool appendAttrPostOps(DnnlPostOpsComposer& dnnlpoc, bool isLastPostOp, dnnl::memory::data_type outDataType, bool allowBinary = true);
    void fuseInto(NodePtr& parentNode) override;
    InferenceEngine::Precision getRuntimePrecision() const override;

    float getAlpha() const { return eltwiseAttrs.alpha; }
    float getBeta() const { return eltwiseAttrs.beta; }
    float getGamma() const { return eltwiseAttrs.gamma; }

    bool isWithBroadcast();
    bool isSpecialConvolutionAddFusing() const { return specialConvolutionAddFusing; }

    // bool needPrepareParams() const override;
    void prepareParams() override;

    void executeDynamicImpl(dnnl::stream strm) override;

    void setDynamicBatchLim(int lim) override;

    enum BroadcastingPolicy {
        PerChannel,
        PerTensor,
        Undefined,
    };

    BroadcastingPolicy getBroadcastingPolicy() const { return broadcastingPolicy; }

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    EltwiseAttrs eltwiseAttrs;
    std::shared_ptr<EltwiseExecutor> execPtr = nullptr;

    BroadcastingPolicy broadcastingPolicy;

    std::vector<bool> broadcastPolicy;
    bool isDynBatchEnabled = false;
    bool specialConvolutionAddFusing = false;

    std::vector<InferenceEngine::Precision> inpPrc;
    InferenceEngine::Precision outPrc;

    std::vector<float> scales = {};
    std::vector<float> shifts = {};
    MemoryPtr scalesMemory;
    MemoryPtr shiftsMemory;

    std::vector<float> depthwiseData = {};
    MemoryPtr depthwiseMemory;
    size_t depthwiseDataSize = 0;

    std::vector<const void*> fqDataPtrs;

    using Initializer = std::function<void(const std::shared_ptr<ngraph::Node>&, Eltwise& node)>;
    static const std::map<const ngraph::DiscreteTypeInfo, Initializer> initializers;

    static BroadcastingPolicy determineBroadcastingPolicy(const std::shared_ptr<ngraph::Node>& op);

    size_t getOpInputsNum() const;

    template <typename T>
    void appendPostOpsImpl(dnnl::post_ops& ops, const VectorDims &postOpDims, std::vector<T>& postOpsMem, const int channelAxis = 1);

    void appendMemory(const std::vector<float> &data, MemoryPtr &memPtr, std::vector<MemoryPtr>& postOpsMem);
    void appendMemory(const std::vector<float> &data, MemoryPtr &memPtr, std::vector<const void*>& postOpsMem);
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
