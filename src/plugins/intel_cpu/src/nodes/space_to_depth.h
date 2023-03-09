// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>
#include "common/permute_kernel.h"

namespace ov {
namespace intel_cpu {
namespace node {

class SpaceToDepth : public Node {
public:
    SpaceToDepth(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    void prepareParams() override;

    enum Mode { BLOCKS_FIRST = 0, DEPTH_FIRST = 1 };

    struct SpaceToDepthAttrs {
        LayoutType layoutType;
        Mode mode;
        size_t blockSize = 0lu;
        size_t blockStep = 1lu;
        size_t dataSize = 1lu;
        size_t nSpatialDims = 0lu;
        VectorDims srcBlockedDims;
        VectorDims destBlockedDims;
        size_t hash() const;
        bool operator==(const SpaceToDepthAttrs& rhs) const;
    };

protected:
    void executeDynamicImpl(dnnl::stream strm) override;

private:
    SpaceToDepthAttrs attrs;

    class JITSpaceToDepthExecutor {
    public:
        explicit JITSpaceToDepthExecutor(const ExecutorContext::CPtr context);
        bool init(const SpaceToDepthAttrs& spaceToDepthAttrs,
                  const std::vector<MemoryDescPtr>& srcDescs,
                  const std::vector<MemoryDescPtr>& dstDescs,
                  const dnnl::primitive_attr &attr);
        void exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst, const int MB);
        ~JITSpaceToDepthExecutor() = default;
    protected:
        const ExecutorContext::CPtr spaceToDepthContext;
    private:
        std::unique_ptr<PermuteKernel> permuteKernel;
    };
    using SpaceToDepthExecutorPtr = std::shared_ptr<JITSpaceToDepthExecutor>;
    using SpaceToDepthExecutorCPtr = std::shared_ptr<const JITSpaceToDepthExecutor>;
    SpaceToDepthExecutorPtr execPtr = nullptr;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
