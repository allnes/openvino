// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/permute_kernel.h"
#include "executors/transpose.hpp"
#include "executors/common/ref_transpose.hpp"
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class Transpose : public Node {
public:
    Transpose(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    const InferenceEngine::SizeVector& getOrder() const {
        return order;
    }

    bool isExecutable() const override;
    bool needPrepareParams() const override;
    void prepareParams() override;

protected:
    void executeDynamicImpl(dnnl::stream strm) override;
    std::shared_ptr<ExecutorContext> transpose_context;

private:
    TransposeExecutorPtr execPtr = nullptr;

    class TransposeJitExecutor : public TransposeExecutor {
    public:
        explicit TransposeJitExecutor(const ExecutorContext::CPtr context);
        bool init(const PermuteParams& permuteParams,
                  const std::vector<MemoryDescPtr>& srcDescs,
                  const std::vector<MemoryDescPtr>& dstDescs,
                  const dnnl::primitive_attr &attr) override {
            pKernel = std::make_shared<PermuteKernel>(permuteParams);
            return true;
        }
        void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const int MB) override;

        std::shared_ptr<PermuteKernel> pKernel;
    };

    InferenceEngine::SizeVector order;
    InferenceEngine::Precision prec;
    bool isOptimized = false;

    const std::vector<std::vector<size_t>> optimizedOrders = {
            std::vector<size_t>{0, 3, 1, 2},
            std::vector<size_t>{0, 4, 1, 2, 3},
            std::vector<size_t>{0, 5, 1, 2, 3, 4},
    };

    PermuteParams params;

    bool isInputOrderConst = false;

    static constexpr size_t INPUT_DATA_IDX = 0lu;
    static constexpr size_t INPUT_ORDER_IDX = 1lu;

    bool performAsReorder = false;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
