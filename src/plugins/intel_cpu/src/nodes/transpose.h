// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/permute_kernel.h"
#include "executors/transpose.hpp"
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

template <typename T>
static void transpose_to_0312(const int MB, const MemoryPtr& srcMemPtr, MemoryPtr& dstMemPtr);
template<typename T>
static void transpose_to_04123(const int MB, const MemoryPtr& srcMemPtr, MemoryPtr& dstMemPtr);
template<typename T>
static void transpose_to_051234(const int MB, const MemoryPtr& srcMemPtr, MemoryPtr& dstMemPtr);

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

private:
    executorPtr execPtr = nullptr;

    class TransposeJitExecutor : public TransposeExecutor {
    public:
        TransposeJitExecutor(const PermuteParams& params);
        bool init(const PermuteParams& permuteParams,
                  const std::vector<MemoryDescPtr>& srcDescs,
                  const std::vector<MemoryDescPtr>& dstDescs,
                  const dnnl::primitive_attr &attr) override { return true; }
        void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const int MB) override;

        std::shared_ptr<PermuteKernel> pKernel;
    };

    class TransposeRefExecutor : public TransposeExecutor {
    public:
        TransposeRefExecutor() = default;
        bool init(const PermuteParams& permuteParams,
                  const std::vector<MemoryDescPtr>& srcDescs,
                  const std::vector<MemoryDescPtr>& dstDescs,
                  const dnnl::primitive_attr &attr) override { return true; }
        void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const int MB) override;
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

    struct TransposeContext {
        MemoryPtr srcMemPtr;
        MemoryPtr dstMemPtr;
        int MB;
    };

    template<typename T>
    struct TransposeOptimizedEmitter {
        void operator()(TransposeContext& ctx) {
            switch (ctx.srcMemPtr->getStaticDims().size()) {
                case 4:
                    transpose_to_0312<T>(ctx.MB, ctx.srcMemPtr, ctx.dstMemPtr);
                    break;
                case 5:
                    transpose_to_04123<T>(ctx.MB, ctx.srcMemPtr, ctx.dstMemPtr);
                    break;
                case 6:
                    transpose_to_051234<T>(ctx.MB, ctx.srcMemPtr, ctx.dstMemPtr);
                    break;
                default:
                    IE_THROW() << "Transpose supports optimized execution with only 4D, 5D and 6D shapes";
            }
        }
    };

    bool isInputOrderConst = false;

    static constexpr size_t INPUT_DATA_IDX = 0lu;
    static constexpr size_t INPUT_ORDER_IDX = 1lu;

    bool performAsReorder = false;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
