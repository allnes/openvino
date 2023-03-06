// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../transpose.hpp"

namespace ov {
namespace intel_cpu {

template <typename T>
static void transpose_to_0312(const int MB, const MemoryPtr& srcMemPtr, MemoryPtr& dstMemPtr);
template<typename T>
static void transpose_to_04123(const int MB, const MemoryPtr& srcMemPtr, MemoryPtr& dstMemPtr);
template<typename T>
static void transpose_to_051234(const int MB, const MemoryPtr& srcMemPtr, MemoryPtr& dstMemPtr);

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

class TransposeRefExecutor : public TransposeExecutor {
public:
    explicit TransposeRefExecutor(const ExecutorContext::CPtr context);
    bool init(const PermuteParams &permuteParams,
              const std::vector<MemoryDescPtr> &srcDescs,
              const std::vector<MemoryDescPtr> &dstDescs,
              const dnnl::primitive_attr &attr) override { return true; }

    void exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst, const int MB) override;
};

} // namespace intel_cpu
} // namespace ov