// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ref_transpose.hpp"
#include "ie_parallel.hpp"

template <typename T>
static void ov::intel_cpu::transpose_to_0312(const int MB, const MemoryPtr& srcMemPtr, MemoryPtr& dstMemPtr) {
    const auto src_data = reinterpret_cast<const T*>(srcMemPtr->GetPtr());
    auto dst_data = reinterpret_cast<T*>(dstMemPtr->GetPtr());

    const int DIM1 = srcMemPtr->getStaticDims()[1];
    const int DIM2 = srcMemPtr->getStaticDims()[2];
    const int DIM3 = srcMemPtr->getStaticDims()[3];

    parallel_for3d(MB, DIM1, DIM2, [&](const int n, const int dim1, const int dim2) {
        for (int dim3 = 0; dim3 < DIM3; ++dim3) {
            const int src_off = n * DIM1 * DIM2 * DIM3 +
                                dim1 * DIM2 * DIM3 +
                                dim2 * DIM3 +
                                dim3;
            const int dst_off = n * DIM1 * DIM2 * DIM3 +
                                dim3 * DIM1 * DIM2 +
                                dim1 * DIM2 +
                                dim2;

            dst_data[dst_off] = src_data[src_off];
        }
    });
}

template<typename T>
static void ov::intel_cpu::transpose_to_04123(const int MB, const MemoryPtr& srcMemPtr, MemoryPtr& dstMemPtr) {
    const auto src_data = reinterpret_cast<const T*>(srcMemPtr->GetPtr());
    auto dst_data = reinterpret_cast<T*>(dstMemPtr->GetPtr());

    const int DIM1 = srcMemPtr->getStaticDims()[1];
    const int DIM2 = srcMemPtr->getStaticDims()[2];
    const int DIM3 = srcMemPtr->getStaticDims()[3];
    const int DIM4 = srcMemPtr->getStaticDims()[4];

    parallel_for4d(MB, DIM1, DIM2, DIM3, [&](const int n, const int dim1, const int dim2, const int dim3) {
        for (int dim4 = 0; dim4 < DIM4; ++dim4) {
            const int src_off = n * DIM1 * DIM2 * DIM3 * DIM4 +
                                dim1 * DIM2 * DIM3 * DIM4 +
                                dim2 * DIM3 * DIM4 +
                                dim3 * DIM4 +
                                dim4;
            const int dst_off = n * DIM1 * DIM2 * DIM3 * DIM4 +
                                dim4 * DIM1 * DIM2 * DIM3 +
                                dim1 * DIM2 * DIM3 +
                                dim2 * DIM3 +
                                dim3;

            dst_data[dst_off] = src_data[src_off];
        }
    });
}

template<typename T>
static void ov::intel_cpu::transpose_to_051234(const int MB, const MemoryPtr& srcMemPtr, MemoryPtr& dstMemPtr) {
    const auto src_data = reinterpret_cast<const T*>(srcMemPtr->GetPtr());
    auto dst_data = reinterpret_cast<T*>(dstMemPtr->GetPtr());

    const int DIM1 = srcMemPtr->getStaticDims()[1];
    const int DIM2 = srcMemPtr->getStaticDims()[2];
    const int DIM3 = srcMemPtr->getStaticDims()[3];
    const int DIM4 = srcMemPtr->getStaticDims()[4];
    const int DIM5 = srcMemPtr->getStaticDims()[5];

    parallel_for5d(MB, DIM1, DIM2, DIM3, DIM4, [&](const int n, const int dim1, const int dim2, const int dim3, const int dim4) {
        for (int dim5 = 0; dim5 < DIM5; ++dim5) {
            const int src_off = n * DIM1 * DIM2 * DIM3 * DIM4 * DIM5 +
                                dim1 * DIM2 * DIM3 * DIM4 * DIM5 +
                                dim2 * DIM3 * DIM4 * DIM5 +
                                dim3 * DIM4 * DIM5 +
                                dim4 * DIM5 +
                                dim5;
            const int dst_off = n * DIM5 * DIM1 * DIM2 * DIM3 * DIM4 +
                                dim5 * DIM1 * DIM2 * DIM3 * DIM4 +
                                dim1 * DIM2 * DIM3 * DIM4 +
                                dim2 * DIM3 * DIM4 +
                                dim3 * DIM4 +
                                dim4;

            dst_data[dst_off] = src_data[src_off];
        }
    });
}

void ov::intel_cpu::RefTransposeExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const int MB) {
    MemoryPtr tmpSrc;
    tmpSrc->setDataHandle(const_cast<void *>(reinterpret_cast<const void *>(src[0]->GetPtr())));
    const size_t dataSize = src[0]->getDesc().getPrecision().size();
    TransposeContext ctx = {tmpSrc, dst[0], MB};
    OV_SWITCH(intel_cpu, TransposeOptimizedEmitter, ctx, dataSize,
              OV_CASE(1, InferenceEngine::PrecisionTrait<InferenceEngine::Precision::U8>::value_type),
              OV_CASE(2, InferenceEngine::PrecisionTrait<InferenceEngine::Precision::U16>::value_type),
              OV_CASE(4, InferenceEngine::PrecisionTrait<InferenceEngine::Precision::I32>::value_type));
}

ov::intel_cpu::RefTransposeExecutor::RefTransposeExecutor(const ExecutorContext::CPtr context) : TransposeExecutor(context) {}

bool ov::intel_cpu::RefTransposeExecutor::init(const ov::intel_cpu::TransposeParams &transposeParams,
                                               const std::vector<MemoryDescPtr> &srcDescs,
                                               const std::vector<MemoryDescPtr> &dstDescs,
                                               const dnnl::primitive_attr &attr) {
    if (transposeParams.transposeExecution != TransposeParams::REF) { return false; }
    return true;
}
