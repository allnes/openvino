// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_eltwise.hpp"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

TensorShape eltwiseShapeCast(const VectorDims &dims) {
    arm_compute::TensorShape tensorShape;
    for (std::size_t i = 0; i < dims.size(); ++i) {
        tensorShape.set(dims.size() - i - 1, dims[i], false);
    }
    if (tensorShape.num_dimensions() == 0) {
        tensorShape.set(0, 1, false);
        tensorShape.set_num_dimensions(1);
    }
    return tensorShape;
}

AclEltwiseExecutor::AclEltwiseExecutor() : EltwiseExecutor() {}

bool AclEltwiseExecutor::init(const EltwiseAttrs &eltwiseAttrs, const std::vector<MemoryDescPtr> &srcDescs,
                              const std::vector<MemoryDescPtr> &dstDescs,
                              const std::vector<EltwisePostOp> &postOps) {
    for (const auto& desc : srcDescs) {
        if (desc->getPrecision() != InferenceEngine::Precision::FP32)
            return false;
    }
    for (const auto& desc : dstDescs) {
        if (desc->getPrecision() != InferenceEngine::Precision::FP32)
            return false;
    }
    auto src1Dims = srcDescs[0]->getShape().getStaticDims();
    auto src2Dims = srcDescs[1]->getShape().getStaticDims();
    auto dstDims = dstDescs[0]->getShape().getStaticDims();

    TensorInfo src1TensorInfo = TensorInfo(eltwiseShapeCast(src1Dims), 1, DataType::F32, DataLayout::NCHW);
    TensorInfo src2TensorInfo = TensorInfo(eltwiseShapeCast(src2Dims), 1, DataType::F32, DataLayout::NCHW);
    TensorInfo dstTensorInfo  = TensorInfo(eltwiseShapeCast(dstDims), 1, DataType::F32, DataLayout::NCHW);

    if (!arm_compute::NEArithmeticAddition::validate(&src1TensorInfo,
                                                     &src2TensorInfo,
                                                     &dstTensorInfo,
                                                     arm_compute::ConvertPolicy::SATURATE))
        return false;

    src1Tensor.allocator()->init(src1TensorInfo);
    src2Tensor.allocator()->init(src2TensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    auto temp_acl_op = std::make_unique<arm_compute::NEArithmeticAddition>();
    temp_acl_op->configure(&src1Tensor, &src2Tensor, &dstTensor, arm_compute::ConvertPolicy::SATURATE);

    return true;
}

    void AclEltwiseExecutor::exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst,
                                  const void *post_ops_data_) {
        src1Tensor.allocator()->import_memory(src[0]->GetPtr());
        src2Tensor.allocator()->import_memory(src[1]->GetPtr());
        dstTensor.allocator()->import_memory(dst[0]->GetPtr());

        acl_op->run();

        src1Tensor.allocator()->free();
        src2Tensor.allocator()->free();
        dstTensor.allocator()->free();
    }
}   // namespace intel_cpu
}   // namespace ov