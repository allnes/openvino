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
    aclEltwiseAttrs = eltwiseAttrs;

    for (const auto &desc : srcDescs) {
        if (desc->getShape().getRank() > 4)
            return false;
    }

    for (const auto &desc : dstDescs) {
        if (desc->getShape().getRank() > 4)
            return false;
    }

    auto src1Dims = srcDescs[0]->getShape().getStaticDims();
    auto src2Dims = srcDescs[1]->getShape().getStaticDims();
    auto dstDims = dstDescs[0]->getShape().getStaticDims();

    TensorInfo src1TensorInfo = TensorInfo(eltwiseShapeCast(src1Dims), 1, DataType::F32, DataLayout::NCHW);
    TensorInfo src2TensorInfo = TensorInfo(eltwiseShapeCast(src2Dims), 1, DataType::F32, DataLayout::NCHW);
    TensorInfo dstTensorInfo = TensorInfo(eltwiseShapeCast(dstDims), 1, DataType::F32, DataLayout::NCHW);



    src1Tensor.allocator()->init(src1TensorInfo);
    src2Tensor.allocator()->init(src2TensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);
    std::cout << static_cast<int>(eltwiseAttrs.algorithm) << std::endl;

    switch (aclEltwiseAttrs.algorithm) {
        case Algorithm::EltwiseAdd:
            if (!NEArithmeticAddition::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ConvertPolicy::SATURATE))
                return false;
            acl_add = std::make_unique<NEArithmeticAddition>();
            acl_add->configure(&src1Tensor, &src2Tensor, &dstTensor, ConvertPolicy::SATURATE);
            break;
        case Algorithm::EltwiseSubtract:
            if (!NEArithmeticSubtraction::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ConvertPolicy::SATURATE))
                return false;
            acl_sub = std::make_unique<NEArithmeticSubtraction>();
            acl_sub->configure(&src1Tensor, &src2Tensor, &dstTensor, ConvertPolicy::SATURATE);
            break;
        case Algorithm::EltwiseMultiply:
            if (!NEPixelWiseMultiplication::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo,
                                                     1.0f, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO))
                return false;
            acl_mul = std::make_unique<NEPixelWiseMultiplication>();
            acl_mul->configure(&src1Tensor, &src2Tensor, &dstTensor, 1.0f, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO);
            break;
        case Algorithm::EltwiseDivide:
            if (!NEElementwiseDivision::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo))
                return false;
            acl_div = std::make_unique<NEElementwiseDivision>();
            acl_div->configure(&src1Tensor, &src2Tensor, &dstTensor);
            break;
        case Algorithm::EltwiseMaximum:
            if (!NEElementwiseMax::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo))
                return false;
            acl_max = std::make_unique<NEElementwiseMax>();
            acl_max->configure(&src1Tensor, &src2Tensor, &dstTensor);
            break;
        case Algorithm::EltwiseMinimum:
            if (!NEElementwiseMin::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo))
                return false;
            acl_min = std::make_unique<NEElementwiseMin>();
            acl_min->configure(&src1Tensor, &src2Tensor, &dstTensor);
            break;
        case Algorithm::EltwiseSquaredDifference:
            if (!NEElementwiseSquaredDiff::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo))
                return false;
            acl_sqdiff = std::make_unique<NEElementwiseSquaredDiff>();
            acl_sqdiff->configure(&src1Tensor, &src2Tensor, &dstTensor);
            break;
        case Algorithm::EltwisePowerStatic:
        case Algorithm::EltwisePowerDynamic:
            if (!NEElementwisePower::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo))
                return false;
            acl_pow = std::make_unique<NEElementwisePower>();
            acl_pow->configure(&src1Tensor, &src2Tensor, &dstTensor);
            break;
        case Algorithm::EltwiseEqual:
            if (!NEElementwiseComparison::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ComparisonOperation::Equal))
                return false;
            acl_comp = std::make_unique<NEElementwiseComparison>();
            acl_comp->configure(&src1Tensor, &src2Tensor, &dstTensor, ComparisonOperation::Equal);
        case Algorithm::EltwiseNotEqual:
            if (!NEElementwiseComparison::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ComparisonOperation::NotEqual))
                return false;
            acl_comp = std::make_unique<NEElementwiseComparison>();
            acl_comp->configure(&src1Tensor, &src2Tensor, &dstTensor, ComparisonOperation::NotEqual);
        case Algorithm::EltwiseGreater:
            if (!NEElementwiseComparison::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ComparisonOperation::Greater))
                return false;
            acl_comp = std::make_unique<NEElementwiseComparison>();
            acl_comp->configure(&src1Tensor, &src2Tensor, &dstTensor, ComparisonOperation::Greater);
        case Algorithm::EltwiseGreaterEqual:
            if (!NEElementwiseComparison::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ComparisonOperation::GreaterEqual))
                return false;
            acl_comp = std::make_unique<NEElementwiseComparison>();
            acl_comp->configure(&src1Tensor, &src2Tensor, &dstTensor, ComparisonOperation::GreaterEqual);
        case Algorithm::EltwiseLess:
            if (!NEElementwiseComparison::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ComparisonOperation::Less))
                return false;
            acl_comp = std::make_unique<NEElementwiseComparison>();
            acl_comp->configure(&src1Tensor, &src2Tensor, &dstTensor, ComparisonOperation::Less);
        case Algorithm::EltwiseLessEqual:
            if (!NEElementwiseComparison::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ComparisonOperation::LessEqual))
                return false;
            acl_comp = std::make_unique<NEElementwiseComparison>();
            acl_comp->configure(&src1Tensor, &src2Tensor, &dstTensor, ComparisonOperation::LessEqual);
        default:
            IE_THROW() << "Unsupported operation type for ACL Eltwise executor";
    }
    return true;
}

void AclEltwiseExecutor::exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst,
                              const void *post_ops_data_) {
    src1Tensor.allocator()->import_memory(src[0]->GetPtr());
    src2Tensor.allocator()->import_memory(src[1]->GetPtr());
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());
    std::cout << static_cast<int>(eltwiseAttrs.algorithm) << std::endl;
    switch (aclEltwiseAttrs.algorithm) {
        case Algorithm::EltwiseAdd: acl_add->run(); break;
        case Algorithm::EltwiseSubtract: acl_sub->run(); break;
        case Algorithm::EltwiseMultiply: acl_mul->run(); break;
        case Algorithm::EltwiseDivide: acl_div->run(); break;
        case Algorithm::EltwiseMaximum: acl_max->run(); break;
        case Algorithm::EltwiseMinimum: acl_min->run(); break;
        case Algorithm::EltwiseSquaredDifference: acl_sqdiff->run(); break;
        case Algorithm::EltwisePowerStatic:
        case Algorithm::EltwisePowerDynamic: acl_pow->run(); break;
        case Algorithm::EltwiseEqual:
        case Algorithm::EltwiseNotEqual:
        case Algorithm::EltwiseGreater:
        case Algorithm::EltwiseGreaterEqual:
        case Algorithm::EltwiseLess:
        case Algorithm::EltwiseLessEqual: acl_comp->run(); break;
        default: IE_THROW() << "Unsupported operation type for ACL Eltwise executor";
    }

    src1Tensor.allocator()->free();
    src2Tensor.allocator()->free();
    dstTensor.allocator()->free();
}
}   // namespace intel_cpu
}   // namespace ov
