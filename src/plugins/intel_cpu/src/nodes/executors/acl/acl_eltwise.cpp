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
    this->aclEltwiseAttrs = eltwiseAttrs;

    auto src1Dims = srcDescs[0]->getShape().getDims();
    auto src2Dims = srcDescs[1]->getShape().getDims();
    auto dstDims  = dstDescs[0]->getShape().getDims();
    auto test = eltwiseShapeCast(src1Dims);
    TensorInfo src1TensorInfo = TensorInfo(eltwiseShapeCast(src1Dims), 1,
                                           precisionToAclDataType(srcDescs[0]->getPrecision()),
                                           getAclDataLayoutByMemoryDesc(srcDescs[0]));
    TensorInfo src2TensorInfo = TensorInfo(eltwiseShapeCast(src2Dims), 1,
                                           precisionToAclDataType(srcDescs[1]->getPrecision()),
                                           getAclDataLayoutByMemoryDesc(srcDescs[1]));
    TensorInfo dstTensorInfo = TensorInfo(eltwiseShapeCast(dstDims), 1,
                                          precisionToAclDataType(dstDescs[0]->getPrecision()),
                                          getAclDataLayoutByMemoryDesc(dstDescs[0]));

    src1Tensor.allocator()->init(src1TensorInfo);
    src2Tensor.allocator()->init(src2TensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    switch (aclEltwiseAttrs.algorithm) {
        case Algorithm::EltwiseAdd:
            if (!NEArithmeticAddition::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ConvertPolicy::SATURATE))
                return false;
            acl_add = std::make_unique<NEArithmeticAddition>();
            acl_add->configure(&src1Tensor, &src2Tensor, &dstTensor, ConvertPolicy::SATURATE);
            run_func = [this]{ acl_add->run(); };
            break;
        case Algorithm::EltwiseSubtract:
            if (!NEArithmeticSubtraction::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ConvertPolicy::SATURATE))
                return false;
            acl_sub = std::make_unique<NEArithmeticSubtraction>();
            acl_sub->configure(&src1Tensor, &src2Tensor, &dstTensor, ConvertPolicy::SATURATE);
            run_func = [this]{ acl_sub->run(); };
            break;
        case Algorithm::EltwiseMultiply:
            if (!NEPixelWiseMultiplication::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo,
                                                     1.0f, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO))
                return false;
            acl_mul = std::make_unique<NEPixelWiseMultiplication>();
            acl_mul->configure(&src1Tensor, &src2Tensor, &dstTensor, 1.0f, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO);
            run_func = [this]{ acl_mul->run(); };
            break;
        case Algorithm::EltwiseDivide:
            if (!NEElementwiseDivision::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo))
                return false;
            acl_div = std::make_unique<NEElementwiseDivision>();
            acl_div->configure(&src1Tensor, &src2Tensor, &dstTensor);
            run_func = [this]{ acl_div->run(); };
            break;
        case Algorithm::EltwiseMaximum:
            if (!NEElementwiseMax::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo))
                return false;
            acl_max = std::make_unique<NEElementwiseMax>();
            acl_max->configure(&src1Tensor, &src2Tensor, &dstTensor);
            run_func = [this]{ acl_max->run(); };
            break;
        case Algorithm::EltwiseMinimum:
            if (!NEElementwiseMin::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo))
                return false;
            acl_min = std::make_unique<NEElementwiseMin>();
            acl_min->configure(&src1Tensor, &src2Tensor, &dstTensor);
            run_func = [this]{ acl_min->run(); };
            break;
        case Algorithm::EltwiseSquaredDifference:
            if (!NEElementwiseSquaredDiff::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo))
                return false;
            acl_sqdiff = std::make_unique<NEElementwiseSquaredDiff>();
            acl_sqdiff->configure(&src1Tensor, &src2Tensor, &dstTensor);
            run_func = [this]{ acl_sqdiff->run(); };
            break;
        case Algorithm::EltwisePowerDynamic:
            if (!NEElementwisePower::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo))
                return false;
            acl_pow = std::make_unique<NEElementwisePower>();
            acl_pow->configure(&src1Tensor, &src2Tensor, &dstTensor);
            run_func = [this]{ acl_pow->run(); };
            break;
        case Algorithm::EltwiseEqual:
            if (!NEElementwiseComparison::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ComparisonOperation::Equal))
                return false;
            acl_comp = std::make_unique<NEElementwiseComparison>();
            acl_comp->configure(&src1Tensor, &src2Tensor, &dstTensor, ComparisonOperation::Equal);
            run_func = [this]{ acl_comp->run(); };
            break;
        case Algorithm::EltwiseNotEqual:
            if (!NEElementwiseComparison::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ComparisonOperation::NotEqual))
                return false;
            acl_comp = std::make_unique<NEElementwiseComparison>();
            acl_comp->configure(&src1Tensor, &src2Tensor, &dstTensor, ComparisonOperation::NotEqual);
            run_func = [this]{ acl_comp->run(); };
            break;
        case Algorithm::EltwiseGreater:
            if (!NEElementwiseComparison::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ComparisonOperation::Greater))
                return false;
            acl_comp = std::make_unique<NEElementwiseComparison>();
            acl_comp->configure(&src1Tensor, &src2Tensor, &dstTensor, ComparisonOperation::Greater);
            run_func = [this]{ acl_comp->run(); };
            break;
        case Algorithm::EltwiseGreaterEqual:
            if (!NEElementwiseComparison::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ComparisonOperation::GreaterEqual))
                return false;
            acl_comp = std::make_unique<NEElementwiseComparison>();
            acl_comp->configure(&src1Tensor, &src2Tensor, &dstTensor, ComparisonOperation::GreaterEqual);
            run_func = [this]{ acl_comp->run(); };
            break;
        case Algorithm::EltwiseLess:
            if (!NEElementwiseComparison::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ComparisonOperation::Less))
                return false;
            acl_comp = std::make_unique<NEElementwiseComparison>();
            acl_comp->configure(&src1Tensor, &src2Tensor, &dstTensor, ComparisonOperation::Less);
            run_func = [this]{ acl_comp->run(); };
            break;
        case Algorithm::EltwiseLessEqual:
            if (!NEElementwiseComparison::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ComparisonOperation::LessEqual))
                return false;
            acl_comp = std::make_unique<NEElementwiseComparison>();
            acl_comp->configure(&src1Tensor, &src2Tensor, &dstTensor, ComparisonOperation::LessEqual);
            run_func = [this]{ acl_comp->run(); };
            break;
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

    std::cout << "Run ACL Eltwise executor" << std::endl;
    run_func();
//    for (int i = 0; i < 4 * 5 * 17 * 2; i++) {
//        std::cout << i << " : " << reinterpret_cast<float*>(src1Tensor.allocator()->data())[i] << " + ";
//        std::cout << reinterpret_cast<float*>(src2Tensor.allocator()->data())[i] << " = ";
//        std::cout << reinterpret_cast<float*>(dstTensor.allocator()->data())[i] << " === ";
//        std::cout << reinterpret_cast<float*>(src[0]->GetPtr())[i] << " + ";
//        std::cout << reinterpret_cast<float*>(src[1]->GetPtr())[i] << " = ";
//        std::cout << reinterpret_cast<float*>(dst[0]->GetPtr())[i] << std::endl;
//    }
    src1Tensor.allocator()->free();
    src2Tensor.allocator()->free();
    dstTensor.allocator()->free();
}
}   // namespace intel_cpu
}   // namespace ov
