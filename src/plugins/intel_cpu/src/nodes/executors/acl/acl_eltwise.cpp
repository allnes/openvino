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

std::vector<ov::intel_cpu::Dim> reshape_sizes(std::vector<ov::intel_cpu::Dim> dims) {
    const size_t MAX_NUM_SHAPE = arm_compute::MAX_DIMS;
    std::vector<ov::intel_cpu::Dim> result_dims(MAX_NUM_SHAPE - 1);
    if (dims.size() >= MAX_NUM_SHAPE) {
        for (int i = 0; i < MAX_NUM_SHAPE - 1; i++) {
            result_dims[i] = dims[i];
        }
        for (int i = MAX_NUM_SHAPE - 1; i < dims.size(); i++) {
            result_dims[MAX_NUM_SHAPE - 2] *= dims[i];
        }
    } else {
        result_dims = dims;
    }
    return result_dims;
}

AclEltwiseExecutor::AclEltwiseExecutor() : EltwiseExecutor() {}

bool AclEltwiseExecutor::init(const EltwiseAttrs &eltwiseAttrs, const std::vector<MemoryDescPtr> &srcDescs,
                              const std::vector<MemoryDescPtr> &dstDescs,
                              const std::vector<EltwisePostOp> &postOps) {
    if (!postOps.empty()) { return false; }

    aclEltwiseAttrs = eltwiseAttrs;

    if (srcDescs.size() == 1) { is_unary_op = true; }

    std::vector<ov::intel_cpu::Dim> src1Dims, src2Dims, dstDims;
    TensorInfo src1TensorInfo, src2TensorInfo, dstTensorInfo;

    { // 1st tensor (unary op)
        src1Dims = reshape_sizes(srcDescs[0]->getShape().getDims());
        src1TensorInfo = TensorInfo(eltwiseShapeCast(src1Dims), 1,
                                    precisionToAclDataType(srcDescs[0]->getPrecision()),
                                    getAclDataLayoutByMemoryDesc(srcDescs[0]));
        src1Tensor.allocator()->init(src1TensorInfo);
    }

    // 2nd tensor (binary op)
    if (!is_unary_op) {
        src2Dims = reshape_sizes(srcDescs[1]->getShape().getDims());
        src2TensorInfo = TensorInfo(eltwiseShapeCast(src2Dims), 1,
                                    precisionToAclDataType(srcDescs[1]->getPrecision()),
                                    getAclDataLayoutByMemoryDesc(srcDescs[1]));
        src2Tensor.allocator()->init(src2TensorInfo);
    }

    { // dest tensor
        dstDims = reshape_sizes(dstDescs[0]->getShape().getDims());
        dstTensorInfo = TensorInfo(eltwiseShapeCast(dstDims), 1,
                                   precisionToAclDataType(dstDescs[0]->getPrecision()),
                                   getAclDataLayoutByMemoryDesc(dstDescs[0]));

        dstTensor.allocator()->init(dstTensorInfo);
    }

    switch (aclEltwiseAttrs.algorithm) {
        case Algorithm::EltwiseAdd:
            if (!NEArithmeticAddition::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ConvertPolicy::SATURATE))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEArithmeticAddition>();
                acl_op->configure(&src1Tensor, &src2Tensor, &dstTensor, ConvertPolicy::SATURATE);
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseMultiply:
            if (!NEPixelWiseMultiplication::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo,
                                                     1.0f, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEPixelWiseMultiplication>();
                acl_op->configure(&src1Tensor, &src2Tensor, &dstTensor, 1.0f, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO);
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseSubtract:
            if (!NEArithmeticSubtraction::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ConvertPolicy::SATURATE))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEArithmeticSubtraction>();
                acl_op->configure(&src1Tensor, &src2Tensor, &dstTensor, ConvertPolicy::SATURATE);
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseDivide:
            if (!NEElementwiseDivision::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEElementwiseDivision>();
                acl_op->configure(&src1Tensor, &src2Tensor, &dstTensor);
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseMaximum:
            if (!NEElementwiseMax::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEElementwiseMax>();
                acl_op->configure(&src1Tensor, &src2Tensor, &dstTensor);
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseMinimum:
            if (!NEElementwiseMin::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEElementwiseMin>();
                acl_op->configure(&src1Tensor, &src2Tensor, &dstTensor);
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseSquaredDifference:
            if (!NEElementwiseSquaredDiff::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEElementwiseSquaredDiff>();
                acl_op->configure(&src1Tensor, &src2Tensor, &dstTensor);
                acl_op->run();
            };
            break;
        case Algorithm::EltwisePowerDynamic:
            if (!NEElementwisePower::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEElementwisePower>();
                acl_op->configure(&src1Tensor, &src2Tensor, &dstTensor);
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseEqual:
            if (!NEElementwiseComparison::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ComparisonOperation::Equal))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEElementwiseComparison>();
                acl_op->configure(&src1Tensor, &src2Tensor, &dstTensor, ComparisonOperation::Equal);
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseNotEqual:
            if (!NEElementwiseComparison::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ComparisonOperation::NotEqual))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEElementwiseComparison>();
                acl_op->configure(&src1Tensor, &src2Tensor, &dstTensor, ComparisonOperation::NotEqual);
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseGreater:
            if (!NEElementwiseComparison::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ComparisonOperation::Greater))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEElementwiseComparison>();
                acl_op->configure(&src1Tensor, &src2Tensor, &dstTensor, ComparisonOperation::Greater);
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseGreaterEqual:
            if (!NEElementwiseComparison::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ComparisonOperation::GreaterEqual))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEElementwiseComparison>();
                acl_op->configure(&src1Tensor, &src2Tensor, &dstTensor, ComparisonOperation::GreaterEqual);
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseLess:
            if (!NEElementwiseComparison::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ComparisonOperation::Less))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEElementwiseComparison>();
                acl_op->configure(&src1Tensor, &src2Tensor, &dstTensor, ComparisonOperation::Less);
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseLessEqual:
            if (!NEElementwiseComparison::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo, ComparisonOperation::LessEqual))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEElementwiseComparison>();
                acl_op->configure(&src1Tensor, &src2Tensor, &dstTensor, ComparisonOperation::LessEqual);
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseRelu:
            if (aclEltwiseAttrs.alpha == 0) {
                if (!NEActivationLayer::validate(&src1TensorInfo, &dstTensorInfo,
                                                 ActivationLayerInfo::ActivationFunction::RELU))
                    return false;
            } else {
                if (!NEActivationLayer::validate(&src1TensorInfo, &dstTensorInfo,
                                                 {ActivationLayerInfo::ActivationFunction::LEAKY_RELU, aclEltwiseAttrs.alpha}))
                    return false;
            }
            exec_func = [this]{
                auto acl_op = std::make_unique<NEActivationLayer>();
                if (aclEltwiseAttrs.alpha == 0) {
                    acl_op->configure(&src1Tensor, &dstTensor, ActivationLayerInfo::ActivationFunction::RELU);
                } else {
                    acl_op->configure(&src1Tensor, &dstTensor,
                                      {ActivationLayerInfo::ActivationFunction::LEAKY_RELU, aclEltwiseAttrs.alpha});
                }
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseGeluErf:
            if (!NEActivationLayer::validate(&src1TensorInfo, &dstTensorInfo, ActivationLayerInfo::ActivationFunction::GELU))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEActivationLayer>();
                acl_op->configure(&src1Tensor, &dstTensor, ActivationLayerInfo::ActivationFunction::GELU);
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseElu:
            if (!NEActivationLayer::validate(&src1TensorInfo, &dstTensorInfo,
                                             {ActivationLayerInfo::ActivationFunction::ELU, aclEltwiseAttrs.alpha}))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEActivationLayer>();
                acl_op->configure(&src1Tensor, &dstTensor, {ActivationLayerInfo::ActivationFunction::ELU, aclEltwiseAttrs.alpha});
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseTanh:
            if (!NEActivationLayer::validate(&src1TensorInfo, &dstTensorInfo,
                                             {ActivationLayerInfo::ActivationFunction::TANH, 1.f, 1.f}))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEActivationLayer>();
                acl_op->configure(&src1Tensor, &dstTensor,
                                  {ActivationLayerInfo::ActivationFunction::TANH, 1.f, 1.f});
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseSigmoid:
            if (!NEActivationLayer::validate(&src1TensorInfo, &dstTensorInfo, ActivationLayerInfo::ActivationFunction::LOGISTIC))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEActivationLayer>();
                acl_op->configure(&src1Tensor, &dstTensor, ActivationLayerInfo::ActivationFunction::LOGISTIC);
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseAbs:
            if (!NEElementwiseUnaryLayer<ElementWiseUnary::ABS>::validate(&src1TensorInfo, &dstTensorInfo))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEElementwiseUnaryLayer<ElementWiseUnary::ABS>>();
                acl_op->configure(&src1Tensor, &dstTensor);
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseSqrt:
            if (!NEActivationLayer::validate(&src1TensorInfo, &dstTensorInfo, ActivationLayerInfo::ActivationFunction::SQRT))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEActivationLayer>();
                acl_op->configure(&src1Tensor, &dstTensor, ActivationLayerInfo::ActivationFunction::SQRT);
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseSoftRelu:
            if (!NEActivationLayer::validate(&src1TensorInfo, &dstTensorInfo, ActivationLayerInfo::ActivationFunction::SOFT_RELU))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEActivationLayer>();
                acl_op->configure(&src1Tensor, &dstTensor, ActivationLayerInfo::ActivationFunction::SOFT_RELU);
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseExp:
            if (!NEExpLayer::validate(&src1TensorInfo, &dstTensorInfo))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEExpLayer>();
                acl_op->configure(&src1Tensor, &dstTensor);
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseClamp:
            if (!NEActivationLayer::validate(&src1TensorInfo, &dstTensorInfo,
                                             {ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, aclEltwiseAttrs.beta, aclEltwiseAttrs.alpha}))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEActivationLayer>();
                acl_op->configure(&src1Tensor, &dstTensor,
                                  {ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, aclEltwiseAttrs.beta, aclEltwiseAttrs.alpha});
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseSwish:
            if (!NEActivationLayer::validate(&src1TensorInfo, &dstTensorInfo,
                                             {ActivationLayerInfo::ActivationFunction::SWISH, aclEltwiseAttrs.beta}))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEActivationLayer>();
                acl_op->configure(&src1Tensor, &dstTensor,
                                  {ActivationLayerInfo::ActivationFunction::SWISH, aclEltwiseAttrs.alpha});
                acl_op->run();
            };
            break;
        case Algorithm::EltwisePrelu:
            if (!NEPReluLayer::validate(&src1TensorInfo, &src2TensorInfo, &dstTensorInfo))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEPReluLayer>();
                acl_op->configure(&src1Tensor, &src2Tensor, &dstTensor);
                acl_op->run();
            };
            break;
        case Algorithm::EltwiseHswish:
            if (!NEActivationLayer::validate(&src1TensorInfo, &dstTensorInfo, ActivationLayerInfo::ActivationFunction::HARD_SWISH))
                return false;
            exec_func = [this]{
                auto acl_op = std::make_unique<NEActivationLayer>();
                acl_op->configure(&src1Tensor, &dstTensor, ActivationLayerInfo::ActivationFunction::HARD_SWISH);
                acl_op->run();
            };
            break;
        default:
            IE_THROW() << "Unsupported operation type for ACL Eltwise executor: " << static_cast<int>(aclEltwiseAttrs.algorithm);
    }
    return true;
}

void AclEltwiseExecutor::exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst,
                              const void *post_ops_data_) {
    src1Tensor.allocator()->import_memory(src[0]->GetPtr());
    if (!is_unary_op) {
        src2Tensor.allocator()->import_memory(src[1]->GetPtr());
    }
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());

    exec_func();

    src1Tensor.allocator()->free();
    if (!is_unary_op) {
        src2Tensor.allocator()->free();
    }
    dstTensor.allocator()->free();
}
}   // namespace intel_cpu
}   // namespace ov
