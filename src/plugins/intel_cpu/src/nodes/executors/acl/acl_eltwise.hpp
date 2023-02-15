// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../eltwise.hpp"
#include "utils/acl_utils.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace ov {
namespace intel_cpu {

arm_compute::TensorShape eltwiseShapeCast(const VectorDims& dims);

class AclEltwiseExecutor : public EltwiseExecutor {
public:
    AclEltwiseExecutor();

    bool init(const EltwiseAttrs& eltwiseAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const std::vector<EltwisePostOp>& postOps) override;

    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void *post_ops_data_) override;

    impl_desc_type getImplType() const override {
        return implType;
    }
private:
    EltwiseAttrs aclEltwiseAttrs{};
    impl_desc_type implType = impl_desc_type::acl;
    arm_compute::Tensor src1Tensor;
    arm_compute::Tensor src2Tensor;
    arm_compute::Tensor dstTensor;
    std::function<void()> exec_func;
    bool is_unary_op = false;
};

class AclEltwiseExecutorBuilder : public EltwiseExecutorBuilder {
public:
    bool isSupported(const EltwiseAttrs& eltwiseAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        for (const auto &desc : srcDescs) {
            if (desc->hasLayoutType(LayoutType::nCsp8c)) {
                return false;
            }
            if (!one_of(desc->getPrecision(),
                        InferenceEngine::Precision::FP16,
                        InferenceEngine::Precision::FP32))
                return false;
        }

        for (const auto &desc : dstDescs) {
            if (desc->hasLayoutType(LayoutType::nCsp8c)) {
                return false;
            }
            if (!one_of(desc->getPrecision(),
                        InferenceEngine::Precision::FP16,
                        InferenceEngine::Precision::FP32))
                return false;
        }

        switch (eltwiseAttrs.algorithm) {
            case Algorithm::EltwiseIsFinite:
            case Algorithm::EltwiseIsInf:
            case Algorithm::EltwiseIsNaN:
            case Algorithm::EltwiseFloorMod:
            case Algorithm::EltwiseMod:
            case Algorithm::EltwisePowerStatic:
            case Algorithm::EltwiseMulAdd:
            case Algorithm::EltwiseLogicalAnd:
            case Algorithm::EltwiseLogicalOr:
            case Algorithm::EltwiseLogicalXor:
            case Algorithm::EltwiseLogicalNot:
            case Algorithm::EltwiseGeluTanh:
            case Algorithm::EltwisePrelu: // TODO: accuracy problem
            case Algorithm::EltwiseMish:
            case Algorithm::EltwiseHsigmoid:
            case Algorithm::EltwiseRoundHalfToEven:
            case Algorithm::EltwiseRoundHalfAwayFromZero:
            case Algorithm::EltwiseErf:
            case Algorithm::EltwiseSoftSign:
                return false;
            default:
                return true;
        }
    }

    EltwiseExecutorPtr makeExecutor() const override {
        return std::make_shared<AclEltwiseExecutor>();
    }
};

}   // namespace intel_cpu
}   // namespace ov