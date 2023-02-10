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
    std::unique_ptr<arm_compute::NEArithmeticAddition> acl_add = nullptr;
    std::unique_ptr<arm_compute::NEArithmeticSubtraction> acl_sub = nullptr;
    std::unique_ptr<arm_compute::NEPixelWiseMultiplication> acl_mul = nullptr;
    std::unique_ptr<arm_compute::NEElementwiseDivision> acl_div = nullptr;
    std::unique_ptr<arm_compute::NEElementwiseSquaredDiff> acl_sqdiff = nullptr;
    std::unique_ptr<arm_compute::NEElementwiseMax> acl_max = nullptr;
    std::unique_ptr<arm_compute::NEElementwiseMin> acl_min = nullptr;
    std::unique_ptr<arm_compute::NEElementwisePower> acl_pow = nullptr;
    std::unique_ptr<arm_compute::NEElementwiseComparison> acl_comp = nullptr;
    std::unique_ptr<arm_compute::NEActivationLayer> acl_act = nullptr;
    std::function<void()> run_func;
};

class AclEltwiseExecutorBuilder : public EltwiseExecutorBuilder {
public:
    bool isSupported(const EltwiseAttrs& eltwiseAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        for (const auto &desc : srcDescs) {
            if (!one_of(desc->getPrecision(),
                        InferenceEngine::Precision::FP32))
                return false;
        }

        for (const auto &desc : dstDescs) {
            if (!one_of(desc->getPrecision(),
                        InferenceEngine::Precision::FP32))
                return false;
        }

        switch (eltwiseAttrs.algorithm) {
            case Algorithm::EltwiseMulAdd:
            case Algorithm::EltwiseFloorMod:
            case Algorithm::EltwisePowerStatic:
            case Algorithm::EltwiseMod:
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