// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../eltwise.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace ov {
namespace intel_cpu {

using namespace InferenceEngine;

class AclEltwiseExecutor : public EltwiseExecutor {
public:
    AclEltwiseExecutor(const ExecutorContext::CPtr context);

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
    std::vector<arm_compute::Tensor> srcTensors, dstTensors;
    std::function<void()> exec_func;
};

class AclEltwiseExecutorBuilder : public EltwiseExecutorBuilder {
public:
    bool isSupported(const EltwiseAttrs& eltwiseAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        for (const auto &srcD : srcDescs) {
            for (const auto &dstD : dstDescs) {
                if ((srcD->getPrecision() != InferenceEngine::Precision::FP32 &&
                     srcD->getPrecision() != InferenceEngine::Precision::FP16) ||
                     srcD->getPrecision() != dstD->getPrecision())
                    return false;
            }
        }
        auto checker = [](const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          Precision ref1, Precision ref2, Precision ref3) -> bool {
            if (srcDescs[0]->getPrecision() == ref1 &&
                srcDescs[1]->getPrecision() == ref2 &&
                dstDescs[0]->getPrecision() == ref3) {
                return true;
            }
            return false;
        };
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
            case Algorithm::EltwiseMish:
            case Algorithm::EltwiseHsigmoid:
            case Algorithm::EltwiseRoundHalfToEven:
            case Algorithm::EltwiseRoundHalfAwayFromZero:
            case Algorithm::EltwiseErf:
            case Algorithm::EltwiseSoftSign:
                return false;
            case Algorithm::EltwiseAdd:
                if (!(one_of(srcDescs[0]->getPrecision(),
                            Precision::U8,
                            Precision::I16,
                            Precision::I32,
                            Precision::FP16,
                            Precision::FP32) &&
                    srcDescs[0]->getPrecision() == srcDescs[1]->getPrecision() &&
                    srcDescs[1]->getPrecision() == dstDescs[0]->getPrecision())) { return false; }
                break;
            case Algorithm::EltwiseMultiply:
                if (!(checker(srcDescs, dstDescs, Precision::U8, Precision::U8, Precision::U8) ||
                      checker(srcDescs, dstDescs, Precision::U8, Precision::U8, Precision::I16) ||
                      checker(srcDescs, dstDescs, Precision::U8, Precision::I16, Precision::I16) ||
                      checker(srcDescs, dstDescs, Precision::I16, Precision::U8, Precision::I16) ||
                      checker(srcDescs, dstDescs, Precision::I16, Precision::I16, Precision::I16) ||
                      checker(srcDescs, dstDescs, Precision::FP16, Precision::FP16, Precision::FP16) ||
                      checker(srcDescs, dstDescs, Precision::FP32, Precision::FP32, Precision::FP32))) {
                    return false;
                }
                break;
            default:
                return true;
        }
    }

    EltwiseExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<AclEltwiseExecutor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov