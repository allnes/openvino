// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../eltwise.hpp"
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
    EltwiseAttrs eltwiseAttrs{};
    impl_desc_type implType = impl_desc_type::acl;
    arm_compute::Tensor src1Tensor;
    arm_compute::Tensor src2Tensor;
    arm_compute::Tensor dstTensor;
    std::unique_ptr<arm_compute::IFunction> acl_op = nullptr;
};

class AclEltwiseExecutorBuilder : public EltwiseExecutorBuilder {
public:
    bool isSupported(const EltwiseAttrs& eltwiseAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        if (srcDescs[0]->getPrecision() != InferenceEngine::Precision::FP32 ||
            srcDescs[1]->getPrecision() != InferenceEngine::Precision::FP32 ||
            dstDescs[0]->getPrecision() != InferenceEngine::Precision::FP32)
            return false;

        if (!srcDescs[0]->hasLayoutType(LayoutType::ncsp) ||
            !srcDescs[1]->hasLayoutType(LayoutType::ncsp) ||
            !dstDescs[0]->hasLayoutType(LayoutType::ncsp))
            return false;

        return true;
    }

    EltwiseExecutorPtr makeExecutor() const override {
        return std::make_shared<AclEltwiseExecutor>();
    }
};

}   // namespace intel_cpu
}   // namespace ov