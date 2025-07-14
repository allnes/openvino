// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_mvn.hpp"

namespace ov::intel_cpu {

using namespace arm_compute;

AclMVNExecutor::AclMVNExecutor(const MVNAttrs& mvnAttrs,
                               const MemoryArgs& memory,
                               const ExecutorContext::CPtr& context)
    : MVNExecutor(mvnAttrs, memory, context) {
    // Initialize the ACL implementation
    auto srcDesc = memory.at(ARG_SRC_0)->getDescPtr();
    auto dstDesc = memory.at(ARG_DST)->getDescPtr();
    std::vector<MemoryDescPtr> srcDescs = {srcDesc};
    std::vector<MemoryDescPtr> dstDescs = {dstDesc};
    
    dnnl::primitive_attr attr;
    init(mvnAttrs, srcDescs, dstDescs, attr);
}

bool AclMVNExecutor::init(const MVNAttrs& mvnAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          [[maybe_unused]] const dnnl::primitive_attr& attr) {
    auto srcDims = srcDescs[0]->getShape().getStaticDims();
    auto dstDims = dstDescs[0]->getShape().getStaticDims();

    size_t X = 0, Y = 0;
    if (mvnAttrs.initAcrossChannels_) {
        if (srcDims.size() >= 2u) {
            Y = srcDims[0];
            X = srcDims[1];
            for (size_t i = 2; i < srcDims.size(); i++) {
                X *= srcDims[i];
            }
        } else {
            Y = 1;
            X = srcDims[0];
        }
    } else {
        if (srcDims.size() > 2u) {
            Y = srcDims[0] * srcDims[1];
            X = srcDims[2];
            for (size_t i = 3; i < srcDims.size(); i++) {
                X *= srcDims[i];
            }
        } else if (srcDims.size() == 2u) {
            Y = srcDims[0] * srcDims[1];
            X = 1;
        } else {
            Y = srcDims[0];
            X = 1;
        }
    }

    TensorInfo srcTensorInfo = TensorInfo(TensorShape(X, Y),
                                          1,
                                          precisionToAclDataType(srcDescs[0]->getPrecision()),
                                          getAclDataLayoutByMemoryDesc(srcDescs[0]));
    TensorInfo dstTensorInfo = TensorInfo(TensorShape(X, Y),
                                          1,
                                          precisionToAclDataType(dstDescs[0]->getPrecision()),
                                          getAclDataLayoutByMemoryDesc(dstDescs[0]));

    if (!arm_compute::NEMeanStdDevNormalizationLayer::validate(&srcTensorInfo, &dstTensorInfo, mvnAttrs.epsValue_)) {
        return false;
    }

    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    mvn = std::make_unique<arm_compute::NEMeanStdDevNormalizationLayer>();
    configureThreadSafe([&] {
        mvn->configure(&srcTensor, &dstTensor, mvnAttrs.epsValue_);
    });

    return true;
}

void AclMVNExecutor::executeImpl(const MemoryArgs& memory) {
    const auto src = memory.at(ARG_SRC_0);
    const auto dst = memory.at(ARG_DST);
    
    srcTensor.allocator()->import_memory(src->getData());
    dstTensor.allocator()->import_memory(dst->getData());

    mvn->run();

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}

// static method for ACL support check
bool AclMVNExecutor::supports(const MVNAttrs& mvnAttrs,
                              const std::vector<MemoryDescPtr>& srcDescs,
                              const std::vector<MemoryDescPtr>& dstDescs) {
    if ((srcDescs[0]->getPrecision() != ov::element::f32 && srcDescs[0]->getPrecision() != ov::element::f16) ||
        srcDescs[0]->getPrecision() != dstDescs[0]->getPrecision()) {
        DEBUG_LOG("NEMeanStdDevNormalizationLayer does not support precisions:",
                  " src[0]=",
                  srcDescs[0]->getPrecision(),
                  " dst[0]=",
                  dstDescs[0]->getPrecision());
        return false;
    }

    if (!(srcDescs[0]->hasLayoutType(LayoutType::ncsp) && dstDescs[0]->hasLayoutType(LayoutType::ncsp)) &&
        !(srcDescs[0]->hasLayoutType(LayoutType::nspc) && dstDescs[0]->hasLayoutType(LayoutType::nspc))) {
        DEBUG_LOG("NEMeanStdDevNormalizationLayer does not support layout:",
                  " src: ",
                  srcDescs[0]->serializeFormat(),
                  " dst: ",
                  dstDescs[0]->serializeFormat());
        return false;
    }

    if (mvnAttrs.epsMode_ == MVNEpsMode::OUTSIDE_SQRT) {
        DEBUG_LOG("NEMeanStdDevNormalizationLayer does not support OUTSIDE_SQRT mode");
        return false;
    }
    if (!mvnAttrs.normalizeVariance_) {
        DEBUG_LOG("NEMeanStdDevNormalizationLayer supports normalize_variance=true only");
        return false;
    }
    if (!mvnAttrs.initAcrossChannels_ && srcDescs[0]->hasLayoutType(LayoutType::nspc)) {
        DEBUG_LOG("initAcrossChannels = false is not supported by ACL for NHWC layout");
        return false;
    }

    return true;
}

}  // namespace ov::intel_cpu
