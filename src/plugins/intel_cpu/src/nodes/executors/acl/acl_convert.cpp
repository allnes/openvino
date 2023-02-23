// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_convert.hpp"
#include "acl_utils.hpp"

using namespace arm_compute;

TensorShape convertShapeCast(const ov::intel_cpu::VectorDims& dims) {
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

bool ov::intel_cpu::ACLConvertExecutor::init(const ConvertAttrs& convertAttrs,
                                             const std::vector<MemoryDescPtr>& srcDescs,
                                             const std::vector<MemoryDescPtr>& dstDescs,
                                             const dnnl::primitive_attr &attr) {
    aclConvertAttrs = convertAttrs;
    auto srcDims = srcDescs[0]->getShape().getDims();
    auto dstDims = dstDescs[0]->getShape().getDims();

    TensorInfo srcTensorInfo = TensorInfo(convertShapeCast(srcDims), 1,
                                          precisionToAclDataType(aclConvertAttrs.srcPrc),
                                          getAclDataLayoutByMemoryDesc(srcDescs[0]));
    TensorInfo dstTensorInfo = TensorInfo(convertShapeCast(dstDims), 1,
                                          precisionToAclDataType(aclConvertAttrs.dstPrc),
                                          getAclDataLayoutByMemoryDesc(dstDescs[0]));
    if (aclConvertAttrs.srcPrc == aclConvertAttrs.dstPrc) {
        if (!arm_compute::NECopy::validate(&srcTensorInfo, &dstTensorInfo)) {
            return false;
        }
    } else {
        if (!arm_compute::NECast::validate(&srcTensorInfo, &dstTensorInfo, arm_compute::ConvertPolicy::SATURATE)) {
            return false;
        }
    }

    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    if (aclConvertAttrs.srcPrc == aclConvertAttrs.dstPrc) {
        acl_copy = std::shared_ptr<arm_compute::NECopy>();
        acl_copy->configure(&srcTensor, &dstTensor);
    } else {
        acl_cast = std::shared_ptr<arm_compute::NECast>();
        acl_cast->configure(&srcTensor, &dstTensor, arm_compute::ConvertPolicy::SATURATE);
    }

    return true;
}

void ov::intel_cpu::ACLConvertExecutor::exec(const std::vector<MemoryCPtr>& src,
                                             const std::vector<MemoryPtr>& dst,
                                             std::unordered_map<int, MemoryPtr> postOpsArgs) {
    srcTensor.allocator()->import_memory(src[0]->GetPtr());
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());

    if (aclConvertAttrs.srcPrc == aclConvertAttrs.dstPrc) {
        acl_copy->run();
    } else {
        acl_cast->run();
    }

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}
