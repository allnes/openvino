// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax.h"

#include <string>
#include <dnnl_types.h>
#include <dnnl_extension_utils.h>
#include <memory_desc/cpu_memory_desc_utils.h>
#include <ngraph/opsets/opset1.hpp>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include <common/primitive_hashing_utils.hpp>
#include <utils/shape_inference/shape_inference_pass_through.hpp>

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool SoftMax::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!std::dynamic_pointer_cast<const ngraph::opset1::Softmax>(op)) {
            errorMessage = "Only opset1 Softmax operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

SoftMax::SoftMax(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context) :
        Node(op, context, PassThroughShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
    softMaxAttrs.axis = ngraph::as_type_ptr<ngraph::op::v1::Softmax>(op)->get_axis();
}

void SoftMax::getSupportedDescriptors() {
    if (descs.size())
        return;

    InferenceEngine::Precision precision = getOriginalInputPrecisionAtPort(0);
    if (precision != InferenceEngine::Precision::FP32 && precision != InferenceEngine::Precision::BF16)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = DnnlExtensionUtils::IEPrecisionToDataType(precision);

    if (getParentEdges().size() != 1)
        IE_THROW() << "Incorrect number of input edges for layer " << getName();
    if (!getChildEdges().size())
        IE_THROW() << "Incorrect number of output edges for layer " << getName();

    const auto &inShape = getInputShapeAtPort(0);
    if (inShape.getRank() == 3) {
        auto in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inShape, inputDataType, memory::format_tag::abc);
        createDescriptor({in_candidate}, {});
    }

    for (auto format : getAvailableFormatsForDims(inShape)) {
        auto in_candidate = std::make_shared<DnnlBlockedMemoryDesc>(inShape, inputDataType, format);

        if (in_candidate->blocksExtended())
            continue;

        createDescriptor({in_candidate}, {});
    }
}

bool SoftMax::created() const {
    return getType() == Type::Softmax;
}

void SoftMax::initOptimalPrimitiveDescriptor() {
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set.";
    auto config = selected_pd->getConfig();
    if (isDynamicNode()) {
        auto outMemDesc = config.outConfs[0].getMemDesc();
        config.outConfs[0].setMemDesc(std::dynamic_pointer_cast<BlockedMemoryDesc>(outMemDesc), BLOCKED_DESC_FULL_MASK);
    } else {
        if (config.inConfs.size() != 1 || config.outConfs.size() != 1 ||
            (config.inConfs[0].getMemDesc()->isDefined() &&
             config.outConfs[0].getMemDesc()->isDefined() && !config.outConfs[0].getPortDesc()->isCompatible(*config.inConfs[0].getPortDesc())))
            IE_THROW() << "Layer " << getName() << " has incorrect selected config!";

        config.inConfs[0].setMemDesc(getConsistentInputDesc(config, 0)->getMemDesc());
        config.outConfs[0].setMemDesc(config.inConfs[0].getMemDesc());
    }
    initDescriptor(config);

    std::vector<MemoryDescPtr> srcMemoryDescs;
    for (int i = 0; i < config.inConfs.size(); i++) {
        srcMemoryDescs.push_back(config.inConfs[i].getMemDesc());
    }
    std::vector<MemoryDescPtr> dstMemoryDescs;
    for (int i = 0; i < config.outConfs.size(); i++) {
        dstMemoryDescs.push_back(config.outConfs[i].getMemDesc());
    }

    auto attr = initPrimitiveAttr();
    auto factory = std::make_shared<SoftMaxExecutorFactory>(softMaxAttrs, srcMemoryDescs, dstMemoryDescs,
                                                           std::make_shared<ExecutorContext>(context, getPrimitivesPriority()));

    supportedPrimitiveDescriptors.push_back({config, impl_desc_type::undef, factory});
}

void SoftMax::createDescriptor(const std::vector<MemoryDescPtr> &inputDesc,
                                         const std::vector<MemoryDescPtr> &outputDesc) {
    auto inpDesc = inputDesc[0]->isDefined() ? inputDesc[0] : MemoryDescUtils::makeDummyDesc(*inputDesc[0]);
    DnnlMemoryDescPtr definedInpMemDesc = MemoryDescUtils::convertToDnnlMemoryDesc(inpDesc);
    auto in_candidate = definedInpMemDesc->getDnnlDesc();

    DnnlDesriptor desc(std::shared_ptr<softmax_forward::desc>(
            new softmax_forward::desc(prop_kind::forward_scoring, in_candidate, softMaxAttrs.axis)));
    descs.push_back(desc);
}

void SoftMax::prepareParams() {
    auto inpDesc = getParentEdgeAt(0)->getMemory().GetDescWithType<DnnlMemoryDesc>();
    const NodeDesc* selected_pd = getSelectedPrimitiveDescriptor();

    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set for node " << getName() << ".";
    softMaxAttrs.implDescType = selected_pd->getImplementationType();
    softMaxAttrs.inp0 = inpDesc;

    SoftmaxKey key = {softMaxAttrs.inp0, softMaxAttrs.implDescType, softMaxAttrs.axis};
    auto engine = getEngine();
    auto builder = [&engine, this](const SoftmaxKey& key) -> std::shared_ptr<SoftMaxExecutor> {
//        softmax_forward::primitive_desc prim_desc;
//        DnnlDesriptor desc(std::shared_ptr<softmax_forward::desc>(
//            new softmax_forward::desc(prop_kind::forward_scoring, key.inp0->getDnnlDesc(), key.axis)));
//        dnnl::primitive_attr attr;
//        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
//        primitive_desc_iterator itpd = desc.createPrimitiveDescriptorIterator(engine, attr);
//
//        while (itpd) {
//            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());
//            if (impl_type == key.implType ||
//                // At least for oneDNN v2.4 the softmax primitive is optimized for the cases where the dimension of the
//                // softmax axis is physically dense. There could be situations where it is not possible to detect the
//                // optimized case in advance in case of dynamic shapes, but in runtime the shape could be suitable for
//                // the optimized implementation, so we have to select the optimized one.
//                (ref_any == key.implType && (impl_type & jit))) {
//                prim_desc = itpd.get();
//                break;
//            }
//            if (!itpd.next_impl())
//                return nullptr;
//        }
//        return std::make_shared<softmax_forward>(prim_desc);
        std::vector<MemoryDescPtr> srcMemoryDescs;
        for (int i = 0; i < getOriginalInputsNumber(); i++) {
            srcMemoryDescs.push_back(getParentEdgeAt(i)->getMemoryPtr()->getDescPtr());
        }
        std::vector<MemoryDescPtr> dstMemoryDescs;
        for (int i = 0; i < getOriginalOutputsNumber(); i++) {
            dstMemoryDescs.push_back(getChildEdgeAt(i)->getMemoryPtr()->getDescPtr());
        }

        auto attr = std::make_shared<dnnl::primitive_attr>(dnnl::primitive_attr());
        auto selectedPD = getSelectedPrimitiveDescriptor();
        auto tmpExecPtr = selectedPD->getExecutorFactoryAs<SoftMaxExecutorFactory>()->makeExecutor(softMaxAttrs, srcMemoryDescs, dstMemoryDescs, *attr.get());
        selectedPD->setImplementationType(tmpExecPtr->getImplType());
        return tmpExecPtr;
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);

    if (!result.first) {
        IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
    }

    execPtr = result.first;

//    softMaxPrim = result.first;
//
//    auto pd = (*softMaxPrim).get_primitive_desc();
//    auto scratchpadMem = getScratchPadMem(pd);
//
//    auto src = getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
//    auto dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
//    softMaxPrimArgs = {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}, {DNNL_ARG_SCRATCHPAD, scratchpadMem->GetPrimitive()}};
}

void SoftMax::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void SoftMax::execute(dnnl::stream strm) {
    if (!execPtr) {
        IE_THROW() << "Can't execute MatMul node. Executor is not created";
    }

    std::vector<MemoryCPtr> srcMemory;
    for (int i = 0; i < getOriginalInputsNumber(); i++) {
        srcMemory.push_back(getParentEdgeAt(i)->getMemoryPtr());
    }
    std::vector<MemoryPtr> dstMemory;
    for (int i = 0; i < getOriginalOutputsNumber(); i++) {
        dstMemory.push_back(getChildEdgeAt(i)->getMemoryPtr());
    }
    std::vector<const void*> postOpsDataPtrs;
    execPtr->exec(srcMemory, dstMemory, postOpsDataPtrs.data());
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
