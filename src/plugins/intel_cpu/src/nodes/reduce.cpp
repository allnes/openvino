// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce.h"

#include "fake_quantize.h"
#include "eltwise.h"
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset4.hpp>

using namespace InferenceEngine;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl;

namespace ov {
namespace intel_cpu {
namespace node {

const std::map<const ngraph::DiscreteTypeInfo, std::function<void(const std::shared_ptr<ngraph::Node>&, Reduce&)>> Reduce::initializers = {
    {ngraph::opset4::ReduceL1::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceL1;
    }},
    {ngraph::opset4::ReduceL2::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceL2;
    }},
    {ngraph::opset1::ReduceLogicalAnd::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceAnd;
    }},
    {ngraph::opset1::ReduceLogicalOr::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceOr;
    }},
    {ngraph::opset1::ReduceMax::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceMax;
    }},
    {ngraph::opset1::ReduceMean::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceMean;
    }},
    {ngraph::opset1::ReduceMin::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceMin;
    }},
    {ngraph::opset1::ReduceProd::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceProd;
    }},
    {ngraph::opset1::ReduceSum::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceSum;
    }}
};

bool Reduce::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (std::dynamic_pointer_cast<const ngraph::op::util::ArithmeticReductionKeepDims>(op) == nullptr &&
                std::dynamic_pointer_cast<const ngraph::op::util::LogicalReductionKeepDims>(op) == nullptr) {
            errorMessage = "Reduce node with name " + op->get_friendly_name() + " is not derived from ArithmeticReductionKeepDims or LogicalReductionKeepDims";
            return false;
        }
        if (const auto reduce = std::dynamic_pointer_cast<const ngraph::op::util::ArithmeticReductionKeepDims>(op)) {
            auto reduceConst = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(reduce->get_input_node_shared_ptr(REDUCE_INDEXES));
            if (!reduceConst) {
                errorMessage = "Second tensor is not constant";
                return false;
            }
        }
        if (const auto reduce = std::dynamic_pointer_cast<const ngraph::op::util::LogicalReductionKeepDims>(op)) {
            auto reduceConst = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(reduce->get_input_node_shared_ptr(REDUCE_INDEXES));
            if (!reduceConst) {
                errorMessage = "Second tensor is not constant";
                return false;
            }
        }
        if (initializers.find(op->get_type_info()) == initializers.end()) {
            errorMessage = "Doesn't support Reduce algorithm: " +  std::string(op->get_type_info().name);
            return false;
        }
        if (std::dynamic_pointer_cast<ngraph::opset1::Constant>(op->get_input_node_shared_ptr(REDUCE_INDEXES)) == nullptr) {
            errorMessage = "Only const 'reduce_indexes' input is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Reduce::Reduce(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, NgraphShapeInferFactory(op, PortMask(REDUCE_INDEXES))) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "Reduce node with name '" + getName() + "'";
        initializers.at(op->get_type_info())(op, *this);
        if (const auto reduce = std::dynamic_pointer_cast<ngraph::op::util::ArithmeticReductionKeepDims>(op)) {
            reduceAttrs.keepDims = reduce->get_keep_dims();
            auto reduceConst = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(reduce->get_input_node_shared_ptr(REDUCE_INDEXES));
            if (!reduceConst)
                IE_THROW() << errorPrefix << " second tensor is not constant!";
            reduceAttrs.axes = reduceConst->cast_vector<int>();
        } else if (const auto reduce = std::dynamic_pointer_cast<ngraph::op::util::LogicalReductionKeepDims>(op)) {
            reduceAttrs.keepDims = reduce->get_keep_dims();
            auto reduceConst = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(reduce->get_input_node_shared_ptr(REDUCE_INDEXES));
            if (!reduceConst)
                IE_THROW() << errorPrefix << " second tensor is not constant!";
            reduceAttrs.axes = reduceConst->cast_vector<int>();
        }

        for (auto &axis : reduceAttrs.axes) {
            if (axis < 0)
                axis += static_cast<int>(getInputShapeAtPort(REDUCE_DATA).getRank());
        }

        reduceAttrs.operation = algorithm;
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void Reduce::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != 2)
        IE_THROW() << errorPrefix << " gets incorrect number of input edges!";
    if (getChildEdges().empty())
        IE_THROW() << errorPrefix << " gets incorrect number of output edges!";

    if (getInputShapeAtPort(REDUCE_INDEXES).getRank() != 1) {
        IE_THROW() << errorPrefix << " gets incorrect index vector dimension! Index vector should be 1 dimension.";
    }

    if (reduceAttrs.keepDims) {
        if (getInputShapeAtPort(REDUCE_DATA).getRank() != getOutputShapeAtPort(0).getRank())
            IE_THROW() << errorPrefix << " gets incorrect number of input/output dimensions!";
    } else {
        // In fact, after the Reduce operation, the shape must be a scalar if the previous one was 1d.
        // But for now, 0d tensor (scalar) is emulated as 1d tensor. Skip checking in such cases.
        bool is_emulated_0d_as_1d = getInputShapeAtPort(REDUCE_DATA).getRank() == 1 && getOutputShapeAtPort(0).getRank() == 1;
        if (getInputShapeAtPort(REDUCE_DATA).getRank() <= getOutputShapeAtPort(0).getRank() && !is_emulated_0d_as_1d)
            IE_THROW() << errorPrefix << "gets incorrect number of input/output dimensions!";
    }
}

void Reduce::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    input_prec = getOriginalInputPrecisionAtPort(REDUCE_DATA);
    output_prec = getOriginalOutputPrecisionAtPort(0);

    if (!fusedWith.empty()) {
        output_prec = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
    }

    if (mayiuse(sse41)) {
        // Since in jit mode we use the output memory as an intermediate accumulator for certain reduce modes, we can't use BF16 output precision due to
        // the possible accuracy loss. Therefore, for such mods, we will change the output precision to FP32.
        if (Precision::BF16 == output_prec) {
            if (!mayiuse(avx512_core)) {
                    output_prec = Precision::FP32;
            } else if (algorithm != Algorithm::ReduceAnd && algorithm != Algorithm::ReduceOr &&
                       algorithm != Algorithm::ReduceMin && algorithm != Algorithm::ReduceMax) {
                            output_prec = Precision::FP32;
            }
        }
    }

    NodeConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(2);
    config.outConfs.resize(1);
    config.inConfs[REDUCE_DATA].constant(false);
    config.inConfs[REDUCE_INDEXES].constant(false);
    config.outConfs[0].constant(false);
    config.inConfs[REDUCE_DATA].inPlace(-1);
    config.inConfs[REDUCE_INDEXES].inPlace(-1);
    config.outConfs[0].inPlace(-1);

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();

    auto pushDesc = [&](LayoutType inFormat, LayoutType outFormat, Precision inPrecision, Precision outPrecision) {
        config.inConfs[REDUCE_DATA].setMemDesc(creatorsMap.at(inFormat)->createSharedDesc(inPrecision, getInputShapeAtPort(REDUCE_DATA)));
        config.inConfs[REDUCE_INDEXES].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(InferenceEngine::Precision::I32,
                                                                                                 getInputShapeAtPort(REDUCE_INDEXES)));
        config.outConfs[0].setMemDesc(creatorsMap.at(outFormat)->createSharedDesc(outPrecision, getOutputShapeAtPort(0)));

        std::vector<MemoryDescPtr> srcMemoryDescs;
        for (int i = 0; i < config.inConfs.size(); i++) {
            srcMemoryDescs.push_back(config.inConfs[i].getMemDesc());
        }
        std::vector<MemoryDescPtr> dstMemoryDescs;
        for (int i = 0; i < config.outConfs.size(); i++) {
            dstMemoryDescs.push_back(config.outConfs[i].getMemDesc());
        }

        auto factory = std::make_shared<ReduceExecutorFactory>(reduceAttrs, srcMemoryDescs, dstMemoryDescs,
                                                               std::make_shared<ExecutorContext>(context, getPrimitivesPriority()));
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::undef, factory});
    };

    if (mayiuse(sse41)) {
        pushDesc(LayoutType::ncsp, LayoutType::ncsp, input_prec, output_prec);
        if ((getInputShapeAtPort(REDUCE_DATA).getRank() == 4 || getInputShapeAtPort(REDUCE_DATA).getRank() == 5) &&
                getInputShapeAtPort(REDUCE_DATA).getMinDims()[1] > 1) {
            if (reduceAttrs.keepDims) {
                if (mayiuse(cpu::x64::avx512_core)) {
                    pushDesc(LayoutType::nspc, LayoutType::nspc, input_prec, output_prec);
                    pushDesc(LayoutType::nCsp16c, LayoutType::nCsp16c, input_prec, output_prec);
                } else if (mayiuse(cpu::x64::avx2) || mayiuse(cpu::x64::sse41)) {
                    pushDesc(LayoutType::nspc, LayoutType::nspc, input_prec, output_prec);
                    pushDesc(LayoutType::nCsp8c, LayoutType::nCsp8c, input_prec, output_prec);
                }
            } else {
                if (mayiuse(cpu::x64::avx512_core)) {
                    pushDesc(LayoutType::nspc, LayoutType::ncsp, input_prec, output_prec);
                    pushDesc(LayoutType::nCsp16c, LayoutType::ncsp, input_prec, output_prec);
                } else if (mayiuse(cpu::x64::avx2) || mayiuse(cpu::x64::sse41)) {
                    pushDesc(LayoutType::nspc, LayoutType::ncsp, input_prec, output_prec);
                    pushDesc(LayoutType::nCsp8c, LayoutType::ncsp, input_prec, output_prec);
                }
            }
        }
    }

    pushDesc(LayoutType::ncsp, LayoutType::ncsp, InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP32);
}

bool Reduce::isExecutable() const {
    return !isInputTensorAtPortEmpty(REDUCE_DATA);
}

void Reduce::prepareParams() {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    const SizeVector &dst_dims = dstMemPtr->getDesc().getShape().getDims();

    std::vector<MemoryDescPtr> srcMemoryDescs;
    for (int i = 0; i < getOriginalInputsNumber(); i++) {
        srcMemoryDescs.push_back(getParentEdgeAt(i)->getMemoryPtr()->getDescPtr());
    }
    std::vector<MemoryDescPtr> dstMemoryDescs;
    for (int i = 0; i < getOriginalOutputsNumber(); i++) {
        dstMemoryDescs.push_back(getChildEdgeAt(i)->getMemoryPtr()->getDescPtr());
    }
    dnnl::primitive_attr attr;
    setPostOps(attr, dst_dims, true);
    auto selectedPD = getSelectedPrimitiveDescriptor();

    execPtr = selectedPD->getExecutorFactoryAs<ReduceExecutorFactory>()->makeExecutor(reduceAttrs, srcMemoryDescs, dstMemoryDescs, attr);
    selectedPD->setImplementationType(execPtr->getImplType());
}

void Reduce::createPrimitive() {
    if (!isExecutable()) {
        return;
    }

    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

void Reduce::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void Reduce::execute(dnnl::stream strm) {
    if (!execPtr) {
        IE_THROW() << "Can't execute Reduce node. Executor is not created";
    }

    std::vector<MemoryCPtr> srcMemory;
    for (int i = 0; i < getOriginalInputsNumber(); i++) {
        srcMemory.push_back(getParentEdgeAt(i)->getMemoryPtr());
    }
    std::vector<MemoryPtr> dstMemory;
    for (int i = 0; i < getOriginalOutputsNumber(); i++) {
        dstMemory.push_back(getChildEdgeAt(i)->getMemoryPtr());
    }

    execPtr->exec(srcMemory, dstMemory, postOpsDataPtrs.data());
}

void Reduce::setPostOps(dnnl::primitive_attr &attr, const VectorDims &postOpDims, bool initWeights) {
    dnnl::post_ops ops;
    postOpsDataPtrs.clear();
    for (auto &node : fusedWith) {
        auto* fakeQuantizeNode = dynamic_cast<FakeQuantize *>(node.get());
        if (fakeQuantizeNode) {
            fakeQuantizeNode->appendPostOps(ops, {}, postOpsDataPtrs);
            continue;
        }

        auto* eltwiseNode = dynamic_cast<Eltwise *>(node.get());
        if (eltwiseNode) {
            eltwiseNode->appendPostOps(ops, postOpDims, postOpsDataPtrs, getFusingAxis());
            continue;
        }
        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

int Reduce::getFusingAxis() const {
    int channelAxis = 1;
    if (!reduceAttrs.keepDims) {
        for (auto &raw_axis : reduceAttrs.axes) {
            int axis = raw_axis >= 0 ? raw_axis : raw_axis + static_cast<int>(getInputShapeAtPort(REDUCE_DATA).getRank());
            if (axis == 1) {
                // channel axis has been reduced and doesn't exist any more
                channelAxis = -1;
                break;
            } else if (axis == 0) {
                channelAxis = 0;
            }
        }
    }
    return channelAxis;
}

bool Reduce::canFuse(const NodePtr& node) const {
    Precision input_prec = getOriginalInputPrecisionAtPort(REDUCE_DATA);
    Precision output_prec = getOriginalOutputPrecisionAtPort(0);

    static const Precision supportedPrecisions[] = {
        Precision::FP32,
        Precision::BF16,
        Precision::I32,
        Precision::I8,
        Precision::U8
    };

    if (!((mayiuse(cpu::x64::sse41)) && (getInputShapeAtPort(REDUCE_DATA).getRank() <= 5) &&
           std::find(std::begin(supportedPrecisions), std::end(supportedPrecisions), input_prec) != std::end(supportedPrecisions) &&
           std::find(std::begin(supportedPrecisions), std::end(supportedPrecisions), output_prec) != std::end(supportedPrecisions))) {
        return false;
    }

    auto jit_beyond_5D = false;
    if (getInputShapeAtPort(REDUCE_DATA).getRank() > 5) {
        if (reduceAttrs.axes.size() <= 1) {
            jit_beyond_5D = true;
        } else {
            for (size_t i = 1; i < reduceAttrs.axes.size(); i++) {
                if (reduceAttrs.axes[i] != reduceAttrs.axes[i - 1] + 1) {
                    jit_beyond_5D = false;
                    break;
                }
                jit_beyond_5D = true;
            }
        }
    }

    if (jit_beyond_5D || algorithm == Algorithm::ReduceAnd || algorithm == Algorithm::ReduceOr) {
        return false;
    }

    // In jit mode we use the output memory as an intermediate accumulator for certain reduce modes.
    // If the post ops node has a lower precision for such modes, post ops fusing won't be supposted, in order to avoid accuracy loss.
    if (output_prec == Precision::FP32 &&
        !node->getOriginalOutputPrecisions().empty() && node->getOriginalOutputPrecisionAtPort(0) != Precision::FP32) {
        if (algorithm != Algorithm::ReduceAnd && algorithm != Algorithm::ReduceOr &&
            algorithm != Algorithm::ReduceMin && algorithm != Algorithm::ReduceMax) {
            return false;
        }
    }

    return canFuseSimpleOperation(node);
}

bool Reduce::created() const {
    return getType() == Type::Reduce;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
