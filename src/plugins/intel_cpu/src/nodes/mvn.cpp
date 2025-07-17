// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn.h"

#include <algorithm>
#include <cassert>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <utility>
#include <vector>

#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "eltwise.h"
#include "fake_quantize.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "node.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/mvn_config.hpp"
#include "nodes/node_config.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/mvn.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"
#include "utils/precision_support.h"

namespace ov::intel_cpu::node {

bool MVN::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_output_partial_shape(0).rank().is_dynamic()) {
            errorMessage = "Unsupported dynamic input rank.";
            return false;
        }
        const auto& inDataRank = op->get_output_partial_shape(0).rank().get_length();
        if (inDataRank < 1 || inDataRank > 5) {
            errorMessage = "First input accepts ranks from 1 to 5. Actual: " + std::to_string(inDataRank);
            return false;
        }

        if (auto mvnOp = ov::as_type_ptr<const ov::op::v6::MVN>(op)) {
            auto axesOp = ov::as_type_ptr<ov::op::v0::Constant>(mvnOp->get_input_node_shared_ptr(1));
            if (!axesOp) {
                errorMessage = "Constant expected as the second input.";
                return false;
            }

            auto epsMode = mvnOp->get_eps_mode();
            if (epsMode != ov::op::MVNEpsMode::INSIDE_SQRT && epsMode != ov::op::MVNEpsMode::OUTSIDE_SQRT) {
                errorMessage = std::string("Just INSIDE_SQRT and OUTSIDE_SQRT epsilon mods are supported. Actual: ") +
                               std::to_string(static_cast<int>(epsMode));
                return false;
            }
            // Validates MVN node axes to check whether it can be executed on the current CPU implementation.
            // Supported cases:
            // 1D: axes: [0]
            // 2D: axes: [1]
            // 3D: axes: [1,2], [2]
            // 4D: axes: [1,2,3], [2,3]
            // 5D: axes: [1,2,3,4], [2,3,4]
            auto axesVal = axesOp->cast_vector<int>();
            for (int& axe : axesVal) {
                axe = axe < 0 ? axe + inDataRank : axe;
            }
            std::sort(axesVal.begin(), axesVal.end());
            if (inDataRank == 1) {
                if (axesVal.size() != 1 || axesVal[0] != 0) {
                    errorMessage = "Unsupported axes.";
                    return false;
                }
            } else {
                if (inDataRank > 5 || (static_cast<size_t>(inDataRank) != axesVal.size() + 1 &&
                                       static_cast<size_t>(inDataRank) != axesVal.size() + 2)) {
                    errorMessage = "Unsupported axes.";
                    return false;
                }
                int value = inDataRank - 1;
                for (int i = axesVal.size() - 1; i >= 0; i--, value--) {
                    if (axesVal[i] != value) {
                        errorMessage = "Unsupported axes.";
                        return false;
                    }
                }
            }
        } else if (auto mvnOp = ov::as_type_ptr<const ov::op::v0::MVN>(op)) {
        } else {
            errorMessage = "Node is not an instance of the MVN operation.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MVN::MVN(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    mvnAttrs.epsMode_ = INSIDE_SQRT;
    if (auto mvnOp = ov::as_type_ptr<ov::op::v6::MVN>(op)) {
        mvnAttrs.normalizeVariance_ = mvnOp->get_normalize_variance();
        mvnAttrs.epsValue_ = mvnOp->get_eps();
        if (mvnOp->get_eps_mode() == ov::op::MVNEpsMode::OUTSIDE_SQRT) {
            mvnAttrs.epsMode_ = OUTSIDE_SQRT;
        }

        mvnAttrs.initAcrossChannels_ = false;
        const auto& inDataShapeSize = getInputShapeAtPort(0).getRank();
        if (inDataShapeSize == mvnOp->input_value(1).get_shape()[0] + 1 || inDataShapeSize == 1) {
            mvnAttrs.initAcrossChannels_ = true;
        }
    } else if (auto mvnOp = ov::as_type_ptr<ov::op::v0::MVN>(op)) {
        mvnAttrs.normalizeVariance_ = mvnOp->get_normalize_variance();
        mvnAttrs.epsValue_ = mvnOp->get_eps();
        mvnAttrs.initAcrossChannels_ = mvnOp->get_across_channels();
    } else {
        OPENVINO_THROW_NOT_IMPLEMENTED("Node is not an instance of MVN from the operation set v0 or v6");
    }
    mvnAttrs.execAcrossChannels_ = mvnAttrs.initAcrossChannels_;
}

void MVN::getSupportedDescriptors() {}

static inline bool isUnaryEltwise(const NodePtr& node) {
    return one_of(node->getAlgorithm(),
                  Algorithm::EltwiseRelu,
                  Algorithm::EltwiseGeluErf,
                  Algorithm::EltwiseGeluTanh,
                  Algorithm::EltwiseElu,
                  Algorithm::EltwiseSigmoid,
                  Algorithm::EltwiseClamp,
                  Algorithm::EltwiseTanh,
                  Algorithm::EltwiseSwish,
                  Algorithm::EltwiseHswish,
                  Algorithm::EltwiseMish,
                  Algorithm::EltwiseHsigmoid,
                  Algorithm::EltwiseRoundHalfToEven,
                  Algorithm::EltwiseRoundHalfAwayFromZero,
                  Algorithm::EltwiseAbs,
                  Algorithm::EltwiseSqrt,
                  Algorithm::EltwiseSoftRelu);
}

void MVN::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    ov::element::Type inputPrecision = getOriginalInputPrecisionAtPort(0);
    ov::element::Type outputPrecision = getOriginalOutputPrecisionAtPort(0);
    if (!hasHardwareSupport(outputPrecision)) {
        outputPrecision = ov::element::f32;
    }

    if (!fusedWith.empty()) {
        outputPrecision = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
        onlyUnaryPostOps = true;
        for (auto& node : fusedWith) {
            if (isUnaryEltwise(node)) {
                continue;
            }
            onlyUnaryPostOps = false;
            break;
        }
    }

    // Create initial memory descriptors
    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    auto srcDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(inputPrecision, getInputShapeAtPort(0));
    auto dstDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(outputPrecision, getOutputShapeAtPort(0));

    // Prepare memory descriptor args for executor factory
    MemoryDescArgs descs;
    descs[ARG_SRC_0] = srcDesc;
    descs[ARG_DST] = dstDesc;

    // Set minimal required fields in mvnAttrs for getProperMemoryDescriptors
    mvnAttrs.src_prc = inputPrecision;
    mvnAttrs.dst_prc = outputPrecision;
    mvnAttrs.layout = MVNLayoutType::mvn_planar;  // Initial layout, will be updated in prepareParams

    // Create planar configuration
    auto planarSrcDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(inputPrecision, getInputShapeAtPort(0));
    auto planarDstDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(outputPrecision, getOutputShapeAtPort(0));

    // Create channel-last configuration
    auto nspcSrcDesc = creatorsMap.at(LayoutType::nspc)->createSharedDesc(inputPrecision, getInputShapeAtPort(0));
    auto nspcDstDesc = creatorsMap.at(LayoutType::nspc)->createSharedDesc(outputPrecision, getOutputShapeAtPort(0));

    // Create configurations
    std::vector<std::pair<MemoryDescPtr, MemoryDescPtr>> configurations = {{planarSrcDesc, planarDstDesc},
                                                                           {nspcSrcDesc, nspcDstDesc}};

    // Add blocked layout configurations for 4D and 5D tensors
    if (getInputShapeAtPort(0).getRank() == 4 && (creatorsMap.count(LayoutType::nCsp8c) != 0u)) {
        auto blocked8SrcDesc =
            creatorsMap.at(LayoutType::nCsp8c)->createSharedDesc(inputPrecision, getInputShapeAtPort(0));
        auto blocked8DstDesc =
            creatorsMap.at(LayoutType::nCsp8c)->createSharedDesc(outputPrecision, getOutputShapeAtPort(0));
        configurations.emplace_back(blocked8SrcDesc, blocked8DstDesc);
    }
    if (getInputShapeAtPort(0).getRank() == 4 && (creatorsMap.count(LayoutType::nCsp16c) != 0u)) {
        auto blocked16SrcDesc =
            creatorsMap.at(LayoutType::nCsp16c)->createSharedDesc(inputPrecision, getInputShapeAtPort(0));
        auto blocked16DstDesc =
            creatorsMap.at(LayoutType::nCsp16c)->createSharedDesc(outputPrecision, getOutputShapeAtPort(0));
        configurations.emplace_back(blocked16SrcDesc, blocked16DstDesc);
    }
    if (getInputShapeAtPort(0).getRank() == 5 && (creatorsMap.count(LayoutType::nCsp8c) != 0u)) {
        auto blocked8SrcDesc =
            creatorsMap.at(LayoutType::nCsp8c)->createSharedDesc(inputPrecision, getInputShapeAtPort(0));
        auto blocked8DstDesc =
            creatorsMap.at(LayoutType::nCsp8c)->createSharedDesc(outputPrecision, getOutputShapeAtPort(0));
        configurations.emplace_back(blocked8SrcDesc, blocked8DstDesc);
    }
    if (getInputShapeAtPort(0).getRank() == 5 && (creatorsMap.count(LayoutType::nCsp16c) != 0u)) {
        auto blocked16SrcDesc =
            creatorsMap.at(LayoutType::nCsp16c)->createSharedDesc(inputPrecision, getInputShapeAtPort(0));
        auto blocked16DstDesc =
            creatorsMap.at(LayoutType::nCsp16c)->createSharedDesc(outputPrecision, getOutputShapeAtPort(0));
        configurations.emplace_back(blocked16SrcDesc, blocked16DstDesc);
    }

    // TODO [DS]: inplace
    bool canBeInplace = !isDynamicNode() && (inputPrecision.size() == outputPrecision.size()) &&
                        (getParentEdgeAt(0)->getParent()->getChildEdges().size() == 1) &&
                        !getParentEdgeAt(0)->getParent()->isConstant();

    const size_t inputsNum = getParentEdges().size();

    // Create supported primitive descriptors for each layout configuration
    for (const auto& config : configurations) {
        NodeConfig nodeConfig;
        nodeConfig.inConfs.resize(inputsNum);
        nodeConfig.outConfs.resize(1);
        nodeConfig.inConfs[0].constant(false);
        nodeConfig.outConfs[0].constant(false);
        nodeConfig.inConfs[0].inPlace(-1);
        nodeConfig.outConfs[0].inPlace(canBeInplace ? 0 : -1);
        if (inputsNum == 2) {
            nodeConfig.inConfs[1].setMemDesc(
                std::make_shared<CpuBlockedMemoryDesc>(ov::element::i32, getInputShapeAtPort(1)));
            nodeConfig.inConfs[1].constant(true);
        }

        // Use the layout-specific descriptors
        nodeConfig.inConfs[0].setMemDesc(config.first);
        nodeConfig.outConfs[0].setMemDesc(config.second);

        supportedPrimitiveDescriptors.emplace_back(nodeConfig, impl_desc_type::undef);
    }
}

void MVN::prepareParams() {
    auto dstMemPtr = getDstMemoryAtPort(0);
    auto srcMemPtr = getSrcMemoryAtPort(0);
    if (!dstMemPtr || !dstMemPtr->isDefined()) {
        THROW_CPU_NODE_ERR("Destination memory is undefined.");
    }
    if (!srcMemPtr || !srcMemPtr->isDefined()) {
        THROW_CPU_NODE_ERR("Input memory is undefined.");
    }
    if (getSelectedPrimitiveDescriptor() == nullptr) {
        THROW_CPU_NODE_ERR("Preferable primitive descriptor is not set.");
    }

    const VectorDims in_dims = srcMemPtr->getStaticDims();
    transformTo5DCase(in_dims);
    mvnAttrs.shape5D = shape5D;

    auto* selectedPD = getSelectedPrimitiveDescriptor();
    mvnAttrs.src_prc = selectedPD->getConfig().inConfs[0].getMemDesc()->getPrecision();
    mvnAttrs.dst_prc = selectedPD->getConfig().outConfs[0].getMemDesc()->getPrecision();
    if (getParentEdgeAt(0)->getMemory().getDesc().hasLayoutType(LayoutType::ncsp)) {
        mvnAttrs.layout = MVNLayoutType::mvn_planar;
    } else if (getParentEdgeAt(0)->getMemory().getDesc().hasLayoutType(LayoutType::nspc)) {
        mvnAttrs.layout = MVNLayoutType::mvn_by_channel;
    } else {
        mvnAttrs.layout = MVNLayoutType::mvn_block;
    }

    // Use modern executor factory pattern for all implementations
    MemoryArgs memoryArgs;
    memoryArgs[ARG_SRC_0] = getSrcMemoryAtPort(0);
    memoryArgs[ARG_DST] = getDstMemoryAtPort(0);

    MemoryDescArgs descs;
    descs[ARG_SRC_0] = getSrcMemoryAtPort(0)->getDescPtr();
    descs[ARG_DST] = getDstMemoryAtPort(0)->getDescPtr();

    // Set post-ops in mvnAttrs
    dnnl::primitive_attr attr;
    setPostOps(attr, true);

    // Pass post-ops data pointers through MVNAttrs
    // This allows the executor to access the actual post-ops data
    mvnAttrs.postOps.clear();
    for (const auto* ptr : postOpsDataPtrs) {
        mvnAttrs.postOps.emplace_back(ptr);
    }

    auto factory =
        std::make_shared<ExecutorFactory<MVNAttrs>>(mvnAttrs,
                                                    std::make_shared<ExecutorContext>(context, getImplPriority()),
                                                    descs);

    auto execPtr = factory->make(memoryArgs);
    if (!execPtr) {
        THROW_CPU_NODE_ERR("Failed to create MVN executor");
    }

    executorPtr = execPtr;

    // Update executor with memory arguments
    executorPtr->update(memoryArgs);

    selectedPD->setImplementationType(executorPtr->implType());
}

void MVN::transformTo5DCase(const VectorDims& shape) {
    size_t rank = shape.size();
    // for 1 and 2 rank, if initAcrossChannels_ is true, adjust shape to fully vectorize under unified 5d procedure.
    // otherwise there are not enough data in spatial dimension to process in one kernel.
    switch (rank) {
    case 1:  // C
        if (mvnAttrs.initAcrossChannels_) {
            shape5D = {1, 1, 1, 1, shape[0]};
            mvnAttrs.execAcrossChannels_ = false;
            break;
        } else {
            shape5D = {1, shape[0], 1, 1, 1};
            break;
        }
    case 2:  // NC
        if (mvnAttrs.initAcrossChannels_) {
            shape5D = {1, shape[0], 1, shape[1], 1};
            mvnAttrs.execAcrossChannels_ = false;
            break;
        } else {
            shape5D = {shape[0], shape[1], 1, 1, 1};
            break;
        }
    case 3: {
        shape5D = {shape[0], shape[1], 1, shape[2], 1};
        break;
    }
    case 4: {
        shape5D = {shape[0], shape[1], 1, shape[2], shape[3]};
        break;
    }
    case 5: {
        shape5D = {shape[0], shape[1], shape[2], shape[3], shape[4]};
        break;
    }
    default: {
        THROW_CPU_NODE_ERR("doesn't support planar layout with rank: ", shape.size());
    }
    }
}

void MVN::setPostOps(dnnl::primitive_attr& attr, [[maybe_unused]] bool initWeights) {
    dnnl::post_ops ops;
    postOpsDataPtrs.clear();
    for (auto& node : fusedWith) {
        int channelAxis = 1;

        auto* fakeQuantizeNode = dynamic_cast<FakeQuantize*>(node.get());
        if (fakeQuantizeNode) {
            fakeQuantizeNode->appendPostOps(ops, {}, postOpsDataPtrs, channelAxis);
            continue;
        }

        auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get());
        if (eltwiseNode) {
            eltwiseNode->appendPostOps(ops, shape5D, postOpsDataPtrs, channelAxis);
            continue;
        }
        THROW_CPU_NODE_ERR("Fusing of ",
                           NameFromType(node->getType()),
                           " operation to ",
                           NameFromType(this->getType()),
                           " node is not implemented");
    }
    attr.set_post_ops(ops);
}

void MVN::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void MVN::execute([[maybe_unused]] const dnnl::stream& strm) {
    if (executorPtr) {
        MemoryArgs memoryArgs;
        memoryArgs[ARG_SRC_0] = getSrcMemoryAtPort(0);
        memoryArgs[ARG_DST] = getDstMemoryAtPort(0);

        executorPtr->execute(memoryArgs);
    } else {
        THROW_CPU_NODE_ERR("Primitive wasn't created");
    }
}

bool MVN::canFuse(const NodePtr& node) const {
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41)) {
        return false;
    }
    // limit post-ops to unary when shape transformed on channel
    // 1D only fused with unary
    int inputRank = getInputShapeAtPort(0).getRank();
    bool unaryEltwise = isUnaryEltwise(node);
    if ((inputRank == 1 && !unaryEltwise) || (inputRank == 2 && !unaryEltwise && mvnAttrs.initAcrossChannels_)) {
        return false;
    }

    return canFuseSimpleOperation(node);
}

bool MVN::created() const {
    return getType() == Type::MVN;
}

}  // namespace ov::intel_cpu::node
