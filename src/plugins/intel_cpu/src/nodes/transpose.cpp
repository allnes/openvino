// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose.h"
#include "ie_parallel.hpp"

#include <algorithm>
#include <string>
#include <dnnl_extension_utils.h>
#include <common/primitive_hashing_utils.hpp>

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {
namespace {
struct TransposeAsReorderKey {
    dnnl::memory::desc src;
    dnnl::memory::desc dest;
    size_t hash() const;
    bool operator==(const TransposeAsReorderKey& rhs) const;
};

size_t TransposeAsReorderKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    seed = hash_combine(seed, get_md_hash(src.data));
    seed = hash_combine(seed, get_md_hash(dest.data));

    return seed;
}

bool TransposeAsReorderKey::operator==(const TransposeAsReorderKey& rhs) const {
    bool retVal = true;
    retVal = src == rhs.src && dest == rhs.dest;
    return retVal;
}
}  // namespace

bool Transpose::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                ov::op::v1::Transpose::get_type_info_static())) {
            errorMessage = "Node is not an instance of the Transpose operation from opset1.";
            return false;
        }

        if (op->get_input_node_ptr(INPUT_ORDER_IDX)->get_type_info() != ov::op::v0::Constant::get_type_info_static()) {
            // TODO: Support parameterized Order input for dynamic shapes.
            errorMessage = "Constant expected as the second input for static shapes.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Transpose::Transpose(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    if (op->get_input_node_ptr(INPUT_ORDER_IDX)->get_type_info() == ov::op::v0::Constant::get_type_info_static()) {
        isInputOrderConst = true;
        order = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(INPUT_ORDER_IDX))->cast_vector<size_t>();

        if (order.empty()) {
            size_t rank = getInputShapeAtPort(INPUT_DATA_IDX).getRank();
            for (size_t i = 1lu; i <= rank; ++i) {
                order.emplace_back(rank - i);
            }
        }
    }
}

void Transpose::getSupportedDescriptors() {
}

void Transpose::initSupportedPrimitiveDescriptors() {
    std::cout << "Transpose::initSupportedPrimitiveDescriptors" << std::endl;
    if (!supportedPrimitiveDescriptors.empty())
        return;

    prec = getOriginalInputPrecisionAtPort(0);

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();

    NodeConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(2);
    config.outConfs.resize(1);
    config.inConfs[INPUT_DATA_IDX].inPlace(-1);
    config.inConfs[INPUT_DATA_IDX].constant(false);
    config.inConfs[INPUT_ORDER_IDX].constant(isInputOrderConst);
    config.inConfs[INPUT_ORDER_IDX].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            Precision::I32, getInputShapeAtPort(INPUT_ORDER_IDX)));
    config.outConfs[0].inPlace(-1);
    config.outConfs[0].constant(false);

    const auto& inputDataShape = getInputShapeAtPort(INPUT_DATA_IDX);
    const auto& outputDataShape = getOutputShapeAtPort(0);
    if (inputDataShape.getRank() == 4 || inputDataShape.getRank() == 5) {
        config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(prec, inputDataShape));
        config.outConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(prec, outputDataShape));
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown});

        const auto& srcDims = inputDataShape.getDims();
        if (srcDims[1] != Shape::UNDEFINED_DIM && srcDims[1] % 8 == 0) {
            config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::nCsp8c)->createSharedDesc(prec, inputDataShape));
            supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown});
        }

        if (srcDims[1] != Shape::UNDEFINED_DIM && srcDims[1] % 16 == 0) {
            config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::nCsp16c)->createSharedDesc(prec, inputDataShape));
            supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown});
        }

        if (prec == Precision::FP32 || prec == Precision::I8 || prec == Precision::U8) {
            config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::nspc)->createSharedDesc(prec, inputDataShape));
            config.outConfs[0].setMemDesc(creatorsMap.at(LayoutType::nspc)->createSharedDesc(prec, outputDataShape));
            supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown});
        }
    } else {
        // general plain case
        config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(prec, inputDataShape));
        config.outConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(prec, outputDataShape));
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown});
    }
}

bool Transpose::isExecutable() const {
    return !isInputTensorAtPortEmpty(0);
}

bool Transpose::needPrepareParams() const {
    if (isOptimized)
        return false;
    return inputShapesModified();
}

void Transpose::prepareParams() {
    std::cout << "Transpose::prepareParams" << std::endl;
    if (performAsReorder) {
        dnnl::primitive_attr attr;
        const auto engine = getEngine();
        auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
        auto& srcMemPtr = getParentEdgeAt(INPUT_DATA_IDX)->getMemoryPtr();
        MemoryPtr src_blocked = std::make_shared<Memory>(engine);
        MemoryPtr dst_blocked = std::make_shared<Memory>(engine);

        dst_blocked->Create(
            DnnlExtensionUtils::makeDescriptor(dstMemPtr->GetDescWithType<DnnlMemoryDesc>()->getDnnlDesc()),
            dstMemPtr->GetData(), false);

        const auto newDims = dst_blocked->getStaticDims();
        auto newDesc = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(newDims),
                                            dst_blocked->GetDataType(),
                                            memory::format_tag::acdb);
        src_blocked->Create(DnnlExtensionUtils::makeDescriptor(newDesc), srcMemPtr->GetData(), false);

        impl_desc_type impl_type = getSelectedPrimitiveDescriptor()->getImplementationType();
        TransposeAsReorderKey key = {src_blocked->GetPrimitive().get_desc(), dst_blocked->GetPrimitive().get_desc()};
        auto builder = [&engine, &impl_type](const TransposeAsReorderKey& key) -> std::shared_ptr<dnnl::primitive> {
            dnnl::primitive_attr attr;
            reorder::primitive_desc pd = dnnl::reorder::primitive_desc(engine, key.src, engine, key.dest, attr, true);

            if (!pd)
                return nullptr;
            auto info = pd.impl_info_str();
            impl_type = parse_impl_name(info);
            return std::make_shared<dnnl::reorder>(pd);
        };

        auto cache = context->getParamsCache();
        auto result = cache->getOrCreate(key, builder);

        if (!result.first) {
            IE_THROW() << "Reorder primitive descriptor was not found for Transpose node " << getName() << ".";
        }

        prim = result.first;

        supportedPrimitiveDescriptors[0].setImplementationType(impl_type);
        primArgs = {{DNNL_ARG_SRC, getParentEdgesAtPort(INPUT_DATA_IDX)[0]->getMemoryPtr()->GetPrimitive()},
                    {DNNL_ARG_DST, getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive()}};
        return;
    }

    auto srcDesc = getParentEdgeAt(INPUT_DATA_IDX)->getMemory().GetDescWithType<BlockedMemoryDesc>();
    params.src_block_dims = srcDesc->getBlockDims();
    auto dstDesc = getChildEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>();
    params.dst_block_dims = dstDesc->getBlockDims();

    if (!isInputOrderConst) {
        auto orderPtr = reinterpret_cast<const int32_t*>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
        auto orderLen = getParentEdgeAt(0)->getMemoryPtr()->GetSize();
        params.order.assign(orderPtr, orderPtr + orderLen);
    }

    auto engine = getEngine();
    auto builder = [&engine, &srcDesc, &dstDesc, this](const PermuteParams& key) -> std::shared_ptr<TransposeJitExecutor> {
        auto jitExec = std::make_shared<TransposeJitExecutor>(transpose_context);
        dnnl::primitive_attr attr;
        jitExec->init(key, {srcDesc}, {dstDesc}, attr);
        return jitExec;
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(params, builder);

    if (!result.first) {
        IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
    }

    execPtr = result.first;
}

void Transpose::createPrimitive() {
    std::cout << "Transpose::createPrimitive" << std::endl;
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(INPUT_DATA_IDX)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        IE_THROW() << "Destination memory was not allocated.";
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        IE_THROW() << "Input memory was not allocated.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << "Preferable primitive descriptor was not set.";

    transpose_context = std::make_shared<ExecutorContext>(context, getPrimitivesPriority());

    if (getParentEdgeAt(INPUT_DATA_IDX)->getMemory().getDesc().hasLayoutType(LayoutType::ncsp) &&
        getChildEdgeAt(0)->getMemory().getDesc().hasLayoutType(LayoutType::ncsp) &&
        order == std::vector<size_t>{0, 3, 1, 2}) {
        performAsReorder = true;
    } else if (getParentEdgeAt(INPUT_DATA_IDX)->getMemory().getDesc().hasLayoutType(LayoutType::ncsp) &&
            std::find(optimizedOrders.begin(), optimizedOrders.end(), order) != optimizedOrders.end()) {
        isOptimized = true;
        execPtr = std::make_shared<TransposeRefExecutor>(transpose_context);
        return;
    }

    if (!performAsReorder) {
        params.data_size = getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].getMemDesc()->getPrecision().size();
        if (isInputOrderConst)
            params.order = order;
        auto srcDesc = getParentEdgeAt(INPUT_DATA_IDX)->getMemory().GetDescWithType<BlockedMemoryDesc>();
        params.src_block_order = srcDesc->getOrder();
        auto dstDesc = getChildEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>();
        params.dst_block_order = dstDesc->getOrder();
    }

    if (inputShapesDefined() && isExecutable()) {
        prepareParams();
        updateLastInputDims();
    }
}

void Transpose::execute(dnnl::stream strm) {
    if (prim) {
        (*prim).execute(strm, primArgs);
    } else if (execPtr) {
        auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
        auto &srcMemPtr = getParentEdgeAt(INPUT_DATA_IDX)->getMemoryPtr();

        int MB = 0;
        if (isDynamicNode()) {
            MB = srcMemPtr->getStaticDims()[0];
        } else {
            MB = batchToProcess();
        }
        execPtr->exec({srcMemPtr}, {dstMemPtr}, MB);
    } else {
        IE_THROW() << "Could not execute Transpose node. Primitive was not created.";
    }
}

void Transpose::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool Transpose::created() const {
    return getType() == Type::Transpose;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
