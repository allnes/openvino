// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "matmul.h"

#include "ngraph/opsets/opset1.hpp"
#include "ie_precision.hpp"
#include "cpu_types.h"
#include "eltwise.h"
#include "fake_quantize.h"
#include "utils/general_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"

#include <numeric>
#include <string>
#include <vector>
#include <memory>

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {
namespace {

struct MatMulKey {
    DnnlMemoryDescCPtr inp0;
    DnnlMemoryDescCPtr inp1;
    DnnlMemoryDescCPtr bias;
    DnnlMemoryDescCPtr out;
    dnnl::primitive_attr attr;
    impl_desc_type implType;

    size_t hash() const;
    bool operator==(const MatMulKey& rhs) const;
};

size_t MatMulKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    for (const auto& ptr : {inp0, inp1, bias, out}) {
        if (ptr) {
            seed = hash_combine(seed, get_md_hash(*ptr->getDnnlDesc().get()));
        }
    }

    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    seed = hash_combine(seed, implType);
    return seed;
}

bool MatMulKey::operator==(const MatMulKey &rhs) const {
    bool retVal = true;
    if (inp0 != rhs.inp0) {
        retVal = retVal && inp0 && rhs.inp0 && inp0->getDnnlDesc() == rhs.inp0->getDnnlDesc();
    }
    if (inp1 != rhs.inp1) {
        retVal = retVal && inp1 && rhs.inp1 && inp1->getDnnlDesc() == rhs.inp1->getDnnlDesc();
    }
    if (bias != rhs.bias) {
        retVal = retVal && bias && rhs.bias && bias->getDnnlDesc() == rhs.bias->getDnnlDesc();
    }
    if (out != rhs.out) {
        retVal = retVal && out && rhs.out && out->getDnnlDesc() == rhs.out->getDnnlDesc();
    }
    retVal = retVal && *attr.get() == *rhs.attr.get() &&
             implType == rhs.implType;
    return retVal;
}

bool canBeExecutedInInt8(const Precision& firstInput, const Precision& secondInput) {
    return one_of(firstInput, Precision::U8, Precision::I8) && secondInput == Precision::I8;
}
} // namespace

bool MatMul::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);
        if (!matMul) {
            errorMessage = "Only opset1 MatMul operation is supported";
            return false;
        }

        for (size_t i = 0; i < matMul->get_input_size(); i++) {
            const auto inShapeRank = matMul->get_input_partial_shape(i).rank().get_length();
            if (inShapeRank < 2) {
                errorMessage = "Unsupported rank: " + std::to_string(inShapeRank) + " on " + std::to_string(i) + " input";
                return false;
            }
        }

        const auto outShapeRank = matMul->get_output_partial_shape(0).rank().get_length();
        if (outShapeRank < 2) {
            errorMessage = "Unsupported rank: " + std::to_string(outShapeRank) + " on output";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

namespace {
class MMShapeInfer : public ShapeInferEmptyPads {
public:
    MMShapeInfer(const size_t& out_rank, const bool& transpose_a, const bool& transpose_b) :
        m_out_rank(out_rank), m_transpose_a(transpose_a), m_transpose_b(transpose_b) {
        m_shapeY = VectorDims(m_out_rank, 1); // for output and cache
    }
    Result infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        const VectorDims& shapeA = input_shapes[0].get();
        const VectorDims& shapeB = input_shapes[1].get();
        const size_t rankA = shapeA.size();
        const size_t rankB = shapeB.size();

        // getSupportedDescriptors has done some shape check.
        // 1. Needn't assert the scalar type since the matmul_shape_inference has checked.
        // 2. Needn't check the compatibility of the last two dims
        // 3. 1-D x 1-D is needed
        // 4. transpose is necessary
        // 5. Just support the same rank of matmul
        // 6. simplify the broadcast check
        if (rankA == 1 && rankB == 1 && shapeA[0] == shapeB[0]) {
            return {{m_shapeY}, ShapeInferStatus::success};
        }

        m_shapeY[m_out_rank-2] = m_transpose_a ? shapeA[rankA-1] : shapeA[rankA-2];
        m_shapeY[m_out_rank-1] = m_transpose_b ? shapeB[rankB-2] : shapeB[rankB-1];

        for (size_t i=0; i < m_out_rank-2; ++i) {
            if (shapeA[i] != shapeB[i]) {
                if (shapeB[i] == 1) {
                    m_shapeY[i] = shapeA[i];
                    continue;
                } else if (shapeA[i] != 1) {
                    IE_THROW() << "Incompatible MatMul batch dimension. Cant merge the first input dimension=" <<
                                  shapeA[i] << " with second input dimension=" << shapeB[i] << " at index=" << i;
                }
            }
            m_shapeY[i] = shapeB[i];
        }

        return {{m_shapeY}, ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    VectorDims m_shapeY;
    const size_t m_out_rank;
    const bool m_transpose_a;
    const bool m_transpose_b;
};

class MMShapeInferFactory : public ShapeInferFactory {
public:
    MMShapeInferFactory(const std::shared_ptr<ngraph::Node>& op) : m_op(op) {}
    ShapeInferPtr makeShapeInfer() const override {
        if (const auto matmul = ov::as_type_ptr<const ngraph::opset1::MatMul>(m_op)) {
            const auto output_rank = matmul->get_output_partial_shape(0).rank().get_length();
            const bool transpose_a = matmul->get_transpose_a();
            const bool transpose_b = matmul->get_transpose_b();
            return std::make_shared<MMShapeInfer>(output_rank, transpose_a, transpose_b);
       } else {
             IE_THROW() << "Unexpected operation type in the MatMul shape inference factory";
       }
    }
private:
    std::shared_ptr<ngraph::Node> m_op;
};
} // namespace

MatMul::MatMul(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context) :
    Node(op, context, MMShapeInferFactory(op)), withBiases(false) {
    std::string errorMessage;
    errorPrefix = "MatMul node with name '" + getName() + "'";

    if (!isSupportedOperation(op, errorMessage))
        IE_THROW(NotImplemented) << errorMessage;

    const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);

    if (!matMul) {
        IE_THROW(NotImplemented) << "Operation with name " << op->get_friendly_name() << ":" << op->get_type_name() <<
            " is not an instance of MatMul from opset1";
    }

    matmulAttrs.transposeA = matMul->get_transpose_a();
    matmulAttrs.transposeB = matMul->get_transpose_b();
}

bool MatMul::canFuse(const NodePtr& node) const {
    return canFuseSimpleOperation(node);
}

void MatMul::setPostOps(dnnl::primitive_attr& attr, const VectorDims& dims, bool initWeights = false) {
    dnnl::post_ops ops;

    dnnl::memory::data_type outputDataType = DnnlExtensionUtils::IEPrecisionToDataType(outputPrecisions[0]);

    bool isINT8 = canBeExecutedInInt8(getOriginalInputPrecisionAtPort(0), getOriginalInputPrecisionAtPort(1));

    DnnlPostOpsComposer dnnlpoc(getEngine(), attr, ops, postOpsArgs, dims, dims.size() - 1, isINT8);

    for (int i = 0; i < fusedWith.size(); ++i) {
        auto& node = fusedWith[i];
        bool isLastPostOp = (i == (fusedWith.size() - 1));

        if (auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get())) {
            eltwiseNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
            continue;
        }

        if (auto* fakeQuantizeNode = dynamic_cast<FakeQuantize*>(node.get())) {
            fakeQuantizeNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType())
                   << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

Node::AttrPtr MatMul::initPrimitiveAttr(const VectorDims &dims) {
    auto attr = std::make_shared<dnnl::primitive_attr>(dnnl::primitive_attr());

    setPostOps(*attr, dims, true);

    return attr;
}

Node::AttrPtr MatMul::initPrimitiveAttr() {
    auto dummyShape = MemoryDescUtils::makeDummyShape(getOutputShapeAtPort(0));
    return initPrimitiveAttr(dummyShape.getStaticDims());
}

void MatMul::getSupportedDescriptors() {
}

void MatMul::initSupportedPrimitiveDescriptors() {
    matmulAttrs.withBias = getOriginalInputsNumber() == 3;

    inputPrecisions = getOriginalInputPrecisions();
    outputPrecisions = getOriginalOutputPrecisions();

    if (inputPrecisions[0].size() != inputPrecisions[1].size())
        inputPrecisions[0] = inputPrecisions[1] = getMaxPrecision(getOriginalInputPrecisions());

    // fallback to fp32 for any precision that cannot be handled natively
    if ((!one_of(inputPrecisions[0] , Precision::U8, Precision::I8, Precision::BF16, Precision::FP32) ||
         !one_of(inputPrecisions[1] , Precision::I8, Precision::BF16, Precision::FP32))) {
        outputPrecisions[0] = inputPrecisions[0] = inputPrecisions[1] = Precision::FP32;
    }

    Precision postOpsPrec = outPortPrec;
    if (!fusedWith.empty()) {
        postOpsPrec = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
    }

    if (canBeExecutedInInt8(firstInPortPrec, secondInPortPrec)) {
        // INT8 mode support wide range of output precisions
        outPortPrec = postOpsPrec;
    } else if (postOpsPrec == Precision::FP32) {
        // all non-INT8 modes support fp32 output precision
        outPortPrec = postOpsPrec;
    } else {
        // otherwise we ignore postOpsPrec and stay with getOriginalOutputPrecisionAtPort(0)
    }

    const auto& inputShape0 = getInputShapeAtPort(0);
    const auto& inputShape1 = getInputShapeAtPort(1);
    const auto& outputShape = getOutputShapeAtPort(0);

    if (inputShape0.getRank() != inputShape1.getRank() || inputShape0.getRank() != outputShape.getRank())
        IE_THROW()  << errorPrefix << " has invalid dims count";

    const int nDims = inputShape0.getRank();
    const auto xAxis = nDims - 1;
    const auto yAxis = nDims - 2;
    const auto xAxis0 = transposeIn[0] ? yAxis : xAxis;
    const auto yAxis0 = transposeIn[0] ? xAxis : yAxis;
    const auto xAxis1 = transposeIn[1] ? yAxis : xAxis;
    const auto yAxis1 = transposeIn[1] ? xAxis : yAxis;

    const auto& inDims0 = getInputShapeAtPort(0).getDims();
    const auto& inDims1 = getInputShapeAtPort(1).getDims();
    const auto& outDims = getOutputShapeAtPort(0).getDims();

    // coverity[copy_paste_error]
    if (!dimsEqualWeak(inDims0[xAxis0], inDims1[yAxis1]) ||
        !dimsEqualWeak(inDims0[yAxis0], outDims[yAxis]) ||
        !dimsEqualWeak(inDims1[xAxis1], outDims[xAxis]))
        IE_THROW()  << errorPrefix << " has incorrect spatial input and output dimensions";

    for (int dim_idx = nDims - 3; dim_idx >= 0; dim_idx--) {
        if ((!dimsEqualWeak(inDims0[dim_idx], outDims[dim_idx]) &&
             !dimsEqualWeak(inDims0[dim_idx], 1)) ||
            (!dimsEqualWeak(inDims1[dim_idx], outDims[dim_idx]) &&
             !dimsEqualWeak(inDims1[dim_idx], 1))) {
            IE_THROW()  << errorPrefix << " has incorrect input batch dimensions";
        }
    }

    if (!canBeExecutedInInt8( inputPrecisions[0], inputPrecisions[1]) && one_of(outputPrecisions[0], Precision::U8, Precision::I8))
        outputPrecisions[0] = Precision::FP32; // INT output is not supported for non-INT inputs


    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    NodeConfig config;
    config.dynBatchSupport = true;
    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        PortConfig portConfig;
        portConfig.inPlace(-1);
        portConfig.constant(false);
        portConfig.setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(inputPrecisions[i], getInputShapeAtPort(i)));

        config.inConfs.push_back(portConfig);
    }

    for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
        PortConfig portConfig;
        portConfig.inPlace(canBeInPlace() ? 0 : -1);
        portConfig.constant(false);
        portConfig.setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(outputPrecisions[i], getOutputShapeAtPort(i)));

        config.outConfs.push_back(portConfig);
    }

    std::vector<MemoryDescPtr> srcMemoryDescs;
    for (int i = 0; i < config.inConfs.size(); i++) {
        srcMemoryDescs.push_back(config.inConfs[i].getMemDesc());
    }

    swapTranspDims(inDims0, inDims1);

    return {Shape(inDims0), Shape(inDims1)};
}

void MatMul::createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                              const std::vector<MemoryDescPtr>& outputDesc) {
    const auto attr = initPrimitiveAttr();
    dnnl::matmul::primitive_desc matmul_desc;
    if (withBiases) {
        matmul_desc = matmul::primitive_desc(
            getEngine(),
            inDataDesc[0]->getDnnlDesc(),
            inDataDesc[1]->getDnnlDesc(),
            getBiasDescFrom(outDataDesc),
            outDataDesc->getDnnlDesc(),
            *attr);
    } else {
        matmul_desc = matmul::primitive_desc(
            getEngine(),
            inDataDesc[0]->getDnnlDesc(),
            inDataDesc[1]->getDnnlDesc(),
            outDataDesc->getDnnlDesc(),
            *attr);
    }

    descs.emplace_back(matmul_desc);
}

void MatMul::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    for (auto& desc : descs) {
        auto itpd = desc;
        while (itpd) {
            NodeConfig config;
            config.dynBatchSupport = true;
            for (size_t i = 0; i < descInputNumbers(); i++) {
                PortConfig portConfig;
                portConfig.inPlace(-1);
                portConfig.constant(false);
                portConfig.setMemDesc(getSrcMemDesc(itpd, i));

                config.inConfs.push_back(portConfig);
            }

            for (size_t i = 0; i < descOutputNumbers(); i++) {
                PortConfig portConfig;
                portConfig.inPlace(canBeInPlace() ? 0 : -1);
                portConfig.constant(false);
                portConfig.setMemDesc(getDstMemDesc(itpd, i));

                config.outConfs.push_back(portConfig);
            }

            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

            supportedPrimitiveDescriptors.emplace_back(config, impl_type);
            if (!itpd.next_impl())
                break;
        }
    }
}

MemoryDescPtr MatMul::getSrcMemDesc(dnnl::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    auto desc = idx > 0 ? primitive_desc_it.weights_desc(idx - 1): primitive_desc_it.src_desc(idx);

    if (idx < 2) // inputs
        return std::make_shared<CpuBlockedMemoryDesc>(
            DnnlExtensionUtils::DataTypeToIEPrecision(desc.get_data_type()),
            getInputShapeAtPort(idx)); /* provide initial shapes, so hide transpose effect */
    else // bias
        return DnnlExtensionUtils::makeDescriptor(desc);
}

bool MatMul::created() const {
    return getType() == Type::MatMul;
}

size_t MatMul::getMaxBatch() const {
    if (!outputShapes.empty())
        return outputShapes[0].getStaticDims()[0];
    return 0;
}

InferenceEngine::Precision MatMul::getRuntimePrecision() const {
    return getMaxPrecision(getInputPrecisions());
}

void MatMul::prepareParams() {
    std::vector<MemoryDescPtr> srcMemoryDescs;
    for (int i = 0; i < getOriginalInputsNumber(); i++) {
        srcMemoryDescs.push_back(getParentEdgeAt(i)->getMemoryPtr()->getDescPtr());
    }
    std::vector<MemoryDescPtr> dstMemoryDescs;
    for (int i = 0; i < getOriginalOutputsNumber(); i++) {
        dstMemoryDescs.push_back(getChildEdgeAt(i)->getMemoryPtr()->getDescPtr());
    }

    MatMulKey key = {src0TransposedDesc, src1TransposedDesc, dnnlBiasMemDesc,
                     dstDnnlDesc, *attr, selected_pd->getImplementationType()};

    auto engine = getEngine();

    auto builder = [&engine](const MatMulKey& key) -> executorPtr {
        dnnl::matmul::primitive_desc matmul_desc;

        if (key.bias) {
            matmul_desc = matmul::primitive_desc(
                engine,
                key.inp0->getDnnlDesc(),
                key.inp1->getDnnlDesc(),
                key.bias->getDnnlDesc(),
                key.out->getDnnlDesc(),
                key.attr);
        } else {
            matmul_desc = matmul::primitive_desc(
                engine,
                key.inp0->getDnnlDesc(),
                key.inp1->getDnnlDesc(),
                key.out->getDnnlDesc(),
                key.attr);
        }

        primitive_desc_iterator itpd = matmul_desc;
        matmul::primitive_desc prim_desc;

        auto itpd_first = itpd;
        while (static_cast<bool>(itpd))  {
            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

            if (impl_type == key.implType) {
                prim_desc = itpd.get();
                break;
            }
            if (!itpd.next_impl()) {
                // In case of dynamic shapes an implementation type chosen as optimal for a primitive_desc with
                // undefined input shapes, is not necessarily available for the primitive_desc with defined shape.
                // Example: brgemm_avx512_amx (Intel Sapphire Rapids Platform) is available for a primitive with
                // undefined input shapes but not available for primitive_desc with input batch 1.
                prim_desc = itpd_first.get();
                break;
            }
        }
        return std::make_shared<DnnlExecutor>(prim_desc);
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);

    execPtr = result.first;
    if (!execPtr) {
        IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
    }

    auto schratchpadMem = getScratchPadMem(execPtr->getScratchPadDesc());

    primArgs[DNNL_ARG_SCRATCHPAD] = schratchpadMem->GetPrimitive();
    primArgs[DNNL_ARG_SRC_0] = src0MemPtr->GetPrimitive();
    primArgs[DNNL_ARG_WEIGHTS_0] = src1MemPtr->GetPrimitive();
    primArgs[DNNL_ARG_DST] = dstMemPtr->GetPrimitive();
    if (withBiases)
        primArgs[DNNL_ARG_BIAS] = getParentEdgeAt(2)->getMemoryPtr()->GetPrimitive();

    appendPostOpArgs(*attr, primArgs, postOpsArgs);
#ifdef CPU_DEBUG_CAPS
    if (result.second == CacheEntryBase::LookUpStatus::Miss) {
        auto pd = execPtr->getPrimitiveDesc();
        DEBUG_LOG("verbose##", getName(), "##", DnnlExtensionUtils::query_pd_info(pd), "\n");
    }
#endif
}

void MatMul::execute(dnnl::stream strm) {
    if (execPtr) {
        execPtr->exec(primArgs, strm);
    } else {
        IE_THROW() << errorPrefix << " doesn't have an initialized executor";
    }
}

void MatMul::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

const std::vector<impl_desc_type>& MatMul::getPrimitivesPriority() {
    std::vector<impl_desc_type> priorities;
    for (const auto& impl : priorities) {
        if (std::find(implPriorities.begin(), implPriorities.end(), impl) == implPriorities.end())
            implPriorities.push_back(impl);
    }
    return implPriorities;
}
}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
