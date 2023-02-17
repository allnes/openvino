// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interpolate.h"
#include "executors/common/ref_interpolate.hpp"

#include "fake_quantize.h"
#include "eltwise.h"
#include <string>
#include <vector>
#include <onednn/dnnl.h>
#include <dnnl_extension_utils.h>
#include "ie_parallel.hpp"
#include <algorithm>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <utils/shape_inference/static_shape.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <ie_ngraph_utils.hpp>
#include "utils/cpu_utils.hpp"
#include <utils/shape_inference/shape_inference_ngraph.hpp>

#include "common/cpu_memcpy.h"
#include "utils/bfloat16.hpp"
#include "emitters/x64/jit_load_store_emitters.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

namespace {
struct InterpolateKey {
    InterpolateAttrs nodeAttrs;
    VectorDims srcDims;
    VectorDims dstDims;
    dnnl::primitive_attr attr;

    size_t hash() const;
    bool operator==(const InterpolateKey& rhs) const;
};

size_t InterpolateKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    seed = hash_combine(seed, nodeAttrs.mode);
    seed = hash_combine(seed, nodeAttrs.coordTransMode);
    seed = hash_combine(seed, nodeAttrs.nearestMode);
    seed = hash_combine(seed, nodeAttrs.layout);

    seed = hash_combine(seed, nodeAttrs.antialias);
    seed = hash_combine(seed, nodeAttrs.cubeCoeff);

    seed = get_vector_hash(seed, nodeAttrs.padBegin);
    seed = get_vector_hash(seed, nodeAttrs.padEnd);

    seed = hash_combine(seed, nodeAttrs.inPrc.getPrecVal());
    seed = hash_combine(seed, nodeAttrs.outPrc.getPrecVal());

    seed = get_vector_hash(seed, srcDims);
    seed = get_vector_hash(seed, dstDims);
    seed = get_vector_hash(seed, nodeAttrs.dataScales);

    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    return seed;
}

bool InterpolateKey::operator==(const InterpolateKey &rhs) const {
    if (nodeAttrs.mode != rhs.nodeAttrs.mode)
        return false;
    if (nodeAttrs.coordTransMode != rhs.nodeAttrs.coordTransMode)
        return false;
    if (nodeAttrs.nearestMode != rhs.nodeAttrs.nearestMode)
        return false;
    if (nodeAttrs.layout != rhs.nodeAttrs.layout)
        return false;
    if (nodeAttrs.antialias != rhs.nodeAttrs.antialias)
        return false;
    if (nodeAttrs.cubeCoeff != rhs.nodeAttrs.cubeCoeff)
        return false;
    if (nodeAttrs.padBegin != rhs.nodeAttrs.padBegin)
        return false;
    if (nodeAttrs.padEnd != rhs.nodeAttrs.padEnd)
        return false;
    if (nodeAttrs.inPrc != rhs.nodeAttrs.inPrc)
        return false;
    if (nodeAttrs.outPrc != rhs.nodeAttrs.outPrc)
        return false;
    if (nodeAttrs.layout != rhs.nodeAttrs.layout)
        return false;

    if (srcDims != rhs.srcDims)
        return false;
    if (dstDims != rhs.dstDims)
        return false;
    if (nodeAttrs.dataScales != rhs.nodeAttrs.dataScales)
        return false;
    if (!(*attr.get() == *rhs.attr.get()))
        return false;

    return true;
}

} // namespace

// shapeND: n     c     d     h    w
// blockND: ncdhw cdhw  dhw   hw   w    1
// index  : 0      1    2     3    4    5
inline SizeVector getBlockND(const SizeVector& shape) {
    int shapeRank = shape.size();
    SizeVector blockND(shapeRank + 1, 1);
    for (int i = shapeRank - 1; i >= 0; i--) {
        blockND[i] = shape[i] * blockND[i+1];
    }
    return blockND;
}

using ngInterpMode = ngraph::opset4::Interpolate::InterpolateMode;
using ngInterpCoordTransf = ngraph::opset4::Interpolate::CoordinateTransformMode;
using ngInterpNearMode = ngraph::opset4::Interpolate::NearestMode;
using ngInterpShapeCalcMode = ngraph::opset4::Interpolate::ShapeCalcMode;

bool Interpolate::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto interp = std::dynamic_pointer_cast<const ngraph::opset4::Interpolate>(op);
        if (!interp) {
            errorMessage = "Only opset4 Interpolate operation is supported";
            return false;
        }
        const auto &interpAttr = interp->get_attrs();
        const auto &interpMode = interpAttr.mode;
        if (!one_of(interpMode, ngInterpMode::nearest, ngInterpMode::linear, ngInterpMode::linear_onnx, ngInterpMode::cubic)) {
            errorMessage = "Does not support interpolate mode: " + ngraph::as_string(interpMode);
            return false;
        }

        const auto &interpCoordTransMode = interpAttr.coordinate_transformation_mode;
        if (!one_of(interpCoordTransMode, ngInterpCoordTransf::half_pixel, ngInterpCoordTransf::pytorch_half_pixel, ngInterpCoordTransf::asymmetric,
                                          ngInterpCoordTransf::tf_half_pixel_for_nn, ngInterpCoordTransf::align_corners)) {
            errorMessage = "Does not support coordinate transformation mode: " + ngraph::as_string(interpCoordTransMode);
            return false;
        }

        if (interpMode == ngInterpMode::nearest) {
            const auto &interpNearestMode = interpAttr.nearest_mode;
            if (!one_of(interpNearestMode, ngInterpNearMode::round_prefer_floor, ngInterpNearMode::round_prefer_ceil, ngInterpNearMode::floor,
                                           ngInterpNearMode::ceil, ngInterpNearMode::simple)) {
                errorMessage = "Does not support nearest round mode: " + ngraph::as_string(interpNearestMode);
                return false;
            }
        }

        const auto &interpShapeCalcMode = interpAttr.shape_calculation_mode;
        if (!one_of(interpShapeCalcMode, ngInterpShapeCalcMode::scales, ngInterpShapeCalcMode::sizes)) {
            errorMessage = "Does not support shape_calculation_mode: " + ngraph::as_string(interpShapeCalcMode);
            return false;
        }

        const size_t dataRank = interp->get_input_partial_shape(DATA_ID).rank().get_length();
        if (dataRank < 1 || dataRank > 5) {
            errorMessage = "Does not support input tensor of rank : " + std::to_string(dataRank);
            return false;
        }

        if (dataRank == 5 && interpMode == ngInterpMode::cubic) {
            errorMessage = "Doesn't support input tensor with rank: " + std::to_string(dataRank) + " for 'cubic' mode ";
            return false;
        }

        if (!isDynamicNgraphNode(op) && interpShapeCalcMode == ngInterpShapeCalcMode::scales &&
                !ngraph::is_type<ngraph::opset1::Constant>(op->get_input_node_ptr(2))) {
            errorMessage = "Only const 'scales' input is supported for static shapes";
            return false;
        }

        if (interp->get_input_size() > 3 && std::dynamic_pointer_cast<const ngraph::opset1::Constant>(interp->get_input_node_shared_ptr(AXES_ID)) == nullptr) {
            errorMessage = "Only const 'axes' input is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

namespace {
/**
 * Interpolate shape inference factory. It defines the input mask depending on the shape calculation mode.
 *
 */
class InterpolateShapeInferFactory : public ShapeInferFactory {
public:
    InterpolateShapeInferFactory(std::shared_ptr<ngraph::Node> op) : m_op(op) {}
    ShapeInferPtr makeShapeInfer() const override {
        IShapeInfer::port_mask_t port_mask = 0x00;
        auto interp = ov::as_type_ptr<ngraph::opset4::Interpolate>(m_op);
        if (!interp) {
            IE_THROW(Unexpected) << "Wrong operation type";
        }
        const auto &attr = interp->get_attrs();

        if (attr.shape_calculation_mode == ngInterpShapeCalcMode::SCALES) {
            port_mask = PortMask(Interpolate::SCALES_ID, Interpolate::AXES_ID);
        } else if (attr.shape_calculation_mode == ngInterpShapeCalcMode::SIZES) {
            port_mask = PortMask(Interpolate::TARGET_SHAPE_ID, Interpolate::AXES_ID);
        } else {
            IE_ASSERT(false) << "Unsupported interpolate shape calculation mode";
        }

        return std::make_shared<NgraphShapeInfer>(make_shape_inference(m_op), port_mask);
    }
private:
    std::shared_ptr<ngraph::Node> m_op;
};
} // namespace

Interpolate::Interpolate(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, InterpolateShapeInferFactory(op)) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "Interpolate node with name '" + getName() + "'";

        const auto interp = std::dynamic_pointer_cast<const ngraph::opset4::Interpolate>(op);

        const auto numInputs = inputShapes.size();
        if (numInputs != 3 && numInputs != 4)
            IE_THROW() << errorPrefix << " has incorrect number of input edges";
        if (outputShapes.size() != 1)
            IE_THROW() << errorPrefix << " has incorrect number of output edges";
        isAxesSpecified = numInputs != 3;

        const auto &interpAttr = interp->get_attrs();

        const size_t dataRank = getInputShapeAtPort(DATA_ID).getRank();
        const auto &interpMode = interpAttr.mode;
        if (interpMode == ngInterpMode::nearest) {
            interpAttrs.mode = InterpolateMode::nearest;
        } else if (interpMode == ngInterpMode::linear) {
            if (dataRank < 5) {
                interpAttrs.mode = InterpolateMode::linear_onnx;
            } else {
                interpAttrs.mode = InterpolateMode::linear;
            }
        } else if (interpMode == ngInterpMode::linear_onnx) {
            interpAttrs.mode = InterpolateMode::linear_onnx;
        } else if (interpMode == ngInterpMode::cubic) {
            interpAttrs.mode = InterpolateMode::cubic;
        } else {
            IE_THROW() << errorPrefix << " has unsupported interpolate mode";
        }

        const auto &interpCoordTransMode = interpAttr.coordinate_transformation_mode;
        if (interpCoordTransMode == ngInterpCoordTransf::half_pixel) {
            interpAttrs.coordTransMode = InterpolateCoordTransMode::half_pixel;
        } else if (interpCoordTransMode == ngInterpCoordTransf::pytorch_half_pixel) {
            interpAttrs.coordTransMode = InterpolateCoordTransMode::pytorch_half_pixel;
        } else if (interpCoordTransMode == ngInterpCoordTransf::asymmetric) {
            interpAttrs.coordTransMode = InterpolateCoordTransMode::asymmetric;
        } else if (interpCoordTransMode == ngInterpCoordTransf::tf_half_pixel_for_nn) {
            interpAttrs.coordTransMode = InterpolateCoordTransMode::tf_half_pixel_for_nn;
        } else if (interpCoordTransMode == ngInterpCoordTransf::align_corners) {
            interpAttrs.coordTransMode = InterpolateCoordTransMode::align_corners;
        } else {
            IE_THROW() << errorPrefix << " has unsupported coordination transformation mode";
        }

        if (interpAttrs.mode == InterpolateMode::nearest) {
            const auto &interpNearestMode = interpAttr.nearest_mode;
            if (interpNearestMode == ngInterpNearMode::round_prefer_floor) {
                interpAttrs.nearestMode = InterpolateNearestMode::round_prefer_floor;
            } else if (interpNearestMode == ngInterpNearMode::round_prefer_ceil) {
                interpAttrs.nearestMode = InterpolateNearestMode::round_prefer_ceil;
            } else if (interpNearestMode == ngInterpNearMode::floor) {
                interpAttrs.nearestMode = InterpolateNearestMode::floor;
            } else if (interpNearestMode == ngInterpNearMode::ceil) {
                interpAttrs.nearestMode = InterpolateNearestMode::ceil;
            } else if (interpNearestMode == ngInterpNearMode::simple) {
                interpAttrs.nearestMode = InterpolateNearestMode::simple;
            } else {
                IE_THROW() << errorPrefix << " has unsupported nearest mode";
            }
        } else if (interpAttrs.mode == InterpolateMode::cubic) {
            interpAttrs.cubeCoeff = static_cast<float>(interpAttr.cube_coeff);
        }
        interpAttrs.antialias = interpAttr.antialias;

        const auto &interpShapeCalcMode = interpAttr.shape_calculation_mode;
        if (interpShapeCalcMode == ngInterpShapeCalcMode::scales) {
            shapeCalcMode = InterpolateShapeCalcMode::scales;
        } else if (interpShapeCalcMode == ngInterpShapeCalcMode::sizes) {
            shapeCalcMode = InterpolateShapeCalcMode::sizes;
        } else {
            IE_THROW() << errorPrefix << " has unsupported shape calculation mode";
        }

        if (interpAttr.pads_begin.empty()) {
            interpAttrs.padBegin.resize(dataRank, 0);
        } else {
            interpAttrs.padBegin.resize(interpAttr.pads_begin.size());
            for (size_t i = 0; i < interpAttr.pads_begin.size(); i++)
                interpAttrs.padBegin[i] = static_cast<int>(interpAttr.pads_begin[i]);
        }

        if (interpAttr.pads_end.empty()) {
            interpAttrs.padEnd.resize(dataRank, 0);
        } else {
            interpAttrs.padEnd.resize(interpAttr.pads_end.size());
            for (size_t i = 0; i < interpAttr.pads_end.size(); i++)
                interpAttrs.padEnd[i] = static_cast<int>(interpAttr.pads_end[i]);
        }

        const auto scalesNode = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(interp->get_input_node_shared_ptr(SCALES_ID));
        if (scalesNode) {
            scales = scalesNode->cast_vector<float>();
            isScaleConstant = true;
        }

        if (isAxesSpecified) {
            axes = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(interp->get_input_node_shared_ptr(AXES_ID))->cast_vector<int>();
        } else {
            axes.resize(dataRank);
            for (int i = 0; i < dataRank; i++) {
                axes[i] = i;
            }
        }
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void Interpolate::getSupportedDescriptors() {
    std::cout << "getSupportedDescriptors" << std::endl;

    if (getParentEdges().size() != 3 && getParentEdges().size() != 4)
        // data, target_shape, scale, axis(optional).
        IE_THROW() << errorPrefix << " has incorrect number of input edges";
    if (getChildEdges().empty())
        IE_THROW() << errorPrefix << " has incorrect number of output edges";

    int dataRank = getInputShapeAtPort(DATA_ID).getRank();

    // get pad
    for (int i = 0; i < interpAttrs.padBegin.size(); i++) {
        if (interpAttrs.padBegin[i] != 0) {
            hasPad = true;
            break;
        }
    }
    for (int i = 0; i < interpAttrs.padEnd.size(); i++) {
        if (interpAttrs.padEnd[i] != 0) {
            hasPad = true;
            break;
        }
    }
    //correct pad
    if (hasPad) {
        auto correctPad = [&](std::vector<int> pad, int rank) {
            int padLen = pad.size();
            if (padLen == rank) {
                return pad;
            }
            std::vector<int> result;
            if (padLen > rank) {
                result.insert(result.end(), pad.begin(), pad.begin() + rank);
            } else {
                result = pad;
                result.insert(result.end(), rank - padLen, 0);
            }
            return result;
        };

        interpAttrs.padBegin = correctPad(interpAttrs.padBegin, dataRank);
        interpAttrs.padEnd = correctPad(interpAttrs.padEnd, dataRank);
    }
}

void Interpolate::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision inputPrecision = getOriginalInputPrecisionAtPort(DATA_ID);
    if ((inputPrecision != Precision::I8) && (inputPrecision != Precision::U8) && (inputPrecision != Precision::BF16)) {
        inputPrecision = Precision::FP32;
    }
    if ((inputPrecision == Precision::BF16) && !mayiuse(avx512_core)) {
        inputPrecision = Precision::FP32;
    }
    Precision outputPrecision = inputPrecision;

    if (!fusedWith.empty()) {
        outputPrecision = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(DATA_ID);
    }

    if (!mayiuse(cpu::x64::sse41)) {
        inputPrecision = outputPrecision = Precision::FP32;
    }

    NodeConfig config;
    config.dynBatchSupport = false;
    if (isAxesSpecified) {
        config.inConfs.resize(4);
    } else {
        config.inConfs.resize(3);
    }
    config.outConfs.resize(1);

    auto targetShapeType = Precision::I32;
    auto scalesType = Precision::FP32;
    auto axesType = Precision::I32;

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    auto pushDesc = [&](LayoutType dataFormat, impl_desc_type implDetail) {
        config.inConfs[DATA_ID].setMemDesc(creatorsMap.at(dataFormat)->createSharedDesc(inputPrecision, getInputShapeAtPort(DATA_ID)));
        config.inConfs[TARGET_SHAPE_ID].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(targetShapeType, getInputShapeAtPort(TARGET_SHAPE_ID)));
        config.inConfs[SCALES_ID].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(scalesType, getInputShapeAtPort(SCALES_ID)));

        if (isAxesSpecified)
            config.inConfs[AXES_ID].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(axesType, getInputShapeAtPort(AXES_ID)));

        config.outConfs[0].setMemDesc(creatorsMap.at(dataFormat)->createSharedDesc(outputPrecision, getOutputShapeAtPort(0)));
#if defined(OPENVINO_ARCH_X86_64)
        supportedPrimitiveDescriptors.push_back({config, impl_type});
#else
        std::vector<MemoryDescPtr> srcMemoryDescs;
        for (int i = 0; i < config.inConfs.size(); i++) {
            srcMemoryDescs.push_back(config.inConfs[i].getMemDesc());
        }
        std::vector<MemoryDescPtr> dstMemoryDescs;
        for (int i = 0; i < config.outConfs.size(); i++) {
            dstMemoryDescs.push_back(config.outConfs[i].getMemDesc());
        }

        auto factory = std::make_shared<InterpolateExecutorFactory>(interpAttrs, srcMemoryDescs, dstMemoryDescs,
                                                                    std::make_shared<ExecutorContext>(context, getPrimitivesPriority()));
        supportedPrimitiveDescriptors.push_back({config, implDetail, factory});
#endif
    };

    const auto &dataMinDims = getInputShapeAtPort(DATA_ID).getMinDims();
    bool isBlkApplied = getInputShapeAtPort(DATA_ID).getRank() > 1 && dataMinDims[1] != Shape::UNDEFINED_DIM && dataMinDims[1] > 1;

    if (!mayiuse(cpu::x64::sse41) || interpAttrs.mode == InterpolateMode::linear) {
        pushDesc(LayoutType::ncsp, ref);
    } else {
        // blk and by_channel JIT kernel on sse41 or above machine
        if (getInputShapeAtPort(DATA_ID).getRank() == 4 || (getInputShapeAtPort(DATA_ID).getRank() == 5 && interpAttrs.mode != InterpolateMode::cubic)) {
            if (mayiuse(cpu::x64::avx512_core)) {
                pushDesc(LayoutType::nspc, jit_avx512);
                if (isBlkApplied)
                    pushDesc(LayoutType::nCsp16c, jit_avx512);
            } else if (mayiuse(cpu::x64::avx2)) {
                pushDesc(LayoutType::nspc, jit_avx2);
                if (isBlkApplied)
                    pushDesc(LayoutType::nCsp8c, jit_avx2);
            } else {
                pushDesc(LayoutType::nspc, jit_sse42);
                if (isBlkApplied)
                    pushDesc(LayoutType::nCsp8c, jit_sse42);
            }
        }

        // planar for 1.ref on machine without sse41(if no sse41, canFuse() is false). 2.JIT kernel for f32 && avx2(gather).(with fuse)
        if (mayiuse(cpu::x64::avx2) && inputPrecision == Precision::FP32) {
            pushDesc(LayoutType::ncsp, jit_avx2);
        }
    }
}

bool Interpolate::needShapeInfer() const {
    if (Node::inputShapesModified()) {
        return true;
    }
    if (shapeCalcMode == InterpolateShapeCalcMode::scales) {
        if (lastScales.empty()) {
            return true;
        }
        const float *scales = reinterpret_cast<const float *>(getParentEdgesAtPort(SCALES_ID)[0]->getMemory().GetPtr());
        for (size_t i = 0; i < lastScales.size(); i++) {
            if (lastScales[i] != scales[i]) {
                return true;
            }
        }
    } else {
        if (lastSizes.empty()) {
            return true;
        }
        const int32_t *sizes = reinterpret_cast<const int32_t *>(getParentEdgesAtPort(TARGET_SHAPE_ID)[0]->getMemory().GetPtr());
        for (size_t i = 0; i < lastSizes.size(); i++) {
            if (sizes[i] != lastSizes[i]) {
                return true;
            }
        }
    }
    return false;
}

void Interpolate::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);

    const size_t port = shapeCalcMode == InterpolateShapeCalcMode::sizes ? TARGET_SHAPE_ID : SCALES_ID;
    const auto &memory = getParentEdgesAtPort(port)[0]->getMemory();
    if (shapeCalcMode == InterpolateShapeCalcMode::scales) {
        const float *scales = reinterpret_cast<const float *>(memory.GetPtr());
        lastScales.assign(scales, scales + memory.getDesc().getShape().getElementsCount());
    } else {
        const int32_t *sizes = reinterpret_cast<const int32_t *>(memory.GetPtr());
        lastSizes.assign(sizes, sizes + memory.getDesc().getShape().getElementsCount());
    }
}

bool Interpolate::needPrepareParams() const {
    return (inputShapesModified() || lastOutputDims != getChildEdgesAtPort(0)[0]->getMemory().getStaticDims());
}

void Interpolate::prepareParams() {\
    if (!shapesDefined()) {
        IE_THROW() << "Can't prepare params for Interpolate node with name: " << getName() << ", because input/output dims aren't defined";
    }

    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(DATA_ID)->getMemoryPtr();
    auto& tsMemPtr = getParentEdgeAt(TARGET_SHAPE_ID)->getMemoryPtr();
    auto& scaleMemPtr = getParentEdgeAt(SCALES_ID)->getMemoryPtr();
    if (getParentEdges().size() > 3) {
        auto &axesMemPtr = getParentEdgeAt(AXES_ID)->getMemoryPtr();
        if (!axesMemPtr || !axesMemPtr->isAllocated())
            IE_THROW() << errorPrefix << " did not allocate axes memory";
    }
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        IE_THROW() << errorPrefix << " did not allocate destination memory";
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        IE_THROW() << errorPrefix << " did not allocate input memory";
    if (!tsMemPtr || !tsMemPtr->isAllocated())
        IE_THROW() << errorPrefix << " did not allocate target shape memory";
    if (!scaleMemPtr || !scaleMemPtr->isAllocated())
        IE_THROW() << errorPrefix << " did not allocate scales memory";
    const NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << errorPrefix << " did not set preferable primitive descriptor";

    const auto &srcDims = srcMemPtr->getStaticDims();
    const auto &dstDims = dstMemPtr->getStaticDims();

    if (!isScaleConstant) {
        const auto& scalesMem = getParentEdgesAtPort(SCALES_ID)[0]->getMemory();
        const float* scalesData = reinterpret_cast<const float *>(scalesMem.GetPtr());
        scales.assign(scalesData, scalesData + scalesMem.getStaticDims()[0]);
    }

    auto dataScales = getScales(getPaddedInputShape(srcDims, interpAttrs.padBegin, interpAttrs.padEnd), dstDims);
    if (getOutputShapeAtPort(0).getRank() > 2 && (dataScales[0] != 1.f || dataScales[1] != 1.f)) {
        IE_THROW() << "Interpolate layer only supports resize on spatial dimensions(depth, height and width)";
    }
    InterpolateKey key = {interpAttrs, srcDims, dstDims, dnnl::primitive_attr()};
    setPostOps(key.attr, dstDims);
#if defined(OPENVINO_ARCH_X86_64)
    auto buildExecutor = [&](const InterpolateKey& key) -> std::shared_ptr<InterpolateExecutor> {
        std::shared_ptr<InterpolateExecutor> executor;
        if ((key.nodeAttrs.mode == InterpolateMode::nearest || key.nodeAttrs.mode == InterpolateMode::linear_onnx ||
            key.nodeAttrs.mode == InterpolateMode::cubic) &&
            ((key.nodeAttrs.layout != InterpolateLayoutType::planar && mayiuse(cpu::x64::sse41)) ||
                (mayiuse(cpu::x64::avx2) && key.nodeAttrs.inPrc == Precision::FP32))) {
            executor = std::make_shared<InterpolateJitExecutor>(key.nodeAttrs,
                                                               key.srcDims,
                                                               key.dstDims,
                                                               key.nodeAttrs.dataScales,
                                                               key.attr);
        }
        return executor;
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, buildExecutor);
    execPtr = result.first;
#else
    std::vector<MemoryDescPtr> srcMemoryDescs;
    for (int i = 0; i < getOriginalInputsNumber(); i++) {
        srcMemoryDescs.push_back(getParentEdgeAt(i)->getMemoryPtr()->getDescPtr());
    }
    std::vector<MemoryDescPtr> dstMemoryDescs;
    for (int i = 0; i < getOriginalOutputsNumber(); i++) {
        dstMemoryDescs.push_back(getChildEdgeAt(i)->getMemoryPtr()->getDescPtr());
    }
    auto selectedPD = getSelectedPrimitiveDescriptor();
    execPtr = selectedPD->getExecutorFactoryAs<InterpolateExecutorFactory>()->makeExecutor(interpAttrs, srcMemoryDescs, dstMemoryDescs, key.attr);
    selectedPD->setImplementationType(execPtr->getImplType());
#endif
    lastOutputDims = dstDims;
}

void Interpolate::createPrimitive() {
    auto& srcMemPtr = getParentEdgeAt(DATA_ID)->getMemoryPtr();
    auto& dstMemPtr = getChildEdgesAtPort(0)[0]->getMemoryPtr();
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        IE_THROW() << errorPrefix << " did not allocate input memory";
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        IE_THROW() << errorPrefix << " did not allocate destination memory";

    if (dstMemPtr->getDesc().hasLayoutType(LayoutType::ncsp)) {
        interpAttrs.layout = InterpolateLayoutType::planar;
    } else if (dstMemPtr->getDesc().hasLayoutType(LayoutType::nCsp8c) ||
               dstMemPtr->getDesc().hasLayoutType(LayoutType::nCsp16c)) {
        interpAttrs.layout = InterpolateLayoutType::block;
    } else {
        interpAttrs.layout = InterpolateLayoutType::by_channel;
    }

    interpAttrs.inPrc = srcMemPtr->getDesc().getPrecision();
    interpAttrs.outPrc = dstMemPtr->getDesc().getPrecision();

    if (shapesDefined() && isExecutable()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

static inline float triangleCoeff(float x) {
    return (std::max)(0.0f, 1 - std::abs(x));
}

void Interpolate::setPostOps(dnnl::primitive_attr &attr, const VectorDims &dims) {
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
            eltwiseNode->appendPostOps(ops, dims, postOpsDataPtrs);
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

// get scales of data rank size
// if "scale" version: set scales with input scales, 1.f for other dims not in axis
// if "size" version: scales = shape[target] / shape[input].pad, 1.f for other dims not in axis
// scales is a required input, but should not use input scales when "size" case, which may added eps that lead to inaccurate result, recalculate scales instead.
std::vector<float> Interpolate::getScales(const VectorDims &srcDimPad, const VectorDims &dstDim) {
    const size_t dataRank = getInputShapeAtPort(DATA_ID).getRank();
    std::vector<float> fullScales(dataRank, 1.f);
    const size_t axesRank = axes.size();
    for (size_t i = 0; i < axesRank; i++) {
        int axis = axes[i];
        fullScales[axis] = (shapeCalcMode == InterpolateShapeCalcMode::scales) ? scales[i] :
                                                                                 static_cast<float>(dstDim[axis]) / static_cast<float>(srcDimPad[axis]);
    }
    return fullScales;
}

void Interpolate::execute(dnnl::stream strm) {
    if (!execPtr) {
        IE_THROW() << "Can't execute Interpolate node. Primitive didn't created";
    }
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(DATA_ID)->getMemoryPtr();

    uint8_t *dst_data = reinterpret_cast<uint8_t*>(dstMemPtr->GetPtr());
    const uint8_t *src_data_origin = reinterpret_cast<uint8_t*>(srcMemPtr->GetData());

    const auto &srcDim = srcMemPtr->getStaticDims();
    const auto &dstDim = dstMemPtr->getStaticDims();
    size_t dimSize = srcDim.size();
    auto srcDimPad = execPtr->getSrcDimPad5d();

    const auto srcDim5d = to5Dim(srcDim);
    const auto srcDimPad5d = to5Dim(srcDimPad);
    const auto dstDim5d = to5Dim(dstDim);
    const auto srcDataSize = srcMemPtr->getDesc().getPrecision().size();

    const uint8_t *src_data = nullptr;
    std::vector<uint8_t> srcPadded;
    if (hasPad) {
        int padB0 = (dimSize > 2) ? interpAttrs.padBegin[0] : 0;
        int padB1 = (dimSize > 2) ? interpAttrs.padBegin[1] : 0;
        int padB2 = (dimSize == 5) ? interpAttrs.padBegin[dimSize - 3] : 0;
        int padB3 = interpAttrs.padBegin[dimSize - 2];
        int padB4 = interpAttrs.padBegin[dimSize - 1];

        SizeVector inShapeBlock = getBlockND(srcDim5d);
        SizeVector inShapePadBlock = getBlockND(srcDimPad5d);

        if (interpAttrs.layout == InterpolateLayoutType::planar) {
            srcPadded.resize(inShapePadBlock[0] * srcDataSize, 0);
            uint8_t *src_data_pad = static_cast<uint8_t *>(&srcPadded[0]);
            parallel_for4d(srcDim5d[0], srcDim5d[1], srcDim5d[2], srcDim5d[3], [&](int n, int c, int d, int h) {
                const uint8_t *src = src_data_origin + (inShapeBlock[1] * n + inShapeBlock[2] * c + inShapeBlock[3] * d + inShapeBlock[4] * h) * srcDataSize;
                uint8_t *srcPad = src_data_pad + (inShapePadBlock[1] * (n + padB0) + inShapePadBlock[2] * (c + padB1) +
                               inShapePadBlock[3] * (d + padB2) + inShapePadBlock[4] * (h + padB3) + padB4) * srcDataSize;
                cpu_memcpy(srcPad, src, srcDim5d[4] * srcDataSize);
            });
            src_data = src_data_pad;
        } else if (interpAttrs.layout == InterpolateLayoutType::by_channel) {
            srcPadded.resize(inShapePadBlock[0] * srcDataSize, 0);
            uint8_t *src_data_pad = static_cast<uint8_t *>(&srcPadded[0]);
            parallel_for4d(srcDim5d[0], srcDim5d[2], srcDim5d[3], srcDim5d[4], [&](int n, int d, int h, int w) {
                const uint8_t *src = src_data_origin + (inShapeBlock[1] * n +
                                (inShapeBlock[3] * d + inShapeBlock[4] * h + inShapeBlock[5] * w) * srcDim5d[1]) * srcDataSize;
                uint8_t *srcPad = src_data_pad + (inShapePadBlock[1] * (n + padB0) + (inShapePadBlock[3] * (d + padB2) +
                                inShapePadBlock[4] * (h + padB3) + inShapePadBlock[5] * (w + padB4)) * srcDimPad5d[1] + padB1) * srcDataSize;
                cpu_memcpy(srcPad, src, srcDim5d[1] * srcDataSize);
            });
            src_data = src_data_pad;
        } else if (interpAttrs.layout == InterpolateLayoutType::block) {
            size_t blkSize = mayiuse(cpu::x64::avx512_core) ? 16 : 8;
            size_t CB = div_up(srcDimPad5d[1], blkSize);
            size_t eltsTotal = srcDimPad5d[0] * CB * srcDimPad5d[2] * srcDimPad5d[3] * srcDimPad5d[4] * blkSize;
            srcPadded.resize(eltsTotal * srcDataSize, 0x0);
            uint8_t *src_data_pad = static_cast<uint8_t *>(&srcPadded[0]);
            if ((srcDim5d[0] != srcDimPad5d[0]) || (srcDim5d[1] != srcDimPad5d[1])) {
                IE_THROW() << "Interpolate layer with name '" << getName() <<
                "' does not support padding on batch and channel dimensions";
            }
            parallel_for5d(srcDim5d[0], CB, srcDim5d[2], srcDim5d[3], srcDim5d[4], [&](int n, int cb, int d, int h, int w) {
                const uint8_t *src = src_data_origin + (n * CB * srcDim5d[2] * srcDim5d[3] * srcDim5d[4] * blkSize) * srcDataSize
                                               + (cb * srcDim5d[2] * srcDim5d[3] * srcDim5d[4] * blkSize) * srcDataSize
                                               + (d * srcDim5d[3] * srcDim5d[4] * blkSize) * srcDataSize
                                               + (h * srcDim5d[4] * blkSize) * srcDataSize
                                               + (w * blkSize) * srcDataSize;
                uint8_t *srcPad = src_data_pad + (n * CB * srcDimPad5d[2] * srcDimPad5d[3] * srcDimPad5d[4] * blkSize) * srcDataSize
                                               + (cb * srcDimPad5d[2] * srcDimPad5d[3] * srcDimPad5d[4] * blkSize) * srcDataSize
                                               + ((d + padB2) * srcDimPad5d[3] * srcDimPad5d[4] * blkSize) * srcDataSize
                                               + ((h + padB3) * srcDimPad5d[4] * blkSize) * srcDataSize
                                               + ((w + padB4) * blkSize) * srcDataSize;
                cpu_memcpy(srcPad, src, blkSize * srcDataSize);
            });
            src_data = src_data_pad;
        }
    } else {
        src_data = src_data_origin;
    }

    execPtr->exec(src_data, dst_data, postOpsDataPtrs.data());
}

bool Interpolate::canFuse(const NodePtr& node) const {
    if (!mayiuse(cpu::x64::sse41) || interpAttrs.mode == InterpolateMode::linear) {
        return false;
    }

    return canFuseSimpleOperation(node);
}

bool Interpolate::created() const {
    return getType() == Type::Interpolate;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
