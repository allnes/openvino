// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interpolate.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "cpu_types.h"
// Removed fusing: no Eltwise/FQ includes needed
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/node_config.h"
#include "openvino/core/enum_names.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/interpolate.hpp"
#include "shape_inference/shape_inference.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"
#include "utils/ngraph_utils.hpp"
namespace ov::intel_cpu::node {

// Removed legacy InterpolateKey cache (new executor path manages its own caches)

using ngInterpMode = ov::op::v4::Interpolate::InterpolateMode;
using ngInterpCoordTransf = ov::op::v4::Interpolate::CoordinateTransformMode;
using ngInterpNearMode = ov::op::v4::Interpolate::NearestMode;
using ngInterpShapeCalcMode = ov::op::v4::Interpolate::ShapeCalcMode;

bool Interpolate::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (const auto interp = ov::as_type_ptr<const ov::op::v4::Interpolate>(op)) {
            const auto& interpAttr = interp->get_attrs();
            const auto& interpMode = interpAttr.mode;
            if (none_of(interpMode,
                        ngInterpMode::NEAREST,
                        ngInterpMode::LINEAR,
                        ngInterpMode::LINEAR_ONNX,
                        ngInterpMode::CUBIC)) {
                errorMessage = "Interpolate-4 does not support interpolate mode: " + ov::as_string(interpMode);
                return false;
            }

            const auto& interpCoordTransMode = interpAttr.coordinate_transformation_mode;
            if (none_of(interpCoordTransMode,
                        ngInterpCoordTransf::HALF_PIXEL,
                        ngInterpCoordTransf::PYTORCH_HALF_PIXEL,
                        ngInterpCoordTransf::ASYMMETRIC,
                        ngInterpCoordTransf::TF_HALF_PIXEL_FOR_NN,
                        ngInterpCoordTransf::ALIGN_CORNERS)) {
                errorMessage = "Interpolate-4 does not support coordinate transformation mode: " +
                               ov::as_string(interpCoordTransMode);
                return false;
            }

            if (interpMode == ngInterpMode::NEAREST) {
                const auto& interpNearestMode = interpAttr.nearest_mode;
                if (none_of(interpNearestMode,
                            ngInterpNearMode::ROUND_PREFER_FLOOR,
                            ngInterpNearMode::ROUND_PREFER_CEIL,
                            ngInterpNearMode::FLOOR,
                            ngInterpNearMode::CEIL,
                            ngInterpNearMode::SIMPLE)) {
                    errorMessage =
                        "Interpolate-4 does not support nearest round mode: " + ov::as_string(interpNearestMode);
                    return false;
                }
            }

            const auto& interpShapeCalcMode = interpAttr.shape_calculation_mode;
            if (none_of(interpShapeCalcMode, ngInterpShapeCalcMode::SCALES, ngInterpShapeCalcMode::SIZES)) {
                errorMessage =
                    "Interpolate-4 does not support shape_calculation_mode: " + ov::as_string(interpShapeCalcMode);
                return false;
            }

            const size_t dataRank = interp->get_input_partial_shape(DATA_ID).rank().get_length();
            if (dataRank < 1 || dataRank > 5) {
                errorMessage = "Interpolate-4 does not support input tensor of rank : " + std::to_string(dataRank);
                return false;
            }

            if (dataRank == 5 && interpMode == ngInterpMode::CUBIC) {
                errorMessage = "Interpolate-4 doesn't support input tensor with rank: " + std::to_string(dataRank) +
                               " for 'cubic' mode ";
                return false;
            }

            if (!isDynamicNgraphNode(op) && interpShapeCalcMode == ngInterpShapeCalcMode::SCALES &&
                !ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(SCALES_ID))) {
                errorMessage = "Only const 'scales' input is supported for static shapes in Interpolate-4";
                return false;
            }

            if (interp->get_input_size() > 3 &&
                ov::as_type_ptr<const ov::op::v0::Constant>(interp->get_input_node_shared_ptr(AXES_ID)) == nullptr) {
                errorMessage = "Only const 'axes' input is supported in Interpolate-4";
                return false;
            }
        } else if (const auto interp = ov::as_type_ptr<const ov::op::v11::Interpolate>(op)) {
            const auto& interpAttr = interp->get_attrs();
            const auto& interpMode = interpAttr.mode;
            if (none_of(interpMode, ngInterpMode::BILINEAR_PILLOW, ngInterpMode::BICUBIC_PILLOW)) {
                errorMessage = "Interpolate-11 does not support interpolate mode: " + ov::as_string(interpMode);
                return false;
            }
            const auto& interpShapeCalcMode = interpAttr.shape_calculation_mode;
            if (none_of(interpShapeCalcMode, ngInterpShapeCalcMode::SCALES, ngInterpShapeCalcMode::SIZES)) {
                errorMessage =
                    "Interpolate-11 does not support shape_calculation_mode: " + ov::as_string(interpShapeCalcMode);
                return false;
            }
            const size_t dataRank = interp->get_input_partial_shape(DATA_ID).rank().get_length();
            if (dataRank < 2 || dataRank > 4) {
                // pillow only resize on H and W. resize on D(depth) is not defined.
                errorMessage = "Interpolate-11 does not support input tensor of rank : " + std::to_string(dataRank);
                return false;
            }
            if (!isDynamicNgraphNode(op) &&
                !ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(SIZE_OR_SCALE_ID_V11))) {
                errorMessage = "Only const 'scales_or_sizes' input is supported for static shapes in Interpolate-11";
                return false;
            }
            if (interp->get_input_size() > 2 && ov::as_type_ptr<const ov::op::v0::Constant>(
                                                    interp->get_input_node_shared_ptr(AXES_ID_V11)) == nullptr) {
                errorMessage = "Only const 'axes' input is supported in Interpolate-11";
                return false;
            }
        } else {
            errorMessage = "Only v4 and v11 interpolate operation are supported";
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
    explicit InterpolateShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(std::move(op)) {}
    [[nodiscard]] ShapeInferPtr makeShapeInfer() const override {
        if (auto interp4 = ov::as_type_ptr<ov::op::v4::Interpolate>(m_op)) {
            const auto& attr = interp4->get_attrs();
            const auto is_supported_mode = (attr.shape_calculation_mode == ngInterpShapeCalcMode::SCALES) ||
                                           (attr.shape_calculation_mode == ngInterpShapeCalcMode::SIZES);
            OPENVINO_ASSERT(is_supported_mode, "Unsupported interpolate shape calculation mode");
            return make_shape_inference(m_op);
        }
        if (auto interp11 = ov::as_type_ptr<ov::op::v11::Interpolate>(m_op)) {
            return make_shape_inference(m_op);
        }
        OPENVINO_THROW("Shape infer factory cannot be created for ",
                       m_op->get_type_name(),
                       " node with name: ",
                       m_op->get_friendly_name(),
                       ", only versions 4 and 11 are supported.");
    }

private:
    std::shared_ptr<ov::Node> m_op;
};
}  // namespace

Interpolate::Interpolate(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, InterpolateShapeInferFactory(op)) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        const auto& inputDataShape = getInputShapeAtPort(DATA_ID);
        dataRank = inputDataShape.getRank();
        if (const auto interp = ov::as_type_ptr<const ov::op::v4::Interpolate>(op)) {
            is_version11 = false;
            const auto numInputs = inputShapes.size();
            CPU_NODE_ASSERT(numInputs == 3 || numInputs == 4, "has incorrect number of input edges");
            CPU_NODE_ASSERT(outputShapes.size() == 1, "has incorrect number of output edges");
            isAxesSpecified = numInputs != 3;

            const auto& interpAttr = interp->get_attrs();

            const auto& interpMode = interpAttr.mode;
            if (interpMode == ngInterpMode::NEAREST) {
                interpAttrs.mode = InterpolateMode::nearest;
            } else if (interpMode == ngInterpMode::LINEAR) {
                if (dataRank < 5) {
                    interpAttrs.mode = InterpolateMode::linear_onnx;
                } else {
                    interpAttrs.mode = InterpolateMode::linear;
                }
            } else if (interpMode == ngInterpMode::LINEAR_ONNX) {
                interpAttrs.mode = InterpolateMode::linear_onnx;
            } else if (interpMode == ngInterpMode::CUBIC) {
                interpAttrs.mode = InterpolateMode::cubic;
            } else {
                CPU_NODE_THROW("has unsupported interpolate mode");
            }

            const auto& interpCoordTransMode = interpAttr.coordinate_transformation_mode;
            if (interpCoordTransMode == ngInterpCoordTransf::HALF_PIXEL) {
                interpAttrs.coordTransMode = InterpolateCoordTransMode::half_pixel;
            } else if (interpCoordTransMode == ngInterpCoordTransf::PYTORCH_HALF_PIXEL) {
                interpAttrs.coordTransMode = InterpolateCoordTransMode::pytorch_half_pixel;
            } else if (interpCoordTransMode == ngInterpCoordTransf::ASYMMETRIC) {
                interpAttrs.coordTransMode = InterpolateCoordTransMode::asymmetric;
            } else if (interpCoordTransMode == ngInterpCoordTransf::TF_HALF_PIXEL_FOR_NN) {
                interpAttrs.coordTransMode = InterpolateCoordTransMode::tf_half_pixel_for_nn;
            } else if (interpCoordTransMode == ngInterpCoordTransf::ALIGN_CORNERS) {
                interpAttrs.coordTransMode = InterpolateCoordTransMode::align_corners;
            } else {
                CPU_NODE_THROW("has unsupported coordination transformation mode");
            }

            if (interpAttrs.mode == InterpolateMode::nearest) {
                const auto& interpNearestMode = interpAttr.nearest_mode;
                if (interpNearestMode == ngInterpNearMode::ROUND_PREFER_FLOOR) {
                    interpAttrs.nearestMode = InterpolateNearestMode::round_prefer_floor;
                } else if (interpNearestMode == ngInterpNearMode::ROUND_PREFER_CEIL) {
                    interpAttrs.nearestMode = InterpolateNearestMode::round_prefer_ceil;
                } else if (interpNearestMode == ngInterpNearMode::FLOOR) {
                    interpAttrs.nearestMode = InterpolateNearestMode::floor;
                } else if (interpNearestMode == ngInterpNearMode::CEIL) {
                    interpAttrs.nearestMode = InterpolateNearestMode::ceil;
                } else if (interpNearestMode == ngInterpNearMode::SIMPLE) {
                    interpAttrs.nearestMode = InterpolateNearestMode::simple;
                } else {
                    CPU_NODE_THROW("has unsupported nearest mode");
                }
            } else if (interpAttrs.mode == InterpolateMode::cubic) {
                interpAttrs.cubeCoeff = static_cast<float>(interpAttr.cube_coeff);
            }
            interpAttrs.antialias = interpAttr.antialias;

            const auto& interpShapeCalcMode = interpAttr.shape_calculation_mode;
            if (interpShapeCalcMode == ngInterpShapeCalcMode::SCALES) {
                interpAttrs.shapeCalcMode = InterpolateShapeCalcMode::scales;
            } else if (interpShapeCalcMode == ngInterpShapeCalcMode::SIZES) {
                interpAttrs.shapeCalcMode = InterpolateShapeCalcMode::sizes;
            } else {
                CPU_NODE_THROW("has unsupported shape calculation mode");
            }

            if (interpAttr.pads_begin.empty()) {
                interpAttrs.padBegin.resize(dataRank, 0);
            } else {
                interpAttrs.padBegin.resize(interpAttr.pads_begin.size());
                for (size_t i = 0; i < interpAttr.pads_begin.size(); i++) {
                    interpAttrs.padBegin[i] = static_cast<int>(interpAttr.pads_begin[i]);
                }
            }

            if (interpAttr.pads_end.empty()) {
                interpAttrs.padEnd.resize(dataRank, 0);
            } else {
                interpAttrs.padEnd.resize(interpAttr.pads_end.size());
                for (size_t i = 0; i < interpAttr.pads_end.size(); i++) {
                    interpAttrs.padEnd[i] = static_cast<int>(interpAttr.pads_end[i]);
                }
            }

            const auto scalesNode =
                ov::as_type_ptr<const ov::op::v0::Constant>(interp->get_input_node_shared_ptr(SCALES_ID));
            if (scalesNode) {
                scales = scalesNode->cast_vector<float>();
                isScaleConstant = true;
            }

            if (isAxesSpecified) {
                axes = ov::as_type_ptr<const ov::op::v0::Constant>(interp->get_input_node_shared_ptr(AXES_ID))
                           ->cast_vector<int>();
            } else {
                axes.resize(dataRank);
                for (int i = 0; i < static_cast<int>(dataRank); i++) {
                    axes[i] = i;
                }
            }
        } else if (const auto interp = ov::as_type_ptr<const ov::op::v11::Interpolate>(op)) {
            is_version11 = true;
            const auto numInputs = inputShapes.size();
            CPU_NODE_ASSERT(numInputs == 2 || numInputs == 3, "has incorrect number of input edges");
            CPU_NODE_ASSERT(outputShapes.size() == 1, "has incorrect number of output edges");
            isAxesSpecified = numInputs != 2;

            const auto& interpAttr = interp->get_attrs();
            const auto& interpMode = interpAttr.mode;
            if (interpMode == ngInterpMode::BILINEAR_PILLOW) {
                interpAttrs.mode = InterpolateMode::bilinear_pillow;
            } else if (interpMode == ngInterpMode::BICUBIC_PILLOW) {
                interpAttrs.mode = InterpolateMode::bicubic_pillow;
                interpAttrs.cubeCoeff = static_cast<float>(interpAttr.cube_coeff);  // fixed to be -0.5
            } else {
                CPU_NODE_THROW("has unsupported interpolate mode");
            }

            // pillow use fixed tf_half_pixel_for_nn style mode for coodinate transformation
            interpAttrs.coordTransMode = InterpolateCoordTransMode::tf_half_pixel_for_nn;
            interpAttrs.antialias = interpAttr.antialias;

            const auto& interpShapeCalcMode = interpAttr.shape_calculation_mode;
            if (interpShapeCalcMode == ngInterpShapeCalcMode::SCALES) {
                interpAttrs.shapeCalcMode = InterpolateShapeCalcMode::scales;
                const auto scalesNode = ov::as_type_ptr<const ov::op::v0::Constant>(
                    interp->get_input_node_shared_ptr(SIZE_OR_SCALE_ID_V11));
                if (scalesNode) {
                    scales = scalesNode->cast_vector<float>();
                    isScaleConstant = true;
                }
            } else if (interpShapeCalcMode == ngInterpShapeCalcMode::SIZES) {
                interpAttrs.shapeCalcMode = InterpolateShapeCalcMode::sizes;
            } else {
                CPU_NODE_THROW("has unsupported shape calculation mode");
            }

            if (interpAttr.pads_begin.empty()) {
                interpAttrs.padBegin.resize(dataRank, 0);
            } else {
                interpAttrs.padBegin.resize(interpAttr.pads_begin.size());
                for (size_t i = 0; i < interpAttr.pads_begin.size(); i++) {
                    interpAttrs.padBegin[i] = static_cast<int>(interpAttr.pads_begin[i]);
                }
            }

            if (interpAttr.pads_end.empty()) {
                interpAttrs.padEnd.resize(dataRank, 0);
            } else {
                interpAttrs.padEnd.resize(interpAttr.pads_end.size());
                for (size_t i = 0; i < interpAttr.pads_end.size(); i++) {
                    interpAttrs.padEnd[i] = static_cast<int>(interpAttr.pads_end[i]);
                }
            }

            if (isAxesSpecified) {
                axes = ov::as_type_ptr<const ov::op::v0::Constant>(interp->get_input_node_shared_ptr(AXES_ID_V11))
                           ->cast_vector<int>();
                if ((interpAttrs.mode == InterpolateMode::bilinear_pillow ||
                     interpAttrs.mode == InterpolateMode::bicubic_pillow) &&
                    dataRank == 4 && axes.size() == 2 && axes[0] == 1 && axes[1] == 2) {
                    interpAttrs.NCHWAsNHWC = true;
                }
            } else {
                axes.resize(dataRank);
                for (int i = 0; i < static_cast<int>(dataRank); i++) {
                    axes[i] = i;
                }
            }
        }
    } else {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto has_non_zero = [](const std::vector<int>& pads) {
        return std::any_of(pads.begin(), pads.end(), [](int value) {
            return value != 0;
        });
    };
    hasPad = has_non_zero(interpAttrs.padBegin) || has_non_zero(interpAttrs.padEnd);
}

void Interpolate::getSupportedDescriptors() {
    // v4: data, target_shape, scale, axis(optional).
    // v11: data, size_or_scale, axis(optional)
    CPU_NODE_ASSERT(getParentEdges().size() == 2 || getParentEdges().size() == 3 || getParentEdges().size() == 4,
                    "has incorrect number of input edges");
    CPU_NODE_ASSERT(!getChildEdges().empty(), "has incorrect number of output edges");

    // get pad
    for (int i : interpAttrs.padBegin) {
        if (i != 0) {
            hasPad = true;
            break;
        }
    }
    for (int i : interpAttrs.padEnd) {
        if (i != 0) {
            hasPad = true;
            break;
        }
    }
    // correct pad
    if (hasPad) {
        interpAttrs.NCHWAsNHWC = false;
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
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    auto inputPrecision = getOriginalInputPrecisionAtPort(DATA_ID);
    if (inputPrecision == ov::element::dynamic) {
        inputPrecision = ov::element::f32;
    }
    auto outputPrecision = getOriginalOutputPrecisionAtPort(0);
    if (outputPrecision == ov::element::dynamic) {
        outputPrecision = inputPrecision;
    }

    const auto targetShapeType = ov::element::i32;
    const auto scalesType = ov::element::f32;
    const auto axesType = ov::element::i32;

    const auto& creators = BlockedDescCreator::getCommonCreators();
    std::vector<MemoryDescArgs> seeds;
    auto srcDescNCSP = creators.at(LayoutType::ncsp)->createSharedDesc(inputPrecision, getInputShapeAtPort(DATA_ID));
    auto dstDescNCSP = creators.at(LayoutType::ncsp)->createSharedDesc(outputPrecision, getOutputShapeAtPort(0));
    seeds.push_back(MemoryDescArgs{{ARG_SRC, srcDescNCSP}, {ARG_DST, dstDescNCSP}});
    auto srcDescNSPC = creators.at(LayoutType::nspc)->createSharedDesc(inputPrecision, getInputShapeAtPort(DATA_ID));
    auto dstDescNSPC = creators.at(LayoutType::nspc)->createSharedDesc(outputPrecision, getOutputShapeAtPort(0));
    seeds.push_back(MemoryDescArgs{{ARG_SRC, srcDescNSPC}, {ARG_DST, dstDescNSPC}});

    auto executionContext = std::make_shared<ExecutorContext>(context, getImplPriority());
    std::vector<MemoryDescArgs> nodeDescriptorsList;
    for (const auto& seed : seeds) {
        auto tempFactory = std::make_shared<ExecutorFactory<InterpolateAttrs>>(interpAttrs, executionContext, seed);
        auto proposed = tempFactory->getProperMemoryDescriptors(seed);
        if (proposed.empty()) {
            nodeDescriptorsList.push_back(seed);
        } else {
            nodeDescriptorsList.insert(nodeDescriptorsList.end(), proposed.begin(), proposed.end());
        }
    }

    for (const auto& nodeDescriptors : nodeDescriptorsList) {
        NodeConfig config;
        if (is_version11) {
            config.inConfs.resize(isAxesSpecified ? 3 : 2);
        } else {
            config.inConfs.resize(isAxesSpecified ? 4 : 3);
        }

        config.inConfs[DATA_ID].setMemDesc(nodeDescriptors.at(ARG_SRC));

        const auto plainDesc = creators.at(LayoutType::ncsp);
        if (is_version11) {
            if (interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::sizes) {
                config.inConfs[SIZE_OR_SCALE_ID_V11].setMemDesc(
                    plainDesc->createSharedDesc(targetShapeType, getInputShapeAtPort(SIZE_OR_SCALE_ID_V11)));
            } else {
                config.inConfs[SIZE_OR_SCALE_ID_V11].setMemDesc(
                    plainDesc->createSharedDesc(scalesType, getInputShapeAtPort(SIZE_OR_SCALE_ID_V11)));
            }
            if (isAxesSpecified) {
                config.inConfs[AXES_ID_V11].setMemDesc(
                    plainDesc->createSharedDesc(axesType, getInputShapeAtPort(AXES_ID_V11)));
            }
        } else {
            config.inConfs[TARGET_SHAPE_ID].setMemDesc(
                plainDesc->createSharedDesc(targetShapeType, getInputShapeAtPort(TARGET_SHAPE_ID)));
            config.inConfs[get_scale_id()].setMemDesc(
                plainDesc->createSharedDesc(scalesType, getInputShapeAtPort(get_scale_id())));
            if (isAxesSpecified) {
                config.inConfs[get_axis_id()].setMemDesc(
                    plainDesc->createSharedDesc(axesType, getInputShapeAtPort(get_axis_id())));
            }
        }

        config.outConfs.resize(1);
        config.outConfs[0].setMemDesc(nodeDescriptors.at(ARG_DST));

        const bool pillowMode = any_of(interpAttrs.mode,
                                       InterpolateMode::bilinear_pillow,
                                       InterpolateMode::bicubic_pillow);
        auto implType = pillowMode ? impl_desc_type::ref : impl_desc_type::undef;
        supportedPrimitiveDescriptors.emplace_back(config, implType);
    }

    // Legacy pushDesc path removed
}

bool Interpolate::needShapeInfer() const {
    if (Node::inputShapesModified()) {
        return true;
    }
    if (interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::scales) {
        if (lastScales.empty()) {
            return true;
        }
        const auto* scales = getSrcDataAtPortAs<const float>(get_scale_id());
        for (size_t i = 0; i < lastScales.size(); i++) {
            if (lastScales[i] != scales[i]) {
                return true;
            }
        }
    } else {
        if (lastSizes.empty()) {
            return true;
        }
        const auto* sizes = getSrcDataAtPortAs<const int32_t>(TARGET_SHAPE_ID);
        for (size_t i = 0; i < lastSizes.size(); i++) {
            if (sizes[i] != lastSizes[i]) {
                return true;
            }
        }
    }
    return false;
}

void Interpolate::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);

    const size_t port = interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::sizes ? TARGET_SHAPE_ID : get_scale_id();
    const auto& memory = getParentEdgeAt(port)->getMemory();
    if (interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::scales) {
        const auto* scales = memory.getDataAs<const float>();
        lastScales.assign(scales, scales + memory.getDesc().getShape().getElementsCount());
    } else {
        const auto* sizes = memory.getDataAs<const int32_t>();
        lastSizes.assign(sizes, sizes + memory.getDesc().getShape().getElementsCount());
    }
}

bool Interpolate::needPrepareParams() const {
    return (inputShapesModified() || lastOutputDims != getChildEdgeAt(0)->getMemory().getStaticDims());
}

inline int Interpolate::get_scale_id() const {
    if (is_version11) {
        return SIZE_OR_SCALE_ID_V11;
    }
    return SCALES_ID;
}
inline int Interpolate::get_axis_id() const {
    if (is_version11) {
        return AXES_ID_V11;
    }
    return AXES_ID;
}

void Interpolate::prepareParams() {
    CPU_NODE_ASSERT(shapesDefined(), "input/output dims aren't defined");

    auto dstMemPtr = getDstMemoryAtPort(0);
    CPU_NODE_ASSERT(dstMemPtr && dstMemPtr->isDefined(), "has undefined destination memory");

    auto srcMemPtr = getSrcMemoryAtPort(DATA_ID);
    CPU_NODE_ASSERT(srcMemPtr && srcMemPtr->isDefined(), "has undefined input memory");

    if (interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::sizes) {
        auto tsMemPtr = getSrcMemoryAtPort(TARGET_SHAPE_ID);
        CPU_NODE_ASSERT(tsMemPtr && tsMemPtr->isDefined(), "has undefined target shape memory");
    } else {
        auto scaleMemPtr = getSrcMemoryAtPort(get_scale_id());
        CPU_NODE_ASSERT(scaleMemPtr && scaleMemPtr->isDefined(), "has undefined scales memory");
    }

    if (isAxesSpecified) {
        auto axesMemPtr = getSrcMemoryAtPort(get_axis_id());
        CPU_NODE_ASSERT(axesMemPtr && axesMemPtr->isDefined(), "has undefined axes memory");
    }

    const NodeDesc* selected_pd = getSelectedPrimitiveDescriptor();
    CPU_NODE_ASSERT(selected_pd, "did not set preferable primitive descriptor");

    const auto& srcDimsOrign = srcMemPtr->getStaticDims();
    const auto& dstDimsOrign = dstMemPtr->getStaticDims();
    VectorDims srcDims = srcDimsOrign;
    VectorDims dstDims = dstDimsOrign;

    if (interpAttrs.NCHWAsNHWC && srcMemPtr->getDesc().hasLayoutType(LayoutType::ncsp)) {
        auto logicalShapeAlign = [](VectorDims& dims) {
            if (dims.size() == 4) {
                std::swap(dims[1], dims[2]);
                std::swap(dims[1], dims[3]);
            }
        };
        logicalShapeAlign(srcDims);
        logicalShapeAlign(dstDims);
        interpAttrs.layout = InterpolateLayoutType::by_channel;
    } else if (interpAttrs.NCHWAsNHWC) {
        interpAttrs.layout = InterpolateLayoutType::by_channel;
    }

    if (interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::scales) {
        if (!isScaleConstant) {
            const auto& scalesMem = getParentEdgeAt(get_scale_id())->getMemory();
            const auto* scalesData = scalesMem.getDataAs<const float>();
            scales.assign(scalesData, scalesData + scalesMem.getStaticDims()[0]);
        }
    }

    const auto paddedDims = getPaddedInputShape(srcDims, interpAttrs.padBegin, interpAttrs.padEnd);
    std::vector<float> dataScales = getScales(srcDims, paddedDims, dstDims);
    if (std::getenv("OV_DEBUG_INTERP_PAD")) {
        std::fprintf(stderr,
                     "[InterpolateNode] paddedDims(H,W)=(%zu,%zu) dstDims(H,W)=(%zu,%zu) scale(H)=%f scale(W)=%f "
                     "hasPad=%d\n",
                     paddedDims[dataRank - 2],
                     paddedDims[dataRank - 1],
                     dstDims[dataRank - 2],
                     dstDims[dataRank - 1],
                     dataScales[dataRank - 2],
                     dataScales[dataRank - 1],
                     hasPad ? 1 : 0);
    }

    // New executor-factory path (ACL/Reference backends), no oneDNN.
    // Create factory for all modes.
    {
        // Fill attrs for factory
        interpAttrs.dataScales = dataScales;  // pass original rank-aligned scales; executor will normalize to 5D
        interpAttrs.hasPad = hasPad;

        // Determine layout based on selected PD or current memory
        if (const NodeDesc* selectedPD = getSelectedPrimitiveDescriptor()) {
            const auto& cfg = selectedPD->getConfig();
            const auto& dataDesc = cfg.inConfs[DATA_ID].getMemDesc();
            if (interpAttrs.NCHWAsNHWC) {
                // preserve channel-last computation semantics for pillow NCHWAsNHWC
                interpAttrs.layout = InterpolateLayoutType::by_channel;
            } else if (dataDesc->hasLayoutType(LayoutType::nspc)) {
                interpAttrs.layout = InterpolateLayoutType::by_channel;
            } else if (dataDesc->hasLayoutType(LayoutType::ncsp)) {
                interpAttrs.layout = InterpolateLayoutType::planar;
            } else if (dataDesc->hasLayoutType(LayoutType::nCsp8c) || dataDesc->hasLayoutType(LayoutType::nCsp16c)) {
                interpAttrs.layout = InterpolateLayoutType::block;
            }
        } else {
            const auto& dataDesc = getSrcMemoryAtPort(DATA_ID)->getDesc();
            if (interpAttrs.NCHWAsNHWC) {
                interpAttrs.layout = InterpolateLayoutType::by_channel;
            } else if (dataDesc.hasLayoutType(LayoutType::nspc)) {
                interpAttrs.layout = InterpolateLayoutType::by_channel;
            } else if (dataDesc.hasLayoutType(LayoutType::ncsp)) {
                interpAttrs.layout = InterpolateLayoutType::planar;
            }
        }

        // Set input/output precisions
        interpAttrs.inPrc = getSrcMemoryAtPort(DATA_ID)->getDesc().getPrecision();
        interpAttrs.outPrc = getDstMemoryAtPort(0)->getDesc().getPrecision();

        // Prepare memory args and create executor
        m_memory[ARG_SRC] = getSrcMemoryAtPort(DATA_ID);
        m_memory[ARG_DST] = getDstMemoryAtPort(0);

        MemoryDescArgs execDescs{{ARG_SRC, m_memory[ARG_SRC]->getDescPtr()}, {ARG_DST, m_memory[ARG_DST]->getDescPtr()}};
        auto executionContext = std::make_shared<ExecutorContext>(context, getImplPriority());
        m_factory = std::make_shared<ExecutorFactory<InterpolateAttrs>>(interpAttrs, executionContext, execDescs);
        if (!m_executor) m_executor = m_factory->make(m_memory, false);
        if (m_executor) m_executor->update(m_memory);
    }
    // Legacy executor is not created in the new architecture

    lastOutputDims = dstDimsOrign;
}

void Interpolate::createPrimitive() {
    auto srcMemPtr = getSrcMemoryAtPort(DATA_ID);
    auto dstMemPtr = getDstMemoryAtPort(0);
    CPU_NODE_ASSERT(srcMemPtr, "has null input memory");
    CPU_NODE_ASSERT(dstMemPtr, "has null destination memory");

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
        if (needPrepareParams()) {
            prepareParams();
        }
        updateLastInputDims();
    }

    // Bind memory arguments for executor-factory path
    m_memory[ARG_SRC] = getSrcMemoryAtPort(DATA_ID);
    m_memory[ARG_DST] = getDstMemoryAtPort(0);

}

// No post-ops are applied for Interpolate in the new executor architecture.

VectorDims Interpolate::getPaddedInputShape(const VectorDims& srcDims,
                                            const std::vector<int>& padBegin,
                                            const std::vector<int>& padEnd) {
    VectorDims paddedShape;
    int dataRank = srcDims.size();
    for (int i = 0; i < dataRank; i++) {
        paddedShape.push_back(srcDims[i] + padBegin[i] + padEnd[i]);
    }
    return paddedShape;
}

// get scales of data rank size
// if "scale" version: set scales with input scales, 1.F for other dims not in axis
// if "size" version: scales = shape[target] / shape[input].pad, 1.F for other dims not in axis
// scales is a required input, but should not use input scales when "size" case, which may added eps or is a dummy
// value, recalculate scales instead.
std::vector<float> Interpolate::getScales(const VectorDims& srcDim,
                                          const VectorDims& srcDimPad,
                                          const VectorDims& dstDim) {
    std::vector<float> fullScales(dataRank, 1.F);
    const size_t axesRank = axes.size();
    for (size_t i = 0; i < axesRank; i++) {
        int axis = axes[i];
        if (interpAttrs.NCHWAsNHWC && dataRank == 4) {
            if (axis == 1) {
                axis = 2;
            } else if (axis == 2) {
                axis = 3;
            }
        }
        const bool pillowMode = interpAttrs.mode == InterpolateMode::bilinear_pillow ||
                                interpAttrs.mode == InterpolateMode::bicubic_pillow;
        const bool usePadded = pillowMode || hasPad;
        const auto& effectiveSrc = usePadded ? srcDimPad : srcDim;
        fullScales[axis] = (interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::scales)
                               ? scales[i]
                               : static_cast<float>(dstDim[axis]) / static_cast<float>(effectiveSrc[axis]);
        if (std::getenv("OV_DEBUG_INTERP_PAD")) {
            std::fprintf(stderr,
                         "[Interpolate::getScales] axis=%d dst=%zu src=%zu padded=%zu usePadded=%d scale=%f mode=%d\n",
                         axis,
                         dstDim[axis],
                         srcDim[axis],
                         srcDimPad[axis],
                         usePadded ? 1 : 0,
                         fullScales[axis],
                         static_cast<int>(interpAttrs.shapeCalcMode));
        }
    }
    return fullScales;
}

void Interpolate::execute([[maybe_unused]] const dnnl::stream& strm) {
    CPU_NODE_ASSERT(m_executor, "Interpolate: executor is not created");
    m_executor->execute(m_memory);
}

bool Interpolate::canFuse(const NodePtr& /*node*/) const {
    // Fusing is disabled for Interpolate in the refactored executor architecture
    // to keep semantics clear and avoid oneDNN-specific post-ops.
    return false;
}

bool Interpolate::created() const {
    return getType() == Type::Interpolate;
}

}  // namespace ov::intel_cpu::node
