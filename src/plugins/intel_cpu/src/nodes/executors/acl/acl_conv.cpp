// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_conv.hpp"

#include <arm_compute/core/CoreTypes.h>
#include <arm_compute/core/Error.h>
#include <arm_compute/core/QuantizationInfo.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/NEON/functions/NEConvolutionLayer.h>
#include <arm_compute/runtime/Scheduler.h>

#include <algorithm>
#include <any>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

#include "acl_utils.hpp"
#include "cpu_shape.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/acl/acl_common_executor.hpp"
#include "nodes/executors/common/common_utils.hpp"
#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/debug_messages.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "post_ops.hpp"
#include "src/cpu/kernels/CpuConvertQuantizedSignednessKernel.h"
#include "utils/general_utils.h"

namespace ov::intel_cpu {
namespace {

class ACLFloatConvolution : public arm_compute::IFunction {
public:
    ACLFloatConvolution(const ACLTensors& tensors,
                        const arm_compute::PadStrideInfo& stride,
                        const arm_compute::WeightsInfo& weights,
                        const arm_compute::Size2D& dilation,
                        const arm_compute::ActivationLayerInfo& activation,
                        const std::vector<float>& scales)
        : source(tensors[ACL_SRC_0].get()),
          bias(tensors[ACL_BIAS].get()),
          destination(tensors[ACL_DST].get()),
          outputScales(scales.size() > 1 ? scales : std::vector<float>{}) {
        auto* input = source;
        // ACL float-output GEMM accepts signed input; XOR 128 preserves q - zero_point.
        if (source->info()->data_type() == arm_compute::DataType::QASYMM8) {
            const auto quantization = source->info()->quantization_info().uniform();
            auto info = source->info()->clone();
            info->set_data_type(arm_compute::DataType::QASYMM8_SIGNED);
            info->set_quantization_info(arm_compute::QuantizationInfo(quantization.scale, quantization.offset - 128));
            signedSource.allocator()->init(*info);
            conversion.configure(source->info(), signedSource.info());
            input = &signedSource;
        }
        convolution.configure(input,
                              tensors[ACL_WEI].get(),
                              outputScales.empty() ? bias : nullptr,
                              destination,
                              stride,
                              weights,
                              dilation,
                              outputScales.empty() ? activation : arm_compute::ActivationLayerInfo());
        if (input == &signedSource) {
            signedSource.allocator()->allocate();
        }
        if (!outputScales.empty() && activation.enabled()) {
            postActivation.configure(destination, destination, activation);
            withActivation = true;
        }
    }

    void prepare() override {
        convolution.prepare();
    }

    void run() override {
        if (source->info()->data_type() == arm_compute::DataType::QASYMM8) {
            arm_compute::ITensorPack pack{{arm_compute::TensorType::ACL_SRC, source},
                                          {arm_compute::TensorType::ACL_DST, &signedSource}};
            arm_compute::Scheduler::get().schedule_op(&conversion,
                                                      arm_compute::Window::DimY,
                                                      conversion.window(),
                                                      pack);
        }
        convolution.run();
        if (!outputScales.empty()) {
            const auto& shape = destination->info()->tensor_shape();
            const auto& strides = destination->info()->strides_in_bytes();
            auto* buffer = destination->buffer() + destination->info()->offset_first_element_in_bytes();
            const auto* biasData =
                bias ? reinterpret_cast<const float*>(bias->buffer() + bias->info()->offset_first_element_in_bytes())
                     : nullptr;
            ov::parallel_for3d(shape[3], shape[2], shape[1], [&](size_t n, size_t y, size_t x) {
                auto* data = reinterpret_cast<float*>(buffer + n * strides[3] + y * strides[2] + x * strides[1]);
                for (size_t c = 0; c < outputScales.size(); ++c) {
                    data[c] = data[c] * outputScales[c] + (biasData ? biasData[c] : 0.0f);
                }
            });
            if (withActivation) {
                postActivation.run();
            }
        }
    }

private:
    arm_compute::Tensor* source;
    arm_compute::Tensor* bias;
    arm_compute::Tensor* destination;
    arm_compute::Tensor signedSource;
    arm_compute::cpu::kernels::CpuConvertQuantizedSignednessKernel conversion;
    arm_compute::NEConvolutionLayer convolution;
    arm_compute::NEActivationLayer postActivation;
    std::vector<float> outputScales;
    bool withActivation = false;
};

}  // namespace

ACLConvolutionExecutor::ACLConvolutionExecutor(const ConvAttrs& attrs,
                                               const MemoryArgs& memory,
                                               [[maybe_unused]] const ExecutorContext::CPtr& context)
    : weightScale(attrs.dqScales),
      inputZeroPoint(attrs.inputZeroPoint.value_or(0)) {
    aclTensorAttrs.hasLayoutTypeNHWC = memory.at(ARG_SRC)->getDescPtr()->hasLayoutType(LayoutType::nspc);

    MemoryDescPtr srcMemPtr = memory.at(ARG_SRC_0)->getDescPtr();
    MemoryDescPtr weiMemPtr = memory.at(ARG_WEI)->getDescPtr();
    MemoryDescPtr dstMemPtr = memory.at(ARG_DST)->getDescPtr();

    Shape weiShape = weiMemPtr->getShape();
    Shape srcShape = srcMemPtr->getShape();
    Shape dstShape = dstMemPtr->getShape();

    const auto with_groups = static_cast<int>(weiShape.getRank() == srcShape.getRank() + 1);
    const int kh = weiShape.getDims()[with_groups + srcShape.getRank() - 2];
    const int kw = weiShape.getDims()[with_groups + srcShape.getRank() - 1];
    const int oc = dstShape.getDims()[1];

    weightsInfo = arm_compute::WeightsInfo(false, kw, kh, oc, false, arm_compute::WeightFormat::UNSPECIFIED);
    auto paddingLeft = (attrs.paddingL.size() >= 2U) ? attrs.paddingL[1] : attrs.paddingL[0];
    auto paddingRight = (attrs.paddingR.size() >= 2U) ? attrs.paddingR[1] : attrs.paddingR[0];
    auto paddingTop = (attrs.paddingL.size() >= 2U) ? attrs.paddingL[0] : 0;
    auto paddingBottom = (attrs.paddingR.size() >= 2U) ? attrs.paddingR[0] : 0;
    padStrideInfo = arm_compute::PadStrideInfo(attrs.stride[0],
                                               attrs.stride[1],
                                               paddingLeft,
                                               paddingRight,
                                               paddingTop,
                                               paddingBottom,
                                               arm_compute::DimensionRoundingType::FLOOR);
    dilation = arm_compute::Size2D(attrs.dilation[1] + 1, attrs.dilation[0] + 1);
    // Instead of checking ==1 ==2 branch, iterate throught the vector
    for (const auto& postOp : attrs.postOps) {
        if (const auto* const activation = std::any_cast<ActivationPostOp>(&postOp)) {
            activationLayerInfo = getActivationLayerInfo(convertToEltwiseAlgorithm(activation->type()),
                                                         activation->alpha(),
                                                         activation->beta(),
                                                         activation->gamma());
        } else if (const auto* const fq = std::any_cast<FakeQuantizePostOp>(&postOp)) {
            fqInputScale = fq->inputScale();
            fqInputShift = fq->inputShift();
            fqOutputScale = fq->outputScale();
            fqOutputShift = fq->outputShift();

            // ACL destination quantization supports only per-tensor scale/shift.
            // For per-channel FQ input scales, move channel-wise ratios to weights scales.
            if (fqInputScale.size() > 1) {
                const auto baseScale = fqInputScale.front();
                const bool hasValidBaseScale = std::fabs(baseScale) > std::numeric_limits<float>::epsilon();
                const bool hasUniformShift = isPerTensorDataWithTolerance(fqInputShift, 0.00005F);

                if (hasValidBaseScale && hasUniformShift) {
                    std::vector<float> scaleRatios(fqInputScale.size(), 1.0F);
                    for (std::size_t i = 0; i < fqInputScale.size(); i++) {
                        scaleRatios[i] = fqInputScale[i] / baseScale;
                    }

                    multiplyAndBroadcastScales(weightScale, scaleRatios, fqInputScale.size());
                    fqInputScale = {baseScale};
                    if (fqInputShift.size() > 1) {
                        fqInputShift = {fqInputShift.front()};
                    }
                }
            }

            foldFakeQuantizeOutputShift(fqInputShift, fqOutputScale, fqOutputShift);
            fqOutputShift.clear();
        } else {
            OPENVINO_THROW("ACLConvolutionExecutor: the executor supports FakeQuantize and Activation post ops only");
        }
    }
}

bool ACLConvolutionExecutor::supports(const ConvConfig& config) {
    VERIFY(config.attrs.postOps.size() <= 2U, UNSUPPORTED_BY_EXECUTOR);

    const auto& srcDesc = config.descs.at(ARG_SRC);
    const auto& weiDesc = config.descs.at(ARG_WEI);
    const auto& dstDesc = config.descs.at(ARG_DST);

    VERIFY(aclSupported({srcDesc, weiDesc, dstDesc}), UNSUPPORTED_ACL_COMMON_PRECONDITION);

    // ACL GemmConv2d supports 4D activations and 4D weight only
    VERIFY(srcDesc->getShape().getRank() == 4 && weiDesc->getShape().getRank() == 4, UNSUPPORTED_BY_EXECUTOR);
    // isQuantized verifies whether src is u8/i8, weights is i8 and FQ is fused if dst is u8/i8
    // the last requirement is due to ACL int32 accumulation that needs to be requantized by non-trivial scales
    const bool hasQuantizationPostOp =
        std::any_of(config.attrs.postOps.begin(), config.attrs.postOps.end(), [](const std::any& op) {
            return std::any_cast<FakeQuantizePostOp>(&op) != nullptr;
        });
    const bool isQuantizedU8 = srcDesc->getPrecision() == ov::element::u8 &&
                               any_of(weiDesc->getPrecision(), ov::element::u8, ov::element::i8) &&
                               dstDesc->getPrecision() == ov::element::u8 && hasQuantizationPostOp;
    const bool isQuantizedI8 = srcDesc->getPrecision() == ov::element::i8 &&
                               weiDesc->getPrecision() == ov::element::i8 &&
                               dstDesc->getPrecision() == ov::element::i8 && hasQuantizationPostOp;
    const bool isQuantizedI8DstF32 = any_of(srcDesc->getPrecision(), ov::element::u8, ov::element::i8) &&
                                     weiDesc->getPrecision() == ov::element::i8 &&
                                     dstDesc->getPrecision() == ov::element::f32;

    VERIFY(isQuantizedU8 || isQuantizedI8 || isQuantizedI8DstF32, UNSUPPORTED_BY_EXECUTOR);
    VERIFY(!config.attrs.inputZeroPoint || isQuantizedI8DstF32, UNSUPPORTED_BY_EXECUTOR);
    VERIFY(!isQuantizedI8DstF32 || config.attrs.stride[0] == config.attrs.stride[1], UNSUPPORTED_BY_EXECUTOR);
    if (config.attrs.withBias) {
        const auto biasPrecision = config.descs.at(ARG_BIAS)->getPrecision();
        if (isQuantizedI8DstF32) {
            VERIFY(biasPrecision == ov::element::f32, UNSUPPORTED_BIAS_PRECISIONS);
        } else {
            VERIFY(biasPrecision == ov::element::i32, UNSUPPORTED_BIAS_PRECISIONS);
        }
    }

    return true;
}

arm_compute::Status ACLConvolutionExecutor::validateTensorsInfo(const ACLInfos& aclMemoryInfos) {
    // Note: LPT propagate dequantization scales from src and weights on conv output, and the result scale
    // is applied as weight scale. So quantization configuration forms in the following way:
    // - src quantization carries a per-tensor zero point for the floating-output path
    // - weights: scale is equal to result dequantization scale after Convolution propagated by LPT
    //            shift is not supported
    // - destination: scale is formed based on requantization FakeQuantize parameters: scale = 1.0 / input scale
    //                shift = input shift. Skipped for f32 dst (dequantize-to-float path).
    aclMemoryInfos[ACLArgs::ACL_SRC_0]->set_quantization_info(arm_compute::QuantizationInfo(1.0f, inputZeroPoint));
    const bool floatOutput = aclMemoryInfos[ACLArgs::ACL_DST]->data_type() == arm_compute::DataType::F32;
    // ACL float dequantization has a uniform scale; apply channel scales after accumulation.
    const bool perChannelFloatOutput = floatOutput && weightScale.size() > 1;
    aclMemoryInfos[ACLArgs::ACL_WEI]->set_quantization_info(weightScale.empty() || perChannelFloatOutput
                                                                ? arm_compute::QuantizationInfo(1.0F)
                                                                : arm_compute::QuantizationInfo(weightScale));
    if (aclMemoryInfos[ACLArgs::ACL_DST]->data_type() != arm_compute::DataType::F32) {
        const auto dstPrecision = aclMemoryInfos[ACLArgs::ACL_DST]->data_type() == arm_compute::DataType::QASYMM8_SIGNED
                                      ? ov::element::i8
                                      : ov::element::u8;
        aclMemoryInfos[ACLArgs::ACL_DST]->set_quantization_info(
            getDstQuantizationInfo(fqInputScale, fqInputShift, dstPrecision));
    }

    auto source = *aclMemoryInfos[ACLArgs::ACL_SRC_0];
    if (floatOutput && source.data_type() == arm_compute::DataType::QASYMM8) {
        source.set_data_type(arm_compute::DataType::QASYMM8_SIGNED);
        source.set_quantization_info(arm_compute::QuantizationInfo(1.0f, inputZeroPoint - 128));
    }
    return arm_compute::NEConvolutionLayer::validate(
        &source,
        aclMemoryInfos[ACLArgs::ACL_WEI].get(),
        perChannelFloatOutput ? nullptr : aclMemoryInfos[ACLArgs::ACL_BIAS].get(),
        aclMemoryInfos[ACLArgs::ACL_DST].get(),
        padStrideInfo,
        weightsInfo,
        dilation,
        perChannelFloatOutput ? arm_compute::ActivationLayerInfo() : activationLayerInfo,
        false,  // enable fast math
        1);     // num_groups
}

ACLFunction ACLConvolutionExecutor::configureFunction(const ACLTensors& aclMemoryTensors) {
    if (aclMemoryTensors[ACL_DST]->info()->data_type() == arm_compute::DataType::F32) {
        return std::make_unique<ACLFloatConvolution>(aclMemoryTensors,
                                                     padStrideInfo,
                                                     weightsInfo,
                                                     dilation,
                                                     activationLayerInfo,
                                                     weightScale);
    }
    auto neConv = std::make_unique<arm_compute::NEConvolutionLayer>();

    neConv->configure(aclMemoryTensors[ACLArgs::ACL_SRC_0].get(),
                      aclMemoryTensors[ACLArgs::ACL_WEI].get(),
                      aclMemoryTensors[ACLArgs::ACL_BIAS].get(),
                      aclMemoryTensors[ACLArgs::ACL_DST].get(),
                      padStrideInfo,
                      weightsInfo,
                      dilation,
                      activationLayerInfo,
                      false,  // enable fast math
                      1);     // num_groups
    return neConv;
}

std::shared_ptr<arm_compute::TensorInfo> ACLConvolutionExecutor::initTensorInfo(
    const arm_compute::TensorShape& tensorShape,
    const arm_compute::DataType& dataType,
    const arm_compute::DataLayout& dataLayout) {
    return ACLCommonExecutor::initTensorInfo(tensorShape, convertToQuantizedType(dataType), dataLayout);
}

}  // namespace ov::intel_cpu
