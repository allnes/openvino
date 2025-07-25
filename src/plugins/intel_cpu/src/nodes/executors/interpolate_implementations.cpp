// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interpolate_config.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/executor_config.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"
#include "utils/arch_macros.h"
#include "openvino/core/type/element_type.hpp"

#if defined(OV_CPU_WITH_ACL)
#include "nodes/executors/acl/acl_interpolate.hpp"
#endif

#if defined(OPENVINO_ARCH_X86_64)
#include "nodes/executors/x64/jit_interpolate.hpp"
#endif

// Reference implementation should always be available
#include "nodes/executors/common/ref_interpolate.hpp"

namespace ov {
namespace intel_cpu {

#if defined(OV_CPU_WITH_ACL)
static bool isACLSupportedConfiguration(
    const InterpolateAttrs& attrs,
    const MemoryDescPtr& srcDesc,
    const MemoryDescPtr& dstDesc) {
    
    const auto& inp_shape = srcDesc->getShape().getDims();
    const auto& out_shape = dstDesc->getShape().getDims();
    
    static const size_t index_h = 2;
    static const size_t index_w = 3;
    float scale_h = static_cast<float>(out_shape[index_h]) / inp_shape[index_h];
    float scale_w = static_cast<float>(out_shape[index_w]) / inp_shape[index_w];
    bool is_upsample = scale_h > 1 && scale_w > 1;
    
    const auto& coord_mode = attrs.coordTransMode;
    const auto& nearest_mode = attrs.nearestMode;
    
    if (coord_mode == InterpolateCoordTransMode::align_corners &&
        nearest_mode == InterpolateNearestMode::round_prefer_ceil) {
        return true;
    }
    
    if (coord_mode == InterpolateCoordTransMode::half_pixel &&
        (nearest_mode == InterpolateNearestMode::simple || nearest_mode == InterpolateNearestMode::round_prefer_ceil)) {
        return false;
    }
    
    if (coord_mode == InterpolateCoordTransMode::asymmetric &&
        (nearest_mode == InterpolateNearestMode::simple || nearest_mode == InterpolateNearestMode::floor)) {
        return is_upsample;
    }
    
    if (is_upsample) {
        bool int_factor = scale_h == static_cast<int>(scale_h) && scale_w == static_cast<int>(scale_w);
        if (int_factor && coord_mode != InterpolateCoordTransMode::asymmetric &&
            (nearest_mode == InterpolateNearestMode::round_prefer_ceil ||
             nearest_mode == InterpolateNearestMode::round_prefer_floor)) {
            return true;
        }
    } else if (scale_h < 1 && scale_w < 1) {
        float down_scale_h = static_cast<float>(inp_shape[index_h]) / out_shape[index_h];
        float down_scale_w = static_cast<float>(inp_shape[index_w]) / out_shape[index_w];
        bool int_factor =
            down_scale_h == static_cast<int>(down_scale_h) && down_scale_w == static_cast<int>(down_scale_w);
        
        if (int_factor && coord_mode != InterpolateCoordTransMode::align_corners &&
            nearest_mode == InterpolateNearestMode::simple) {
            return true;
        }
        
        if (int_factor && nearest_mode == InterpolateNearestMode::round_prefer_ceil &&
            ((out_shape[index_h] > 1 && out_shape[index_w] > 1) ||
             coord_mode != InterpolateCoordTransMode::half_pixel)) {
            return true;
        }
    }
    
    return false;
}

static bool isACLInterpolateSupported(const executor::Config<InterpolateAttrs>& config) {
    const auto& attrs = config.attrs;
    const auto& srcDesc = config.descs.at(ARG_SRC);
    const auto& dstDesc = config.descs.at(ARG_DST);
    
    if (srcDesc->getShape().getDims().size() != 4u) {
        DEBUG_LOG("ACL Interpolate does not support src shape rank: ", srcDesc->getShape().getDims().size());
        return false;
    }
    
    const auto& pads_begin = attrs.padBegin;
    const auto& pads_end = attrs.padEnd;
    
    if (!std::all_of(pads_begin.begin(), pads_begin.end(), [](int i) { return i == 0; }) ||
        !std::all_of(pads_end.begin(), pads_end.end(), [](int i) { return i == 0; })) {
        DEBUG_LOG("ACL Interpolate does not support padding");
        return false;
    }
    
    if (attrs.antialias ||
        attrs.coordTransMode == InterpolateCoordTransMode::tf_half_pixel_for_nn ||
        attrs.nearestMode == InterpolateNearestMode::ceil) {
        DEBUG_LOG("ACL Interpolate does not support antialias, tf_half_pixel_for_nn, ceil modes");
        return false;
    }
    
    if (attrs.mode == InterpolateMode::cubic || 
        attrs.mode == InterpolateMode::bilinear_pillow ||
        attrs.mode == InterpolateMode::bicubic_pillow) {
        DEBUG_LOG("ACL Interpolate does not support cubic, bilinear_pillow, bicubic_pillow modes");
        return false;
    }
    
    if (attrs.shapeCalcMode == InterpolateShapeCalcMode::scales &&
        one_of(attrs.coordTransMode,
               InterpolateCoordTransMode::half_pixel,
               InterpolateCoordTransMode::asymmetric) &&
        one_of(attrs.mode, InterpolateMode::linear, InterpolateMode::linear_onnx)) {
        DEBUG_LOG("ACL Interpolate does not support scales mode with linear/linear_onnx and half_pixel/asymmetric");
        return false;
    }
    
    if (attrs.mode == InterpolateMode::nearest &&
        !isACLSupportedConfiguration(attrs, srcDesc, dstDesc)) {
        DEBUG_LOG("ACL Interpolate isSupportedConfiguration method fails for nearest mode");
        return false;
    }
    
    if (attrs.coordTransMode == InterpolateCoordTransMode::pytorch_half_pixel) {
        DEBUG_LOG("ACL Interpolate does not support pytorch_half_pixel mode");
        return false;
    }
    
    return true;
}
#endif

using namespace ov::element;
using namespace executor;

// Define shorthands for executor functions
using InterpolateExecutorCreator = std::function<ExecutorPtr(const InterpolateAttrs&,
                                                             const MemoryArgs&,
                                                             const ExecutorContext::CPtr)>;

template<typename T, typename Attrs>
struct InterpolateCreateDefault {
    ExecutorPtr operator()(const Attrs& attrs,
                          const MemoryArgs& memory,
                          const ExecutorContext::CPtr context) const {
        return std::make_shared<T>(attrs, nullptr, memory, context);
    }
};

template <>
const std::vector<ExecutorImplementation<InterpolateAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<InterpolateAttrs>> interpolateImplementations {
        // clang-format off
#if defined(OV_CPU_WITH_ACL)
        OV_CPU_INSTANCE_ACL(
            "interpolate_acl",
            ExecutorType::Acl,
            OperationType::Interpolate,
            ShapeTolerance::Agnostic,
            // Predicate function
            [](const executor::Config<InterpolateAttrs>& config) -> bool {
                return isACLInterpolateSupported(config);
            },
            HasNoOptimalConfig<InterpolateAttrs>{},
            AcceptsAnyShape<InterpolateAttrs>{},
            InterpolateCreateDefault<ACLInterpolateExecutor, InterpolateAttrs>{}
        )
#endif
#if defined(OPENVINO_ARCH_X86_64)
        OV_CPU_INSTANCE_X64(
            "interpolate_jit",
            ExecutorType::jit_x64, 
            OperationType::Interpolate,
            ShapeTolerance::Agnostic,
            // Predicate function
            [](const executor::Config<InterpolateAttrs>& config) -> bool {
                return jitInterpolateSupported(config.attrs, config.descs);
            },
            HasNoOptimalConfig<InterpolateAttrs>{},
            AcceptsAnyShape<InterpolateAttrs>{},
            InterpolateCreateDefault<JitInterpolateExecutor, InterpolateAttrs>{}
        )
#endif
        OV_CPU_INSTANCE_COMMON(
            "interpolate_ref",
            ExecutorType::Common,
            OperationType::Interpolate,
            ShapeTolerance::Agnostic,
            // Predicate function - always true for reference
            [](const executor::Config<InterpolateAttrs>& config) -> bool {
                return true;
            },
            HasNoOptimalConfig<InterpolateAttrs>{},
            AcceptsAnyShape<InterpolateAttrs>{},
            InterpolateCreateDefault<RefInterpolateExecutor, InterpolateAttrs>{}
        )
        // clang-format on
    };
    
    return interpolateImplementations;
}

}  // namespace intel_cpu
}  // namespace ov