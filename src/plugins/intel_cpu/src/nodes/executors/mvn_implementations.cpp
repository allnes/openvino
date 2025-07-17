// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "mvn_config.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/precision_matcher.hpp"
#include "nodes/executors/type_mask.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/arch_macros.h"
#include "utils/debug_capabilities.h"
#include "nodes/executors/debug_messages.hpp"

#if defined(OPENVINO_ARCH_X86_64)
#include "nodes/executors/x64/jit_mvn.hpp"
#endif

#if defined(OV_CPU_WITH_ACL)
#include "nodes/executors/acl/acl_mvn.hpp"
#endif

#include "nodes/executors/common/ref_mvn.hpp"

namespace ov::intel_cpu {

using namespace TypeMaskAlias;
using namespace executor;

// Mapping notation for MVN arguments
static const MappingNotation mvnMappingNotation{ARG_SRC, ARG_DST};

// // Layout configuration for MVN - support planar, channel-last and blocked formats
// using LayoutConfig = std::vector<LayoutType>;
// static const LayoutConfig mvnPlanarLayoutConfig{LayoutType::ncsp, LayoutType::ncsp};
// static const LayoutConfig mvnByChannelLayoutConfig{LayoutType::nspc, LayoutType::nspc};
// static const LayoutConfig mvnBlockedC8LayoutConfig{LayoutType::nCsp8c, LayoutType::nCsp8c};
// static const LayoutConfig mvnBlockedC16LayoutConfig{LayoutType::nCsp16c, LayoutType::nCsp16c};

// Type mapping for MVN - supports f32, bf16, f16, i8, u8
static const TypeMapping mvnTypeMapping {
    // {src, dst}                                   pt<src, dst>
    {{_f32, _f32},   pt(bypass(), bypass())},
    {{_bf16, _bf16}, pt(bypass(), bypass())},
    {{_f16, _f16},   pt(bypass(), bypass())},
    {{_i8, _f32},    pt(bypass(), bypass())},  // i8 input -> f32 output
    {{_u8, _f32},    pt(bypass(), bypass())},  // u8 input -> f32 output
    {{_i8, _i8},     pt(bypass(), bypass())},  // i8 input -> i8 output
    {{_u8, _u8},     pt(bypass(), bypass())},  // u8 input -> u8 output
    // Fallback to f32 for any unsupported type configuration
    {{_any, _any},   pt(just<ov::element::f32>(), just<ov::element::f32>())},
};

// Accept the input/output layouts as-is without conversion
static LayoutType getLayoutType(const MemoryDescPtr& desc) {
    if (desc->hasLayoutType(LayoutType::nspc)) return LayoutType::nspc;
    if (desc->hasLayoutType(LayoutType::nCsp16c)) return LayoutType::nCsp16c;
    if (desc->hasLayoutType(LayoutType::nCsp8c)) return LayoutType::nCsp8c;
    return LayoutType::ncsp;
}

// to keep OV_CPU_INSTANCE macros aligned
// clang-format off
template <>
const std::vector<ExecutorImplementation<MVNAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<MVNAttrs>> mvnImplementations {
        OV_CPU_INSTANCE_X64(
            "mvn_jit_x64",
            ExecutorType::jit_x64,
            OperationType::MVN,
            ShapeTolerance::Agnostic,
            // supports
            [](const executor::Config<MVNAttrs>& config) -> bool {
                return MVNJitExecutor::supports(config);
            },
            // createOptimalConfig
            [](const executor::Config<MVNAttrs>& config) -> std::optional<executor::Config<MVNAttrs>> {
                // Choose layout config based on input layout
                std::vector<LayoutType> actualLayouts = {
                    getLayoutType(config.descs.at(ARG_SRC_0)),
                    getLayoutType(config.descs.at(ARG_DST)),
                };
                return createOptimalConfigCommon(config,
                                                 mvnTypeMapping,
                                                 actualLayouts,
                                                 mvnMappingNotation);
            },
            AcceptsAnyShape<MVNAttrs>{},
            CreateDefault<MVNJitExecutor, MVNAttrs>{}
            )
        OV_CPU_INSTANCE_ACL(
            "mvn_acl",
            ExecutorType::Acl,
            OperationType::MVN,
            ShapeTolerance::Agnostic,
            // supports
            [](const executor::Config<MVNAttrs>& config) -> bool {
                return ACLMVNExecutor::supports(config);
            },
            // createOptimalConfig
            [](const executor::Config<MVNAttrs>& config) -> std::optional<executor::Config<MVNAttrs>> {
                // Choose layout config based on input layout
                std::vector<LayoutType> actualLayouts = {
                    getLayoutType(config.descs.at(ARG_SRC_0)),
                    getLayoutType(config.descs.at(ARG_DST)),
                };
                return createOptimalConfigCommon(config,
                                                 mvnTypeMapping,
                                                 actualLayouts,
                                                 mvnMappingNotation);
            },
            AcceptsAnyShape<MVNAttrs>{},
            CreateDefault<ACLMVNExecutor, MVNAttrs>{}
            )
        OV_CPU_INSTANCE_COMMON(
            "mvn_ref",
            ExecutorType::Common,
            OperationType::MVN,
            ShapeTolerance::Agnostic,
            // supports - always returns true as fallback
            SupportsAnyConfig<MVNAttrs>{},
            // createOptimalConfig
            [](const executor::Config<MVNAttrs>& config) -> std::optional<executor::Config<MVNAttrs>> {
                // Reference implementation accepts whatever layout is provided
                std::vector<LayoutType> actualLayouts = {
                    getLayoutType(config.descs.at(ARG_SRC_0)),
                    getLayoutType(config.descs.at(ARG_DST)),
                };
                return createOptimalConfigCommon(config,
                                                 mvnTypeMapping,
                                                 actualLayouts,
                                                 mvnMappingNotation);
            },
            AcceptsAnyShape<MVNAttrs>{},
            CreateDefault<MVNRefExecutor, MVNAttrs>{}
            )
    };
    
    return mvnImplementations;
}
// clang-format on

}  // namespace ov::intel_cpu