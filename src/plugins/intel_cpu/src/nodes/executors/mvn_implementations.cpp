// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "mvn_config.hpp"
#include "nodes/executors/mvn.hpp"
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

// static bool is_data_type_supported(const ov::element::Type& dt) {
//     return one_of(dt, ov::element::f32, ov::element::bf16, ov::element::f16);
// }

// Mapping notation for MVN arguments
static const MappingNotation mvnMappingNotation{ARG_SRC, ARG_DST};

// Layout configuration for MVN - support both planar and channel-last formats
using LayoutConfig = std::vector<LayoutType>;
static const LayoutConfig mvnPlanarLayoutConfig{LayoutType::ncsp, LayoutType::ncsp};
static const LayoutConfig mvnByChannelLayoutConfig{LayoutType::nspc, LayoutType::nspc};

// Type mapping for MVN - supports f32, bf16, f16
static const TypeMapping mvnTypeMapping {
    // {src, dst}                                   pt<src, dst>
    {{_f32, _f32},   pt(bypass(), bypass())},
    {{_bf16, _bf16}, pt(bypass(), bypass())},
    {{_f16, _f16},   pt(bypass(), bypass())},
    // Fallback to f32 for any unsupported type configuration
    {{_any, _any},   pt(just<ov::element::f32>(), just<ov::element::f32>())},
};

// to keep OV_CPU_INSTANCE macros aligned
// clang-format off
template <>
const std::vector<ExecutorImplementation<MVNAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<MVNAttrs>> mvnImplementations {
// TODO: Re-enable JIT implementation once JIT kernels are reimplemented
// OV_CPU_INSTANCE_X64(
//             "mvn_jit_x64",
//             ExecutorType::jit_x64,
//             OperationType::MVN,
//             ShapeTolerance::Agnostic,
//             // supports
//             [](const executor::Config<MVNAttrs>& config) -> bool {
//                 VERIFY(srcType(config) == ov::element::f32 || 
//                        srcType(config) == ov::element::bf16 || 
//                        srcType(config) == ov::element::f16, UNSUPPORTED_SRC_PRECISIONS);
//                 VERIFY(dstType(config) == ov::element::f32 || 
//                        dstType(config) == ov::element::bf16 || 
//                        dstType(config) == ov::element::f16, UNSUPPORTED_DST_PRECISIONS);
//                 return MVNJitExecutor::supports(config.attrs, config.descs.inputs, config.descs.outputs);
//             },
//             // createOptimalConfig
//             [](const executor::Config<MVNAttrs>& config) -> std::optional<executor::Config<MVNAttrs>> {
//                 return createOptimalConfigCommon(config,
//                                                  mvnTypeMapping,
//                                                  mvnLayoutConfig,
//                                                  mvnMappingNotation);
//             },
//             AcceptsAnyShape<MVNAttrs>{},
//             CreateDefault<MVNJitExecutor, MVNAttrs>{}
//             )
        OV_CPU_INSTANCE_ACL(
            "mvn_acl",
            ExecutorType::Acl,
            OperationType::MVN,
            ShapeTolerance::Agnostic,
            // supports
            [](const executor::Config<MVNAttrs>& config) -> bool {
                VERIFY(srcType(config) == ov::element::f32 || 
                       srcType(config) == ov::element::bf16 || 
                       srcType(config) == ov::element::f16, UNSUPPORTED_SRC_PRECISIONS);
                VERIFY(dstType(config) == ov::element::f32 || 
                       dstType(config) == ov::element::bf16 || 
                       dstType(config) == ov::element::f16, UNSUPPORTED_DST_PRECISIONS);
                // ACL only supports normalize_variance=true
                VERIFY(config.attrs.normalizeVariance_, "ACL MVN supports normalize_variance=true only");
                // ACL doesn't support OUTSIDE_SQRT mode
                VERIFY(config.attrs.epsMode_ == MVNEpsMode::INSIDE_SQRT, "ACL MVN supports INSIDE_SQRT mode only");
                return true;
            },
            // createOptimalConfig
            [](const executor::Config<MVNAttrs>& config) -> std::optional<executor::Config<MVNAttrs>> {
                // Choose layout config based on input layout
                const auto& srcDesc = config.descs.at(ARG_SRC_0);
                bool isChannelLast = srcDesc->hasLayoutType(LayoutType::nspc);
                const auto& layoutConfig = isChannelLast ? mvnByChannelLayoutConfig : mvnPlanarLayoutConfig;
                
                return createOptimalConfigCommon(config,
                                                 mvnTypeMapping,
                                                 layoutConfig,
                                                 mvnMappingNotation);
            },
            AcceptsAnyShape<MVNAttrs>{},
            CreateDefault<AclMVNExecutor, MVNAttrs>{}
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
                // Choose layout config based on input layout
                const auto& srcDesc = config.descs.at(ARG_SRC_0);
                bool isChannelLast = srcDesc->hasLayoutType(LayoutType::nspc);
                const auto& layoutConfig = isChannelLast ? mvnByChannelLayoutConfig : mvnPlanarLayoutConfig;
                
                return createOptimalConfigCommon(config,
                                                 mvnTypeMapping,
                                                 layoutConfig,
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