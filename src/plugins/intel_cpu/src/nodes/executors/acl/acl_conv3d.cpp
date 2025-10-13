// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_conv3d.hpp"

#include <arm_compute/core/CoreTypes.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/NEON/functions/NEConv3D.h>

#include <any>
#include <memory>

#include "acl_utils.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/debug_messages.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "post_ops.hpp"

namespace ov::intel_cpu {

ACLConv3DExecutor::ACLConv3DExecutor(const ConvAttrs& attrs,
                                     const MemoryArgs& /*memory*/,
                                     const ExecutorContext::CPtr& /*context*/) {
    // We'll convert src/dst shapes to NDHWC explicitly in updateTensorsShapes
    // Build Conv3dInfo from ConvAttrs
    // Expect stride/dilation/padding vectors of size 3 (D, H, W) or at least size >= 1
    const auto sd = attrs.stride;
    const auto dl = attrs.dilation;
    const auto pl = attrs.paddingL;
    const auto pr = attrs.paddingR;

    const unsigned sx = static_cast<unsigned>((sd.size() >= 3) ? sd[2] : (sd.size() >= 1 ? sd[0] : 1));
    const unsigned sy = static_cast<unsigned>((sd.size() >= 2) ? sd[1] : (sd.size() >= 1 ? sd[0] : 1));
    const unsigned sz = static_cast<unsigned>((sd.size() >= 3) ? sd[0] : 1);

    const unsigned dx = static_cast<unsigned>((dl.size() >= 3) ? (dl[2] + 1) : ((dl.size() >= 1) ? (dl[0] + 1) : 1));
    const unsigned dy = static_cast<unsigned>((dl.size() >= 2) ? (dl[1] + 1) : ((dl.size() >= 1) ? (dl[0] + 1) : 1));
    const unsigned dz = static_cast<unsigned>((dl.size() >= 3) ? (dl[0] + 1) : 1);

    const unsigned pad_left =
        static_cast<unsigned>((pl.size() >= 3) ? pl[2] : (pl.size() >= 2 ? pl[1] : (pl.size() >= 1 ? pl[0] : 0)));
    const unsigned pad_right =
        static_cast<unsigned>((pr.size() >= 3) ? pr[2] : (pr.size() >= 2 ? pr[1] : (pr.size() >= 1 ? pr[0] : 0)));
    const unsigned pad_top = static_cast<unsigned>((pl.size() >= 2) ? pl[1] : (pl.size() >= 1 ? 0 : 0));
    const unsigned pad_bottom = static_cast<unsigned>((pr.size() >= 2) ? pr[1] : (pr.size() >= 1 ? 0 : 0));
    const unsigned pad_front = static_cast<unsigned>((pl.size() >= 3) ? pl[0] : 0);
    const unsigned pad_back = static_cast<unsigned>((pr.size() >= 3) ? pr[0] : 0);

    arm_compute::Size3D stride3d{sx, sy, sz};
    arm_compute::Padding3D padding3d{pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back};
    arm_compute::Size3D dilation3d{dx, dy, dz};

    // Map single Activation post-op if present
    arm_compute::ActivationLayerInfo act_info;
    if (attrs.postOps.size() == 1) {
        if (const auto* const activation = std::any_cast<ActivationPostOp>(attrs.postOps.data())) {
            act_info = getActivationLayerInfo(convertToEltwiseAlgorithm(activation->type()),
                                              activation->alpha(),
                                              activation->beta(),
                                              activation->gamma());
        }
    }

    // Use FLOOR rounding to match existing 2D path
    m_conv3d_info = arm_compute::Conv3dInfo(stride3d,
                                            padding3d,
                                            act_info,
                                            dilation3d,
                                            arm_compute::DimensionRoundingType::FLOOR,
                                            false);
}

bool ACLConv3DExecutor::supports(const ConvConfig& config) {
    // Only float precisions, non-quantized, rank 5 tensors, no groups
    const auto src_prec = config.descs.at(ARG_SRC)->getPrecision();
    const auto wei_prec = config.descs.at(ARG_WEI)->getPrecision();
    const bool is_quantized =
        ((src_prec == ov::element::u8) || (src_prec == ov::element::i8)) && (wei_prec == ov::element::i8);
    VERIFY(!is_quantized, UNSUPPORTED_SRC_PRECISIONS);

    // Expect 5D input/output/weights
    VERIFY(srcRank(config) == 5 && dstRank(config) == 5 && weiRank(config) >= 5, UNSUPPORTED_BY_EXECUTOR);

    // Groups not supported for NEConv3D path currently
    VERIFY(!config.attrs.isGrouped, UNSUPPORTED_BY_EXECUTOR);

    // Supported dtypes: f16/f32
    const auto s = srcType(config);
    const auto w = weiType(config);
    const auto d = dstType<ConvConfig>(config);
    VERIFY((s == ov::element::f16) || (s == ov::element::f32), UNSUPPORTED_SRC_PRECISIONS);
    VERIFY((w == ov::element::f16) || (w == ov::element::f32), UNSUPPORTED_WEI_PRECISIONS);
    VERIFY((d == ov::element::f16) || (d == ov::element::f32), UNSUPPORTED_DST_PRECISIONS);

    // Post-ops: allow at most one Activation post-op
    VERIFY(config.attrs.postOps.size() <= 1U, UNSUPPORTED_POST_OPS);

    return true;
}

void ACLConv3DExecutor::updateTensorsShapes(ACLShapes& aclMemoryShapes) {
    // Convert src and dst shapes to NDHWC ordering expected by NEConv3D
    // ACLCommon prepares shapes with reversed dims (e.g., NCDHW -> WHDCN),
    // changeLayoutToNH_C turns WHDCN into CWHDN which corresponds to NDHWC.
    std::vector<arm_compute::TensorShape*> nhwcShapes;
    if (aclMemoryShapes[ACLArgs::ACL_SRC_0].num_dimensions() >= 5) {
        nhwcShapes.push_back(&aclMemoryShapes[ACLArgs::ACL_SRC_0]);
    }
    if (aclMemoryShapes[ACLArgs::ACL_DST].num_dimensions() >= 5) {
        nhwcShapes.push_back(&aclMemoryShapes[ACLArgs::ACL_DST]);
    }
    if (!nhwcShapes.empty()) {
        changeLayoutToNH_C(nhwcShapes);
    }
    // Weights are kept as produced by shapeCast (kW,kH,kD,IC,OC), matching ACL expectation
}

arm_compute::Status ACLConv3DExecutor::validateTensorsInfo(const ACLInfos& aclMemoryInfos) {
    // Force NDHWC for 3D inputs/outputs expected by NEConv3D
    if (aclMemoryInfos[ACLArgs::ACL_SRC_0]) {
        aclMemoryInfos[ACLArgs::ACL_SRC_0]->set_data_layout(arm_compute::DataLayout::NDHWC);
    }
    if (aclMemoryInfos[ACLArgs::ACL_DST]) {
        aclMemoryInfos[ACLArgs::ACL_DST]->set_data_layout(arm_compute::DataLayout::NDHWC);
    }
    return arm_compute::NEConv3D::validate(aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
                                           aclMemoryInfos[ACLArgs::ACL_WEI].get(),
                                           aclMemoryInfos[ACLArgs::ACL_BIAS].get(),
                                           aclMemoryInfos[ACLArgs::ACL_DST].get(),
                                           m_conv3d_info);
}

ACLFunction ACLConv3DExecutor::configureFunction(const ACLTensors& aclMemoryTensors) {
    auto fn = std::make_unique<arm_compute::NEConv3D>();
    // Ensure tensors carry NDHWC layout for NEConv3D
    if (aclMemoryTensors[ACLArgs::ACL_SRC_0] && aclMemoryTensors[ACLArgs::ACL_SRC_0]->info()) {
        aclMemoryTensors[ACLArgs::ACL_SRC_0]->info()->set_data_layout(arm_compute::DataLayout::NDHWC);
    }
    if (aclMemoryTensors[ACLArgs::ACL_DST] && aclMemoryTensors[ACLArgs::ACL_DST]->info()) {
        aclMemoryTensors[ACLArgs::ACL_DST]->info()->set_data_layout(arm_compute::DataLayout::NDHWC);
    }
    fn->configure(aclMemoryTensors[ACLArgs::ACL_SRC_0].get(),
                  aclMemoryTensors[ACLArgs::ACL_WEI].get(),
                  aclMemoryTensors[ACLArgs::ACL_BIAS].get(),
                  aclMemoryTensors[ACLArgs::ACL_DST].get(),
                  m_conv3d_info);
    return fn;
}

}  // namespace ov::intel_cpu
