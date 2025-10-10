// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

// Decompose 3D ConvolutionBackpropData (NCDHW) into optional Interpolate (NEAREST), Pad, and Conv3D with transposed weights.
// For stride > 1, upsampling is performed via Interpolate to ensure fast path on ARM.
class Deconv3DDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("Deconv3DDecomposition");
    Deconv3DDecomposition();
};

}  // namespace ov::intel_cpu
