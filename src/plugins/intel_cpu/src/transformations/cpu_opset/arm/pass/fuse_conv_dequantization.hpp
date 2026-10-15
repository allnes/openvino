// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

// Before: q(i8/u8) -> [Convert] -> Subtract(zp) -> Conv(W_i8) -> Multiply(scales) -> tail
// After:  q(i8/u8) -> Conv{zp, scales; f32 output} -> [Convert(f16)] -> tail
// The executor applies the per-tensor zero point and scalar/per-channel DQ scales.
// If LPT folded scales into a following FQ, they remain in FQ and Conv uses scale 1.
// Scaling precedes the optional f16 conversion so raw integer sums cannot overflow f16.
class FuseConvolutionDequantization : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseConvolutionDequantization");
    FuseConvolutionDequantization();
};

}  // namespace ov::intel_cpu
