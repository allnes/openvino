// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/pass/matcher_pass.hpp>

namespace ov::intel_cpu {

// Force 3D Deconvolution (ConvolutionBackpropData rank-5) to FP32 on ARM
// to enable oneDNN implementation and avoid ref_f16 fallback. Surround with
// Convert(f16->f32) on inputs and Convert(f32->f16) on output to keep model dtype.
class Deconv3DForceFP32 : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("Deconv3DForceFP32", "0");
    Deconv3DForceFP32();
};

}  // namespace ov::intel_cpu

