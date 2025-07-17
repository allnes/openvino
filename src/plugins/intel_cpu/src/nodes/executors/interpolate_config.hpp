// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_types.h"
#include "nodes/executors/executor_config.hpp"

#define MAX_INPUT_INTERPOLATE 8

namespace ov {
namespace intel_cpu {

static constexpr float PILLOW_BILINEAR_WINDOW_SCALE = 1.0f;
static constexpr float PILLOW_BICUBIC_WINDOW_SCALE = 2.0f;

enum InterpolateLayoutType : uint8_t { 
    planar, 
    block, 
    by_channel 
};

enum InterpolateMode : uint8_t { 
    nearest, 
    linear, 
    linear_onnx, 
    cubic, 
    bilinear_pillow, 
    bicubic_pillow 
};

enum InterpolateCoordTransMode : uint8_t {
    half_pixel,
    pytorch_half_pixel,
    asymmetric,
    tf_half_pixel_for_nn,
    align_corners
};

enum class InterpolateNearestMode : uint8_t { 
    round_prefer_floor, 
    round_prefer_ceil, 
    floor, 
    ceil, 
    simple 
};

enum class InterpolateShapeCalcMode : uint8_t { 
    sizes, 
    scales 
};

struct InterpolateAttrs {
    InterpolateShapeCalcMode shapeCalcMode = InterpolateShapeCalcMode::sizes;
    InterpolateMode mode = InterpolateMode::nearest;
    InterpolateCoordTransMode coordTransMode = InterpolateCoordTransMode::half_pixel;
    InterpolateNearestMode nearestMode = InterpolateNearestMode::round_prefer_floor;
    bool antialias = false;
    float cubeCoeff = -0.75f;
    std::vector<int> padBegin;
    std::vector<int> padEnd;
    ov::element::Type inPrc;
    ov::element::Type outPrc;
    InterpolateLayoutType layout = InterpolateLayoutType::planar;
    std::vector<float> dataScales;
    bool hasPad = false;
    // Some FEs or preprocessing step resize spatial dimension for tensors with NHWC layout memory,
    // but import them with a planar layout[abcd] with axis[1,2] for convenience. In this case, for pillow modes without
    // pad, the nhwc layout path and the specific kernel(nhwc layout executor) can be used for this planar layout and
    // axis settings(NCHWAsNHWC is true) to get better perf. To this end the following mapping is used:
    // 1. logical shape alignment [abcd-nhwc] to [adbc-nchw].
    // 2. axis alignment [1,2] to [2,3].
    // 3. config planar layout support and treated it as channel_first layout.
    bool NCHWAsNHWC = false;
};

using InterpolateConfig = ov::intel_cpu::executor::Config<InterpolateAttrs>;

}  // namespace intel_cpu
}  // namespace ov