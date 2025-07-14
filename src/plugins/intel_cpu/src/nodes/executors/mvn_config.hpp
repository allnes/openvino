// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "cpu_types.h"
#include "openvino/core/type/element_type.hpp"
#include "executor_config.hpp"
#include "post_ops.hpp"

namespace ov::intel_cpu {

enum MVNLayoutType {
    mvn_planar,
    mvn_block,
    mvn_by_channel
};

enum MVNEpsMode {
    INSIDE_SQRT,
    OUTSIDE_SQRT
};

struct MVNAttrs {
    MVNLayoutType layout = mvn_planar;
    bool initAcrossChannels_ = false;
    bool execAcrossChannels_ = false;
    bool normalizeVariance_ = true;
    float epsValue_ = 1e-9f;
    MVNEpsMode epsMode_ = INSIDE_SQRT;
    ov::element::Type src_prc = ov::element::f32;
    ov::element::Type dst_prc = ov::element::f32;
    VectorDims shape5D;
    PostOps postOps;
};

struct jit_mvn_config_params {
    MVNLayoutType layout = mvn_planar;
    bool across_channels = false;
    bool normalize_variance = false;
    ov::element::Type src_prc;
    ov::element::Type dst_prc;
    int src_data_size = 0;
    int dst_data_size = 0;
};

struct jit_mvn_call_args {
    const void* src;
    void* dst;
    float* sum;
    float* mean;
    float* variance;
    size_t work_amount;
    size_t oc_off;
    // shape need for shape agnostic kernel passed with each infer.
    // OC for block layout and nspc per channel, tails for ncsp and nspc across channel.
    size_t rt_shape_size;
    const void* post_op_data;
};

using MVNConfig = executor::Config<MVNAttrs>;

}  // namespace ov::intel_cpu