// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor_factory.hpp"
#include "interpolate_config.hpp"

namespace ov {
namespace intel_cpu {

// Using the modern ExecutorFactory pattern
using InterpolateExecutorFactory = ExecutorFactory<InterpolateAttrs>;
using InterpolateExecutorFactoryPtr = std::shared_ptr<InterpolateExecutorFactory>;
using InterpolateExecutorFactoryCPtr = std::shared_ptr<const InterpolateExecutorFactory>;

}  // namespace intel_cpu
}  // namespace ov