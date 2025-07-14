// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "mvn_config.hpp"
#include "executor_implementation.hpp"
#include "implementations.hpp"

namespace ov::intel_cpu {

template <>
const std::vector<ExecutorImplementation<MVNAttrs>>& getImplementations();

}  // namespace ov::intel_cpu