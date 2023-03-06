// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose_list.hpp"

namespace ov {
namespace intel_cpu {

const std::vector<TransposeExecutorDesc>& getTransposeExecutorsList() {
    static std::vector<TransposeExecutorDesc> descs = {
            OV_CPU_INSTANCE_COMMON(ExecutorType::Common, std::make_shared<RefTransposeExecutorBuilder>())
    };

    return descs;
}

}   // namespace intel_cpu
}   // namespace ov