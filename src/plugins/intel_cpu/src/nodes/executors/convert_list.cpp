// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_list.hpp"

namespace ov {
namespace intel_cpu {

const std::vector<ConvertExecutorDesc>& getConvertExecutorsList() {
    static std::vector<ConvertExecutorDesc> descs = {
//            OV_CPU_INSTANCE_X64(ExecutorType::x64, std::make_shared<JitConvertExecutorBuilder>())
            OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<ACLConvertExecutorBuilder>())
//            OV_CPU_INSTANCE_COMMON(ExecutorType::Common, std::make_shared<RefConvertExecutorBuilder>())
    };

    return descs;
}

}   // namespace intel_cpu
}   // namespace ov