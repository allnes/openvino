// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_list.hpp"

namespace ov {
namespace intel_cpu {

const std::vector<SoftMaxExecutorDesc>& getSoftMaxExecutorsList() {
    static std::vector<SoftMaxExecutorDesc> descs = {
//        OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<AclSoftMaxExecutorBuilder>())
        OV_CPU_INSTANCE_DNNL(ExecutorType::Dnnl, std::make_shared<DNNLSoftMaxExecutorBuilder>())
    };

    return descs;
}

}   // namespace intel_cpu
}   // namespace ov