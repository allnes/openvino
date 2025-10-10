// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "acl_common_executor.hpp"
#include "nodes/executors/convolution_config.hpp"

namespace ov::intel_cpu {

// ACL-based 3D Convolution executor using NEConv3D (NEON)
class ACLConv3DExecutor : public ACLCommonExecutor {
public:
    ACLConv3DExecutor(const ConvAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context);

    static bool supports(const ConvConfig& config);
    void updateTensorsShapes(ACLShapes& aclMemoryShapes) override;

    arm_compute::Status validateTensorsInfo(const ACLInfos& aclMemoryInfos) override;
    ACLFunction configureFunction(const ACLTensors& aclMemoryTensors) override;

private:
    // 3D conv parameters
    arm_compute::Conv3dInfo m_conv3d_info{};
};

using ACLConv3DExecutorPtr = std::shared_ptr<ACLConv3DExecutor>;

}  // namespace ov::intel_cpu
