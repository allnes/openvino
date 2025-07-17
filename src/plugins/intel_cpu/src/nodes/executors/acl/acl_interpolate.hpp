// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "arm_compute/runtime/NEON/functions/NEScale.h"
#include "arm_compute/runtime/Tensor.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/interpolate_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "post_ops.hpp"

namespace ov {
namespace intel_cpu {

using PostOpsPtr = std::shared_ptr<PostOps>;

class ACLInterpolateExecutor : public Executor {
public:
    ACLInterpolateExecutor(const InterpolateAttrs& attrs,
                          const PostOpsPtr& postOps,
                          const MemoryArgs& memory,
                          const ExecutorContext::CPtr context);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    
    impl_desc_type implType() const override {
        return impl_desc_type::acl;
    }

private:
    InterpolateAttrs m_attrs;
    arm_compute::SamplingPolicy acl_coord = arm_compute::SamplingPolicy::CENTER;
    arm_compute::InterpolationPolicy acl_policy = arm_compute::InterpolationPolicy::NEAREST_NEIGHBOR;
    arm_compute::Tensor srcTensor, dstTensor;
    std::unique_ptr<arm_compute::NEScale> acl_scale;
    std::vector<uint8_t> m_padded_input;
    const uint8_t* padPreprocess(const MemoryCPtr& src, const MemoryPtr& dst);
};

}  // namespace intel_cpu
}  // namespace ov