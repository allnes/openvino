// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/executor.hpp"
#include "nodes/executors/interpolate_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "post_ops.hpp"

namespace ov {
namespace intel_cpu {

namespace legacy {
class InterpolateJitExecutor;
}

using PostOpsPtr = std::shared_ptr<PostOps>;

class JitInterpolateExecutor : public Executor {
public:
    JitInterpolateExecutor(const InterpolateAttrs& attrs,
                          const PostOpsPtr& postOps,
                          const MemoryArgs& memory,
                          const ExecutorContext::CPtr context);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    
    impl_desc_type implType() const override {
        return m_implType;
    }

private:
    InterpolateAttrs attrs;
    PostOpsPtr postOps;
    VectorDims srcDims;
    VectorDims dstDims;
    std::vector<float> dataScales;
    impl_desc_type m_implType;
    
    // PostOps handling (following MVN pattern)
    std::vector<const void*> postOpsDataPtrs;
    std::vector<uint8_t> postOpsDataBuffer;
    std::vector<void*> postOpsPtrArray;
    std::vector<MemoryPtr> postOpsMemory;
    MemoryArgs memoryArgs;
    const ExecutorContext::CPtr context;
    
    // Legacy executor instance with custom deleter
    std::unique_ptr<void, void(*)(void*)> legacyExecutor;
    
    void setPostOps(dnnl::primitive_attr& attr, bool initWeights = false);
};

bool jitInterpolateSupported(const InterpolateAttrs& config, const MemoryDescArgs& descs);

}  // namespace intel_cpu
}  // namespace ov