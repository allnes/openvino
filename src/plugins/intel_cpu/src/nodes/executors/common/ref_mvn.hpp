// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "nodes/executors/executor.hpp"
#include "nodes/executors/mvn_config.hpp"
#include "nodes/executors/memory_arguments.hpp"

namespace ov::intel_cpu {

class MVNRefExecutor : public Executor {
public:
    MVNRefExecutor(const MVNAttrs& mvnAttrs,
                   const MemoryArgs& memory,
                   const ExecutorContext::CPtr& context);

    bool init(const MVNAttrs& mvnAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr& attr);

    bool update(const MemoryArgs& memory) override {
        memoryArgs = memory;
        return true;
    }
    
    void execute() override {
        executeImpl(memoryArgs);
    }
    
    void execute(const MemoryArgs& memory) override {
        executeImpl(memory);
    }

    void executeImpl(const MemoryArgs& memory);

    impl_desc_type implType() const override { return impl_desc_type::ref; }
    
    static bool supports(const MVNAttrs& attrs,
                        const std::vector<MemoryDescPtr>& srcDescs,
                        const std::vector<MemoryDescPtr>& dstDescs);

private:
    void mvn_ref(const uint8_t* src_data, uint8_t* dst_data, const VectorDims& shape5d);

    MVNAttrs attrs;
    MemoryArgs memoryArgs;
    const ExecutorContext::CPtr context;
    size_t src_data_size = 0;
    size_t dst_data_size = 0;
    VectorDims shape5D;
};

}  // namespace ov::intel_cpu