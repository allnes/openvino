// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "mvn_config.hpp"
#include "onednn/iml_type_mapper.h"
#include "executor.hpp"
#include "memory_arguments.hpp"

namespace ov::intel_cpu {

class MVNExecutor : public Executor {
public:
    MVNExecutor(const MVNAttrs& mvnAttrs, 
                const MemoryArgs& memory,
                const ExecutorContext::CPtr& context);
    virtual ~MVNExecutor() = default;
    
    virtual bool init(const MVNAttrs& mvnAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr& attr) = 0;
                      
    bool update(const MemoryArgs& memory) override {
        memoryArgs = memory;
        return true;
    }
    
    void execute() override {
        executeImpl(memoryArgs);
    }
    
    virtual void executeImpl(const MemoryArgs& memory) = 0;
    
    impl_desc_type implType() const override {
        return getImplType();
    }
    
    virtual impl_desc_type getImplType() const = 0;
    
    static VectorDims transformTo5DCase(const VectorDims& shape, bool initAcrossChannels);

protected:
    MVNAttrs attrs;
    MemoryArgs memoryArgs;
    const ExecutorContext::CPtr context;
};

using MVNExecutorPtr = std::shared_ptr<MVNExecutor>;
using MVNExecutorCPtr = std::shared_ptr<const MVNExecutor>;

}  // namespace ov::intel_cpu
