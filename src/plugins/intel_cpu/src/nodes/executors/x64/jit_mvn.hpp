// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "nodes/executors/executor.hpp"
#include "nodes/executors/mvn_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"

namespace ov::intel_cpu {

class MVNJitExecutor : public Executor {
public:
    MVNJitExecutor(const MVNAttrs& mvnAttrs,
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

    impl_desc_type implType() const override { 
        // Return specific ISA implementation type based on runtime capabilities
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
            return impl_desc_type::jit_avx512;
        } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
            return impl_desc_type::jit_avx2;
        } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41)) {
            return impl_desc_type::jit_sse42;
        }
        return impl_desc_type::ref;
    }
    
    static bool supports(const MVNAttrs& attrs,
                        const std::vector<MemoryDescPtr>& srcDescs,
                        const std::vector<MemoryDescPtr>& dstDescs);

    bool canReuseShapeAgnosticKernel(const VectorDims& newShape5D) const;

private:
    MVNAttrs attrs;
    MemoryArgs memoryArgs;
    const ExecutorContext::CPtr context;
    mutable VectorDims shape5D;
};

}  // namespace ov::intel_cpu