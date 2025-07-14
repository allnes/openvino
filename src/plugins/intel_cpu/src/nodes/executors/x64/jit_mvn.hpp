// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "nodes/executors/mvn.hpp"
#include "nodes/executors/mvn_config.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"

namespace ov::intel_cpu {

// Forward declarations from node namespace
namespace node {
struct jit_uni_mvn_mean_variance_kernel;
struct jit_uni_mvn_kernel;
}

class MVNJitExecutor : public MVNExecutor {
public:
    MVNJitExecutor(const MVNAttrs& mvnAttrs,
                   const MemoryArgs& memory, 
                   const ExecutorContext::CPtr& context);

    bool init(const MVNAttrs& mvnAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr& attr) override;

    void executeImpl(const MemoryArgs& memory) override;

    impl_desc_type getImplType() const override { 
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

private:
    void mvn_pln(const uint8_t* src_data, uint8_t* dst_data, const void* post_ops_data_, const VectorDims& shape5d);
    void mvn_blk(const uint8_t* src_data, uint8_t* dst_data, const void* post_ops_data_, const VectorDims& shape5d);
    void mvn_nspc(const uint8_t* src_data, uint8_t* dst_data, const void* post_ops_data_, const VectorDims& shape5d);

    std::shared_ptr<node::jit_uni_mvn_mean_variance_kernel> mvn_mean_kernel;
    std::shared_ptr<node::jit_uni_mvn_mean_variance_kernel> mvn_variance_kernel;
    std::shared_ptr<node::jit_uni_mvn_kernel> mvn_kernel;
    
    VectorDims shape5D;
    dnnl::primitive_attr attr_;
};

}  // namespace ov::intel_cpu