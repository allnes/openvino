// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_mvn.hpp"
#include "nodes/executors/common/ref_mvn.hpp"

#include "utils/cpu_utils.hpp"
#include <memory>

#if defined(OPENVINO_ARCH_X86_64)
#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/plugin/x64/utils.hpp"
#include "nodes/kernels/x64/mlp_utils.hpp"
#include "nodes/kernels/x64/jit_kernel_base.hpp"
#include "utils/debug_capabilities.h"
#endif

using namespace dnnl;
using namespace dnnl::impl::cpu::x64;
using namespace ov;

namespace ov {
namespace intel_cpu {

MVNJitExecutor::MVNJitExecutor(const MVNAttrs& mvnAttrs,
                               const MemoryArgs& memory, 
                               const ExecutorContext::CPtr& context) 
    : attrs(mvnAttrs), memoryArgs(memory), context(context) {
}

bool MVNJitExecutor::init(const MVNAttrs& mvnAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const dnnl::primitive_attr& attr) {
    shape5D = mvnAttrs.shape5D;
    // Simplified JIT executor - just return true to indicate support
    return true;
}

void MVNJitExecutor::executeImpl(const MemoryArgs& memory) {
    // Use reference implementation
    MVNRefExecutor refExecutor(attrs, memory, context);
    refExecutor.executeImpl(memory);
}

bool MVNJitExecutor::canReuseShapeAgnosticKernel(const VectorDims& newShape5D) const {
#if defined(OPENVINO_ARCH_X86_64)
    // Shape-agnostic kernel optimization
    // Reuse kernel if the shape is the same or only batch size changed
    if (shape5D[0] != newShape5D[0]) {
        if (shape5D[1] == newShape5D[1] && shape5D[2] == newShape5D[2] && 
            shape5D[3] == newShape5D[3] && shape5D[4] == newShape5D[4]) {
            shape5D = newShape5D;
            return true;
        }
    }
#endif
    return false;
}

bool MVNJitExecutor::supports(const MVNAttrs& attrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs) {
#if defined(OPENVINO_ARCH_X86_64)
    // Force f32 precision for SSE4.1 due to performance considerations
    if (mayiuse(cpu::x64::sse41) && !mayiuse(cpu::x64::avx2)) {
        if (attrs.src_prc != ov::element::f32 || attrs.dst_prc != ov::element::f32) {
            return false;
        }
    }
#endif
    
    // For simplified version, support all cases that reference implementation supports
    return MVNRefExecutor::supports(attrs, srcDescs, dstDescs);
}

}  // namespace intel_cpu
}  // namespace ov