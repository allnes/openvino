// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_mvn.hpp"

#include <cpu/x64/xbyak/xbyak.h>

#include <any>
#include <common/c_types_map.hpp>
#include <common/primitive_hashing_utils.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <functional>
#include <memory>

#include "cpu/x64/injectors/jit_uni_depthwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_quantization_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/plugin/x64/jit_load_store_emitters.hpp"
#include "emitters/plugin/x64/utils.hpp"
#include "nodes/executors/common/ref_mvn.hpp"
#include "nodes/kernels/x64/jit_kernel_base.hpp"
#include "nodes/kernels/x64/mlp_utils.hpp"
#include "openvino/core/parallel.hpp"
#include "post_ops.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/debug_capabilities.h"

using namespace dnnl;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;
using namespace ov;

namespace ov {
namespace intel_cpu {

namespace {

struct MVNKey {
    MVNAttrs mvnAttrs;
    dnnl::primitive_attr attr;

    [[nodiscard]] size_t hash() const;
    bool operator==(const MVNKey& rhs) const;
};

size_t MVNKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    seed = hash_combine(seed, mvnAttrs.initAcrossChannels_);
    seed = hash_combine(seed, mvnAttrs.execAcrossChannels_);
    seed = hash_combine(seed, mvnAttrs.normalizeVariance_);
    seed = hash_combine(seed, mvnAttrs.epsValue_);
    seed = hash_combine(seed, mvnAttrs.epsMode_);
    seed = hash_combine(seed, mvnAttrs.src_prc.hash());
    seed = hash_combine(seed, mvnAttrs.dst_prc.hash());
    seed = hash_combine(seed, mvnAttrs.layout);
    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    return seed;
}

bool MVNKey::operator==(const MVNKey& rhs) const {
    bool retVal = true;
    retVal = retVal && mvnAttrs.initAcrossChannels_ == rhs.mvnAttrs.initAcrossChannels_ &&
             mvnAttrs.execAcrossChannels_ == rhs.mvnAttrs.execAcrossChannels_ &&
             mvnAttrs.normalizeVariance_ == rhs.mvnAttrs.normalizeVariance_ &&
             mvnAttrs.epsValue_ == rhs.mvnAttrs.epsValue_ && mvnAttrs.epsMode_ == rhs.mvnAttrs.epsMode_ &&
             mvnAttrs.src_prc == rhs.mvnAttrs.src_prc && mvnAttrs.dst_prc == rhs.mvnAttrs.dst_prc &&
             mvnAttrs.layout == rhs.mvnAttrs.layout;
    retVal = retVal && *attr.get() == *rhs.attr.get();
    return retVal;
}

}  // namespace

MVNJitExecutor::MVNJitExecutor(const MVNAttrs& mvnAttrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context)
    : attrs(mvnAttrs),
      memoryArgs(memory),
      context(context),
      shape5D(mvnAttrs.shape5D) {
    legacyJitExecutor = std::make_shared<legacy::MVNJitExecutorLagacy>(attrs, dnnl::primitive_attr());
}

bool MVNJitExecutor::init(const MVNAttrs& mvnAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const dnnl::primitive_attr& attr) {
    shape5D = mvnAttrs.shape5D;
    attrs = mvnAttrs;

    // Extract post-ops data from MVNAttrs
    postOpsDataPtrs.clear();

    // MVNAttrs.postOps contains the post-ops data pointers
    // Each element is a std::any containing a const void* pointer
    for (const auto& postOp : attrs.postOps) {
        // Extract the data pointer from std::any
        if (postOp.type() == typeid(const void*)) {
            postOpsDataPtrs.push_back(std::any_cast<const void*>(postOp));
        }
    }

    // Create a key for caching
    MVNKey key{attrs, attr};

    auto builder = [&](const MVNKey& key) -> std::shared_ptr<legacy::MVNJitExecutorLagacy> {
        return std::make_shared<legacy::MVNJitExecutorLagacy>(key.mvnAttrs, key.attr);
    };

    // Use context's cache if available
    if (context) {
        auto cache = context->getRuntimeCache();
        auto result = cache->getOrCreate(key, builder);
        legacyJitExecutor = result.first;
    } else {
        // Fallback if no context available
        legacyJitExecutor = builder(key);
    }

    return true;
}

void MVNJitExecutor::executeImpl(const MemoryArgs& memory) {
    // Extract memory pointers from MemoryArgs
    const auto* src_data = memory.at(ARG_SRC)->getDataAs<const uint8_t>();
    auto* dst_data = memory.at(ARG_DST)->getDataAs<uint8_t>();

    // Pass post-ops data to the legacy executor
    const void* post_ops_data = nullptr;
    if (!postOpsDataPtrs.empty()) {
        post_ops_data = postOpsDataPtrs.data();
    }

    // Call legacy executor with proper parameters
    legacyJitExecutor->exec(src_data, dst_data, post_ops_data, shape5D);
}

bool MVNJitExecutor::canReuseShapeAgnosticKernel(const VectorDims& newShape5D) const {
    // Shape-agnostic kernel optimization
    // Reuses kernel if the shape is the same or only batch size changed
    if (shape5D[0] != newShape5D[0]) {
        if (shape5D[1] == newShape5D[1] && shape5D[2] == newShape5D[2] && shape5D[3] == newShape5D[3] &&
            shape5D[4] == newShape5D[4]) {
            shape5D = newShape5D;
            return true;
        }
    }
    return false;
}

bool MVNJitExecutor::supports(const MVNConfig& config) {
    // JIT implementation supports all precisions
    // The legacy implementation handles precision conversions internally
    return true;
}

}  // namespace intel_cpu
}  // namespace ov