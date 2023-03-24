// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax.hpp"
#include <string>
#include <dnnl_types.h>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include <common/primitive_hashing_utils.hpp>

namespace ov {
namespace intel_cpu {

size_t SoftmaxKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    seed = hash_combine(seed, get_md_hash(inp0->getDnnlDesc().data));
    seed = hash_combine(seed, implType);
    seed = hash_combine(seed, axis);
    return seed;
}

bool SoftmaxKey::operator==(const SoftmaxKey& rhs) const {
    bool retVal = true;
    if (inp0 != rhs.inp0) {
        retVal = retVal && inp0 && rhs.inp0 && inp0->getDnnlDesc() == rhs.inp0->getDnnlDesc();
    }

    retVal = retVal && implType == rhs.implType && axis == rhs.axis;
    return retVal;
}

SoftMaxExecutor::SoftMaxExecutor(const ExecutorContext::CPtr context) : softMaxContext(context) {}

}   // namespace intel_cpu
}   // namespace ov
