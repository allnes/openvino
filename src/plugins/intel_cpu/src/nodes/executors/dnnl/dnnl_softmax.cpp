// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_softmax.hpp"
#include <string>
#include <dnnl_types.h>
#include <dnnl_extension_utils.h>
#include <dnnl_descriptor.h>
#include <memory_desc/cpu_memory_desc_utils.h>
#include <ngraph/opsets/opset1.hpp>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include <common/primitive_hashing_utils.hpp>
#include <utils/shape_inference/shape_inference_pass_through.hpp>

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {

DNNLSoftMaxExecutor::DNNLSoftMaxExecutor(const ExecutorContext::CPtr context) : SoftMaxExecutor(context) {}

bool DNNLSoftMaxExecutor::init(const SoftMaxAttrs &softMaxAttrs, const std::vector<MemoryDescPtr> &srcDescs,
                               const std::vector<MemoryDescPtr> &dstDescs, const dnnl::primitive_attr &attr) {
    dnnlStream = dnnl::engine(softMaxContext->getEngine());
    auto engine = softMaxContext->getEngine();
    auto localAttrs = dnnl::primitive_attr(attr.get()->clone());

    softmax_forward::primitive_desc prim_desc;
    DnnlDesriptor desc(std::shared_ptr<softmax_forward::desc>(
            new softmax_forward::desc(prop_kind::forward_scoring, softMaxAttrs.inp0->getDnnlDesc(), softMaxAttrs.axis)));
    localAttrs.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    primitive_desc_iterator itpd = desc.createPrimitiveDescriptorIterator(engine, localAttrs);

    bool isExecutable = true;
    while (itpd) {
        impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());
        if (impl_type == softMaxAttrs.implDescType ||
            // At least for oneDNN v2.4 the softmax primitive is optimized for the cases where the dimension of the
            // softmax axis is physically dense. There could be situations where it is not possible to detect the
            // optimized case in advance in case of dynamic shapes, but in runtime the shape could be suitable for
            // the optimized implementation, so we have to select the optimized one.
            (ref_any == softMaxAttrs.implDescType && (impl_type & jit))) {
            prim_desc = itpd.get();
            break;
        }
        if (!itpd.next_impl()) {
            isExecutable = false;
            softMaxPrim = nullptr;
            break;
        }
    }

    if (isExecutable) {
        softMaxPrim = std::make_shared<softmax_forward>(prim_desc);
    }

    auto scratchpadMemoryDesc = DnnlExtensionUtils::makeDescriptor(prim_desc.query_md(dnnl::query::scratchpad_md));
    scratchpadMemory = softMaxContext->getScratchPad()->createScratchPadMem(scratchpadMemoryDesc);
    return true;
}

void DNNLSoftMaxExecutor::exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst,
                               const void **post_ops_data_) {
    std::unordered_map<int, dnnl::memory> primArgs;
    primArgs[DNNL_ARG_SRC] = src[0]->GetPrimitive();
    primArgs[DNNL_ARG_DST] = dst[0]->GetPrimitive();
    primArgs[DNNL_ARG_SCRATCHPAD] = scratchpadMemory->GetPrimitive();

    if (softMaxPrim) {
        (*softMaxPrim).execute(dnnlStream, primArgs);
    }
}

}   // namespace intel_cpu
}   // namespace ov
