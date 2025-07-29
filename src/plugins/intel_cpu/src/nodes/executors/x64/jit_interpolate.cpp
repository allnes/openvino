// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_interpolate.hpp"
#include "jit_interpolate_legacy.hpp"
#include "nodes/interpolate.h"
#include "memory_desc/cpu_memory_desc.h"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"
#include "cpu_types.h"
#include "common/dnnl_thread.hpp"
#include "dnnl_postops_composer.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "dnnl_extension_utils.h"
#include "cpu_memory.h"

namespace ov {
namespace intel_cpu {

using namespace node;

JitInterpolateExecutor::JitInterpolateExecutor(const InterpolateAttrs& attrs,
                                               const PostOpsPtr& postOps,
                                               const MemoryArgs& memory,
                                               const ExecutorContext::CPtr context)
    : attrs(attrs), postOps(postOps), memoryArgs(memory), context(context), legacyExecutor(nullptr, [](void*){}) {
    
    // Determine implementation type based on available ISA
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
        m_implType = impl_desc_type::jit_avx512;
    } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
        m_implType = impl_desc_type::jit_avx2;
    } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41)) {
        m_implType = impl_desc_type::jit_sse42;
    } else {
        m_implType = impl_desc_type::jit_uni;
    }
    
    // Extract dimensions from memory descriptors
    const auto& srcDesc = memory.at(ARG_SRC)->getDescPtr();
    const auto& dstDesc = memory.at(ARG_DST)->getDescPtr();
    
    srcDims = srcDesc->getShape().getStaticDims();
    dstDims = dstDesc->getShape().getStaticDims();
    
    // Apply NCHWAsNHWC transformation if needed (following upstream master pattern)
    // Note: NCHWAsNHWC is false when hasPad is true (see interpolate.cpp line 2137)
    if (attrs.NCHWAsNHWC && srcDesc->hasLayoutType(LayoutType::ncsp)) {
        auto logicalShapeAlign = [](VectorDims& Dims) {
            size_t C = Dims[3];
            Dims[3] = Dims[2];
            Dims[2] = Dims[1];
            Dims[1] = C;
        };
        logicalShapeAlign(srcDims);
        logicalShapeAlign(dstDims);
    }
    
    // Note: Do NOT apply padding to srcDims here - InterpolateExecutorBase constructor
    // will call getPaddedInputShape() to apply padding when creating srcDimPad5d
    
    // Use dataScales from attrs if provided, otherwise calculate them
    // Note: attrs.dataScales are pre-calculated with padded dimensions by the node
    if (attrs.dataScales.empty()) {
        // This should not happen as the node always provides dataScales, but handle it just in case
        // Calculate scales from unpadded dimensions (padding will be applied by base class)
        dataScales.resize(srcDims.size());
        for (size_t i = 0; i < srcDims.size(); i++) {
            dataScales[i] = static_cast<float>(dstDims[i]) / static_cast<float>(srcDims[i]);
        }
    } else {
        // Use pre-calculated scales
        dataScales = attrs.dataScales;
    }
    
    // Create primitive attributes for legacy executor
    dnnl::primitive_attr attr;
    setPostOps(attr, false);
    
    // Create legacy executor with transformed dimensions
    // Note: Legacy executor's base class will apply padding internally via getPaddedInputShape()
    auto* legacyExec = new legacy::InterpolateJitExecutor(
        attrs, 
        srcDims,  // Unpadded dims - base class will pad them
        dstDims, 
        dataScales,
        attr
    );
    
    legacyExecutor = std::unique_ptr<void, void(*)(void*)>(
        legacyExec, 
        [](void* p) { delete static_cast<legacy::InterpolateJitExecutor*>(p); }
    );
}

bool JitInterpolateExecutor::update(const MemoryArgs& memory) {
    // For now, we don't support dynamic shapes, so return false
    return false;
}

void JitInterpolateExecutor::execute(const MemoryArgs& memory) {
    const auto& srcMem = memory.at(ARG_SRC);
    const auto& dstMem = memory.at(ARG_DST);
    
    const uint8_t* srcData = srcMem->getDataAs<const uint8_t>();
    uint8_t* dstData = dstMem->getDataAs<uint8_t>();
    
    // Prepare PostOps data following MVN pattern
    const void* postOpsData = postOpsPtrArray.empty() ? nullptr : 
        static_cast<const void*>(postOpsPtrArray.data());
    
    // Call legacy executor
    auto* executor = static_cast<legacy::InterpolateJitExecutor*>(legacyExecutor.get());
    executor->exec(srcData, dstData, postOpsData);
}

void JitInterpolateExecutor::setPostOps(dnnl::primitive_attr& attr, bool /*initWeights*/) {
    if (!postOps || postOps->empty()) {
        return;
    }
    
    // Use DnnlPostOpsComposer to convert PostOps to dnnl format
    VectorDims outputDims = dstDims;
    size_t idxOC = attrs.layout == InterpolateLayoutType::by_channel ? outputDims.size() - 1 : 1;
    
    const bool isINT8 =
        (attrs.inPrc == ov::element::i8 || attrs.inPrc == ov::element::u8) && attrs.outPrc == ov::element::i8;
    const auto outDataType = DnnlExtensionUtils::ElementTypeToDataType(attrs.outPrc);
    
    // Create memory args for post-ops composer
    MemoryArgs postOpsMemoryArgs = memoryArgs;
    // Create a dummy empty bias memory descriptor if not present
    if (postOpsMemoryArgs.count(ARG_BIAS) == 0) {
        auto biasDesc = std::make_shared<CpuBlockedMemoryDesc>(ov::element::f32, Shape{});
        postOpsMemoryArgs[ARG_BIAS] = std::make_shared<Memory>(context->getEngine(), biasDesc);
    }
    
    DnnlPostOpsComposer composer(*postOps,
                                 context->getEngine(),
                                 outputDims,
                                 idxOC,
                                 isINT8,
                                 1 << 0,  // weight scale mask per channel
                                 postOpsMemoryArgs,
                                 outDataType,
                                 {},     // legacy DQ scales
                                 true,   // use legacy post ops for compatibility
                                 true);  // use legacy zero points
    
    auto primAttrs = composer.compose();
    attr = primAttrs.attr;
    
    // Clear previous data
    postOpsDataPtrs.clear();
    postOpsDataBuffer.clear();
    postOpsPtrArray.clear();
    postOpsMemory.clear();
    
    // Collect all post-ops data memory from DnnlPostOpsComposer
    for (const auto& cpuArg : primAttrs.cpuArgs) {
        // Check if this is post-op data
        if (cpuArg.first >= DNNL_ARG_ATTR_MULTIPLE_POST_OP(0)) {
            // Keep the memory alive by storing the MemoryPtr
            postOpsMemory.push_back(cpuArg.second);
            
            const auto* memPtr = cpuArg.second.get();
            if (memPtr && memPtr->getData()) {
                postOpsDataPtrs.push_back(memPtr->getData());
            }
        }
    }
    
    // Create the pointer array that legacy executor expects
    if (!postOpsDataPtrs.empty()) {
        postOpsPtrArray.clear();
        
        // For each post-op data pointer, add it to the array
        for (const auto& ptr : postOpsDataPtrs) {
            postOpsPtrArray.push_back(const_cast<void*>(ptr));
        }
    }
}

bool jitInterpolateSupported(const InterpolateAttrs& config, const MemoryDescArgs& descs) {
    // Check if x64 JIT is available
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41)) {
        return false;
    }
    
    // Check supported modes - only linear mode is not supported by JIT
    // The legacy JIT implementation supports: nearest, linear_onnx, cubic, bilinear_pillow, bicubic_pillow
    if (config.mode == InterpolateMode::linear) {
        return false;  // Not supported by JIT (assertion in legacy implementation)
    }
    
    // Check rank
    const auto& srcDesc = descs.at(ARG_SRC);
    const auto dataRank = srcDesc->getShape().getRank();
    
    if (!any_of(dataRank, 4u, 5u) && !dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
        return false;
    }
    
    // Check precision
    const auto srcPrecision = srcDesc->getPrecision();
    const auto dstPrecision = descs.at(ARG_DST)->getPrecision();
    
    if (srcPrecision != dstPrecision) {
        return false;
    }
    
    if (!any_of(srcPrecision, ov::element::f32, ov::element::bf16, ov::element::f16,
                ov::element::i8, ov::element::u8)) {
        return false;
    }
    
    // Check layout
    if (config.layout != InterpolateLayoutType::planar && 
        config.layout != InterpolateLayoutType::block &&
        config.layout != InterpolateLayoutType::by_channel) {
        return false;
    }
    
    return true;
}

}  // namespace intel_cpu
}  // namespace ov