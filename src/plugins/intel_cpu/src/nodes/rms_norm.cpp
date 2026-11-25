// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rms_norm.h"

#include <cmath>
#include <common/utils.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "node.h"
#include "nodes/eltwise.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "ov_ops/rms.hpp"
#include "shape_inference/custom/rms_norm.hpp"
#include "utils/general_utils.h"
#ifdef OPENVINO_ARCH_X86_64
#    include "cpu/x64/cpu_isa_traits.hpp"
#    include "cpu_parallel.hpp"
#    include "kernels/x64/rms_kernel.hpp"
#    include "nodes/kernels/x64/jit_kernel_base.hpp"
#endif

#include <string>
#include <vector>

using namespace dnnl::impl;

namespace ov::intel_cpu::node {

struct RMSNormKey {
    ov::element::Type source_precision;
    ov::element::Type destination_precision;
    size_t data_size;
    size_t scale_size;
    float eps;
    float output_scale;
    [[nodiscard]] size_t hash() const;
    bool operator==(const RMSNormKey& rhs) const;
};

size_t RMSNormKey::hash() const {
    size_t seed = 0;
    seed = hash_combine(seed, source_precision.hash());
    seed = hash_combine(seed, destination_precision.hash());
    seed = hash_combine(seed, data_size);
    seed = hash_combine(seed, scale_size);
    seed = hash_combine(seed, eps);
    seed = hash_combine(seed, output_scale);

    return seed;
}

bool RMSNormKey::operator==(const RMSNormKey& rhs) const {
    auto retVal = source_precision == rhs.source_precision && destination_precision == rhs.destination_precision &&
                  data_size == rhs.data_size && scale_size == rhs.scale_size && eps == rhs.eps &&
                  output_scale == rhs.output_scale;

    return retVal;
}

#if defined(OPENVINO_ARCH_X86_64)
static std::shared_ptr<kernel::JitKernelBase> createJitKernel(const kernel::jit_rms_compile_params& param) {
    std::shared_ptr<kernel::JitKernelBase> res;

    if (cpu::x64::mayiuse(cpu::x64::avx512_core)) {
        res = std::make_shared<kernel::jit_rms_kernel<dnnl::impl::cpu::x64::avx512_core>>(param);
    } else if (cpu::x64::mayiuse(cpu::x64::avx2)) {
        res = std::make_shared<kernel::jit_rms_kernel<dnnl::impl::cpu::x64::avx2>>(param);
    }

    if (res) {
        res->create_kernel();
    }

    return res;
}

static void execJitKernel(const std::shared_ptr<kernel::JitKernelBase>& ker,
                          const uint8_t* src,
                          uint8_t* dst,
                          const float* scale) {
    kernel::jit_rms_call_args call_args{src, scale, dst};
    (*ker)(&call_args);
}

struct RMSNorm::RMSNormExecutor : public RMSNorm::Executor {
    explicit RMSNormExecutor(const RMSNormKey& key)
        : m_source_precision(key.source_precision),
          m_destination_precision(key.destination_precision) {
        kernel::jit_rms_compile_params jcp;
        jcp.src_prc = key.source_precision;
        jcp.dst_prc = key.destination_precision;
        jcp.data_size = key.data_size;
        jcp.scale_size = key.scale_size;
        jcp.eps = key.eps;
        jcp.output_scale = key.output_scale;
        m_kernel = createJitKernel(jcp);
    }
    void execute(const std::vector<MemoryPtr>& inputs,
                 const MemoryPtr output,
                 const CpuParallelPtr& cpu_parallel) override {
        static constexpr float identity_scale = 1.0F;
        auto* src = inputs[0]->getDataAs<uint8_t>();
        auto* dst = output->getDataAs<uint8_t>();
        const auto* scale = inputs.size() > 1 ? inputs[1]->getDataAs<float>() : &identity_scale;

        const auto& src_strides = inputs[0]->getDescWithType<BlockedMemoryDesc>()->getStrides();
        const auto& dst_strides = output->getDescWithType<BlockedMemoryDesc>()->getStrides();
        const auto& shape = inputs[0]->getStaticDims();
        const auto src_stride = src_strides[src_strides.size() - 2] * m_source_precision.size();
        const auto dst_stride = dst_strides[dst_strides.size() - 2] * m_destination_precision.size();
        auto n = shape_size(shape) / shape[shape.size() - 1];
        cpu_parallel->parallel_for(n, [&](size_t i) {
            execJitKernel(m_kernel, src + i * src_stride, dst + i * dst_stride, scale);
        });
    }

private:
    ov::element::Type m_source_precision;
    ov::element::Type m_destination_precision;
    std::shared_ptr<kernel::JitKernelBase> m_kernel;
};
#endif  // OPENVINO_ARCH_X86_64

RMSNorm::RMSNorm(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, RMSNormShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto rms = ov::as_type_ptr<const ov::op::internal::RMS>(op);
    m_eps = static_cast<float>(rms->get_epsilon());
}

bool RMSNorm::canFuse(const NodePtr& node) const {
    // A reduced-precision RMS output must round before the next operation.
    if (getOriginalOutputPrecisionAtPort(0) != ov::element::f32 || !fusedWith.empty() ||
        node->getAlgorithm() != Algorithm::EltwisePowerStatic || !node->getFusedWith().empty() ||
        node->getOriginalInputPrecisionAtPort(0) != ov::element::f32 ||
        none_of(node->getOriginalOutputPrecisionAtPort(0), ov::element::f32, ov::element::f16, ov::element::bf16)) {
        return false;
    }
    const auto eltwise = std::dynamic_pointer_cast<Eltwise>(node);
    return eltwise && eltwise->getAlpha() == 1.0 && eltwise->getGamma() == 0.0 && eltwise->getBeta() != 0.0 &&
           !std::signbit(eltwise->getGamma()) && std::isfinite(eltwise->getBeta());
}

void RMSNorm::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    auto precision = getOriginalInputPrecisionAtPort(0);
    if (none_of(precision, ov::element::f32, ov::element::bf16, ov::element::f16)) {
        precision = ov::element::f32;
    }
    auto output_precision =
        fusedWith.empty() ? getOriginalOutputPrecisionAtPort(0) : fusedWith.back()->getOriginalOutputPrecisionAtPort(0);
    if (none_of(output_precision, ov::element::f32, ov::element::bf16, ov::element::f16)) {
        output_precision = ov::element::f32;
    }

    auto impl_type = impl_desc_type::ref;
#if defined(OPENVINO_ARCH_X86_64)
    if (cpu::x64::mayiuse(cpu::x64::avx512_core)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (cpu::x64::mayiuse(cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    }
#endif

    std::vector<PortConfigurator> inPortConfigs{{LayoutType::ncsp, precision}};
    if (getOriginalInputsNumber() > 1) {
        inPortConfigs.emplace_back(LayoutType::ncsp, ov::element::f32);
    }
    addSupportedPrimDesc(inPortConfigs, {{LayoutType::ncsp, output_precision}}, impl_type);
}

void RMSNorm::createPrimitive() {
    const auto source_precision = getSrcMemoryAtPort(0)->getPrecision();
    const auto destination_precision = getDstMemoryAtPort(0)->getPrecision();
    auto data_dims = getSrcMemoryAtPort(0)->getDescWithType<BlockedMemoryDesc>()->getBlockDims();
    size_t data_size = data_dims[data_dims.size() - 1];
    size_t scale_size = getOriginalInputsNumber() > 1
                            ? shape_size(getSrcMemoryAtPort(1)->getDescWithType<BlockedMemoryDesc>()->getBlockDims())
                            : 1;

    const float output_scale =
        fusedWith.empty() ? 1.F : static_cast<float>(std::dynamic_pointer_cast<Eltwise>(fusedWith.front())->getBeta());
    RMSNormKey key = {source_precision, destination_precision, data_size, scale_size, m_eps, output_scale};

    auto builder = [&]([[maybe_unused]] const RMSNormKey& key) -> std::shared_ptr<Executor> {
#ifdef OPENVINO_ARCH_X86_64
        return std::make_shared<RMSNormExecutor>(key);
#else
        return nullptr;
#endif
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);
    OPENVINO_ASSERT(result.first,
                    "RMSNorm Executor creation fails with precisions " + source_precision.to_string() + " -> " +
                        destination_precision.to_string());
    m_executor = result.first;
}

void RMSNorm::execute([[maybe_unused]] const dnnl::stream& strm) {
    auto orginInputNumber = getOriginalInputsNumber();
    std::vector<MemoryPtr> inputs(orginInputNumber);

    for (size_t i = 0; i < orginInputNumber; i++) {
        inputs[i] = getSrcMemoryAtPort(i);
    }

    m_executor->execute(inputs, getDstMemoryAtPort(0), context->getCpuParallel());
}

bool RMSNorm::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto rms = ov::as_type_ptr<const ov::op::internal::RMS>(op);
        if (rms) {
#if defined(OPENVINO_ARCH_X86_64)
            const bool has_avx2 = cpu::x64::mayiuse(cpu::x64::avx2);
#else
            constexpr bool has_avx2 = false;
#endif
            if (!has_avx2) {
                errorMessage = "RMSNorm needs avx2+.";
                return false;
            }
            // check the last dimension of data
            auto data_pshape = op->input_value(0).get_partial_shape();
            if (data_pshape.rank().is_dynamic()) {
                errorMessage = "RMSNorm data rank is not static.";
                return false;
            }
            const auto& data_rank = op->get_input_partial_shape(0).rank().get_length();
            if (data_rank <= 1) {
                errorMessage = "RMSNorm data rank must be greater than 1.";
                return false;
            }
            if (data_pshape[data_rank - 1].is_dynamic()) {
                errorMessage = "RMSNorm last dimension of data is not static.";
                return false;
            }

            // check scale
            if (op->get_input_size() > 1) {
                if (op->get_input_partial_shape(1).is_dynamic()) {
                    errorMessage = "RMSNorm scale shape is not static.";
                    return false;
                }
                auto scale_pshape = op->get_input_partial_shape(1);
                if (scale_pshape.rank().get_length() > 1) {
                    for (int64_t i = 0; i < scale_pshape.rank().get_length() - 1; i++) {
                        if (scale_pshape[i] != 1) {
                            errorMessage = "RMSNorm scale shape must be [1,..., N].";
                            return false;
                        }
                    }
                }
            }
        } else {
            errorMessage = "Only RMSNorm operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node
