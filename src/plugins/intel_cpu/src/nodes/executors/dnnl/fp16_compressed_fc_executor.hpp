// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "dnnl_extension_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_matmul_primitive.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/reorder.h"
#include "thread_pool_imp.hpp"

namespace ov::intel_cpu::fp16_compressed_fc {
inline bool plain(const MemoryPtr& memory) {
    auto md = MemoryDescUtils::convertToDnnlMemoryDesc(memory->getDescPtr())->getDnnlDesc();
    if (md.get_format_kind() != dnnl::memory::format_kind::blocked || md.get_inner_nblks() != 0 ||
        md.get_submemory_offset() != 0) {
        return false;
    }
    auto dims = md.get_dims();
    auto strides = md.get_strides();
    if (dims != md.get_padded_dims()) {
        return false;
    }
    for (auto offset : md.get_padded_offsets()) {
        if (offset != 0) {
            return false;
        }
    }
    int64_t stride = 1;
    for (size_t i = dims.size(); i-- > 0;) {
        if (strides[i] != stride) {
            return false;
        }
        stride *= dims[i];
    }
    return true;
}
struct Params {
    MemoryPtr memory;
    size_t channels = 1, groups = 1;
    explicit Params(const MemoryPtr& value, size_t n, size_t k) : memory(value) {
        OPENVINO_ASSERT(plain(value));
        auto dims = value->getShape().getStaticDims();
        while (dims.size() > 2 && dims.front() == 1) {
            dims.erase(dims.begin());
        }
        if (dims.size() == 1 && dims.front() == 1) {
            dims.push_back(1);
        }
        OPENVINO_ASSERT(dims.size() == 2 && dims[0] > 0 && dims[1] > 0);
        channels = dims[0];
        groups = dims[1];
        OPENVINO_ASSERT(n % channels == 0 && k % groups == 0);
    }
};
class Fp16CompressedFCExecutor final : public Executor {
public:
    Fp16CompressedFCExecutor(FCAttrs attrs, ExecutorContext::CPtr context)
        : m_attrs(std::move(attrs)),
          m_context(std::move(context)) {}

    bool update(const MemoryArgs& memory) override {
        if (!m_primitive || !m_src_desc->isCompatible(memory.at(ARG_SRC)->getDesc()) ||
            !m_dst_desc->isCompatible(memory.at(ARG_DST)->getDesc())) {
            initialize(memory);
        }
        return true;
    }

    void execute(const MemoryArgs& memory) override {
        m_args[DNNL_ARG_SRC].set_data_handle(memory.at(ARG_SRC)->getData());
        m_args[DNNL_ARG_DST].set_data_handle(memory.at(ARG_DST)->getData());
        m_args[DNNL_ARG_SCRATCHPAD].set_data_handle(m_scratch->getData());
        if (const auto bias = m_args.find(DNNL_ARG_BIAS); bias != m_args.end()) {
            bias->second.set_data_handle(memory.at(ARG_BIAS)->getData());
        }
        m_primitive->execute(m_args);
    }

    impl_desc_type implType() const override {
        return m_primitive ? m_primitive->implType() : impl_desc_type::undef;
    }

    void moveMemToNumaNode(int id) override {
        if (m_weights && m_numa != id) {
            (void)mbind_move(m_args.at(DNNL_ARG_WEIGHTS), id);
            if (const auto bias = m_args.find(DNNL_ARG_BIAS); bias != m_args.end()) {
                (void)mbind_move(bias->second, id);
            }
            m_numa = id;
        }
    }

private:
    void initialize(const MemoryArgs& memory) {
        const auto& quant = memory.at(ARG_WEI);
        const auto type = quant->getPrecision();
        OPENVINO_ASSERT(m_attrs.constantWeights && !m_attrs.sparseWeights);
        OPENVINO_ASSERT(type == ov::element::i4 || type == ov::element::u4 || type == ov::element::i8 ||
                        type == ov::element::u8);
        OPENVINO_ASSERT(plain(quant) && plain(memory.at(ARG_SRC)) && plain(memory.at(ARG_DST)));
        const auto dims = quant->getShape().getStaticDims();
        OPENVINO_ASSERT(dims.size() == 2);
        // An optimized Transpose preserves logical [N, K] dimensions while its
        // shared data remains in KN order, as do the decompression parameters.
        const size_t n = dims[0];
        const size_t k = dims[1];
        OPENVINO_ASSERT(memory.at(ARG_SRC)->getPrecision() == ov::element::f16);
        OPENVINO_ASSERT(memory.at(ARG_SRC)->getShape().getStaticDims().back() == k);
        OPENVINO_ASSERT(memory.at(ARG_DST)->getShape().getStaticDims().back() == n);
        Params scales(memory.at(ARG_WEI | ARG_ATTR_SCALES), n, k);
        std::unique_ptr<Params> zero_points;
        if (const auto it = memory.find(ARG_WEI | ARG_ATTR_ZERO_POINTS); it != memory.end()) {
            zero_points = std::make_unique<Params>(it->second, n, k);
        }
        size_t rows = 1;
        const auto& source_dims = memory.at(ARG_SRC)->getShape().getStaticDims();
        for (size_t i = 0; i + 1 < source_dims.size(); ++i) {
            rows *= source_dims[i];
        }
        const auto raw_desc = std::make_shared<DnnlBlockedMemoryDesc>(ov::element::f16, Shape({n, k}));
        // Descriptor-only views: FC applies the same weights to all leading
        // dimensions, so [M, 1, K] must select one GEMM with M rows.
        float descriptor_storage = 0.F;
        MemoryArgs native = memory;
        native[ARG_WEI] = std::make_shared<Memory>(m_context->getEngine(), raw_desc, &descriptor_storage, false);
        native[ARG_SRC] =
            std::make_shared<Memory>(m_context->getEngine(),
                                     std::make_shared<DnnlBlockedMemoryDesc>(ov::element::f16, Shape({rows, k})),
                                     &descriptor_storage,
                                     false);
        native[ARG_DST] =
            std::make_shared<Memory>(m_context->getEngine(),
                                     std::make_shared<DnnlBlockedMemoryDesc>(ov::element::f16, Shape({rows, n})),
                                     &descriptor_storage,
                                     false);
        native.erase(ARG_WEI | ARG_ATTR_SCALES);
        native.erase(ARG_WEI | ARG_ATTR_ZERO_POINTS);
        MatMulAttrs attrs;
        attrs.transposeB = true;
        attrs.withBias = !memory.at(ARG_BIAS)->getDesc().empty();
        attrs.constantWeights = true;
        attrs.postOps = m_attrs.postOps;
        // Preserve the plain NK weight view rather than asking FC descriptor
        // selection for different packed layouts in prefill and decode.
        m_shape_data = DnnlMatMulPrimitive::createShapeAgnosticData(attrs, native, m_context, false);
        m_primitive = DnnlMatMulPrimitive::create(native, attrs, m_context, m_shape_data);
        const dnnl::memory::desc weight_view({static_cast<int64_t>(k), static_cast<int64_t>(n)},
                                             dnnl::memory::data_type::f16,
                                             dnnl::memory::format_tag::ba);
        OPENVINO_ASSERT(m_primitive && m_primitive->weightsDesc()->getDnnlDesc() == weight_view);
        const auto key =
            std::string("fp16_fc_weights/") + (m_attrs.weightsNonTransposed ? "kn/" : "nk/") +
            DnnlExtensionUtils::computeWeightsStringHash(quant, m_primitive->weightsDesc()) + "/" +
            DnnlExtensionUtils::computeWeightsStringHash(
                quant,
                MemoryDescUtils::convertToDnnlMemoryDesc(quant->getDescPtr())) +
            "/" +
            DnnlExtensionUtils::computeWeightsStringHash(
                scales.memory,
                MemoryDescUtils::convertToDnnlMemoryDesc(scales.memory->getDescPtr())) +
            (zero_points ? "/" + DnnlExtensionUtils::computeWeightsStringHash(
                                     zero_points->memory,
                                     MemoryDescUtils::convertToDnnlMemoryDesc(zero_points->memory->getDescPtr()))
                         : "/no_zp");
        auto create_weights = [&]() -> MemoryPtr {
            auto normalize = [&](const MemoryPtr& parameter, ov::element::Type precision) -> MemoryPtr {
                if (parameter->getPrecision() == precision) {
                    return parameter;
                }
                auto normalized = std::make_shared<Memory>(m_context->getEngine(),
                                                           parameter->getDescPtr()->cloneWithNewPrecision(precision));
                node::Reorder::reorderData(*parameter,
                                           *normalized,
                                           m_context->getRuntimeCache(),
                                           m_context->getThreadPool());
                return normalized;
            };
            const bool kn = m_attrs.weightsNonTransposed;
            auto group_sizes = [kn, n, k](const Params& parameters) -> dnnl::memory::dims {
                const auto channel_size = static_cast<int64_t>(n / parameters.channels);
                const auto group_size = static_cast<int64_t>(k / parameters.groups);
                return kn ? dnnl::memory::dims{group_size, channel_size} : dnnl::memory::dims{channel_size, group_size};
            };
            auto parameter_view = [this, kn](const Params& parameters, const MemoryPtr& memory) {
                const auto channels = static_cast<int64_t>(parameters.channels);
                const auto groups = static_cast<int64_t>(parameters.groups);
                const auto dimensions =
                    kn ? dnnl::memory::dims{groups, channels} : dnnl::memory::dims{channels, groups};
                const auto precision = DnnlExtensionUtils::ElementTypeToDataType(memory->getPrecision());
                return dnnl::memory({dimensions, precision, dnnl::memory::format_tag::ab},
                                    m_context->getEngine(),
                                    memory->getData());
            };
            const dnnl::memory::dims dimensions =
                kn ? dnnl::memory::dims{static_cast<int64_t>(k), static_cast<int64_t>(n)}
                   : dnnl::memory::dims{static_cast<int64_t>(n), static_cast<int64_t>(k)};
            const dnnl::memory::desc source_desc(dimensions,
                                                 DnnlExtensionUtils::ElementTypeToDataType(type),
                                                 dnnl::memory::format_tag::ab);
            const dnnl::memory::desc destination_desc(dimensions,
                                                      dnnl::memory::data_type::f16,
                                                      kn ? dnnl::memory::format_tag::ba : dnnl::memory::format_tag::ab);
            dnnl::primitive_attr attributes;
            attributes.set_scales(DNNL_ARG_SRC,
                                  3,
                                  group_sizes(scales),
                                  DnnlExtensionUtils::ElementTypeToDataType(scales.memory->getPrecision()));
            MemoryPtr zero_memory;
            auto result = std::make_shared<Memory>(m_context->getEngine(), m_primitive->weightsDesc());
            dnnl_primitive_args arguments{
                {DNNL_ARG_FROM, dnnl::memory(source_desc, m_context->getEngine(), quant->getData())},
                {DNNL_ARG_TO, dnnl::memory(destination_desc, m_context->getEngine(), result->getData())},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, parameter_view(scales, scales.memory)}};
            if (zero_points) {
                zero_memory = normalize(zero_points->memory, ov::element::i32);
                attributes.set_zero_points(DNNL_ARG_SRC, 3, group_sizes(*zero_points), dnnl::memory::data_type::s32);
                arguments[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC] = parameter_view(*zero_points, zero_memory);
            }
            const dnnl::reorder::primitive_desc descriptor(m_context->getEngine(),
                                                           source_desc,
                                                           m_context->getEngine(),
                                                           destination_desc,
                                                           attributes);
            auto stream = make_stream(m_context->getEngine(), m_context->getThreadPool());
            // Existing grouped reorder performs dequantization and the final FP16 store.
            dnnl::reorder(descriptor).execute(stream, arguments);
            stream.wait();
            return result;
        };
        // One model-owned FP16 copy is shared by streams and both execution
        // phases. WeightsSharing stores weak references, so keep its owner alive.
        if (const auto cache = m_context->getWeightsCache()) {
            m_weights = MemoryPtr(*cache->findOrCreate(key, create_weights));
        } else {
            m_weights = create_weights();
        }
        m_scratch = m_context->getScratchPad()->createScratchPadMem(m_primitive->scratchPadDesc());
        m_args = m_shape_data->m_primAttrs.dnnlArgs;
        m_args[DNNL_ARG_WEIGHTS] = m_weights->getPrimitive();
        m_args[DNNL_ARG_SCRATCHPAD] =
            dnnl::memory(m_primitive->scratchPadDesc()->getDnnlDesc(), m_context->getEngine(), DNNL_MEMORY_NONE);
        m_args[DNNL_ARG_SRC] =
            dnnl::memory(m_primitive->srcDesc()->getDnnlDesc(), m_context->getEngine(), DNNL_MEMORY_NONE);
        m_args[DNNL_ARG_DST] =
            dnnl::memory(m_primitive->dstDesc()->getDnnlDesc(), m_context->getEngine(), DNNL_MEMORY_NONE);
        if (attrs.withBias) {
            m_args[DNNL_ARG_BIAS] = native.at(ARG_BIAS)->getPrimitive();
        }
        m_src_desc = memory.at(ARG_SRC)->getDescPtr();
        m_dst_desc = memory.at(ARG_DST)->getDescPtr();
    }

    FCAttrs m_attrs;
    ExecutorContext::CPtr m_context;
    DnnlMatMulPrimitivePtr m_primitive;
    DnnlShapeAgnosticDataPtr m_shape_data;
    MemoryPtr m_weights, m_scratch;
    MemoryDescPtr m_src_desc, m_dst_desc;
    dnnl_primitive_args m_args;
    int m_numa = -1;
};
}  // namespace ov::intel_cpu::fp16_compressed_fc
