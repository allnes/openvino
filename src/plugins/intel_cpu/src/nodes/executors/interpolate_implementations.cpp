// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_config.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/interpolate.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"

#include "nodes/executors/ref/interpolate.hpp"
#if defined(OV_CPU_WITH_ACL)
#    include "nodes/executors/acl/acl_interpolate.hpp"
#endif

namespace ov::intel_cpu {

using namespace executor;

namespace {

bool is_interpolate_jit_feature_enabled() {
#if defined(OPENVINO_ARCH_X86_64)
    static const bool enabled = [] {
        const char* flag = std::getenv("OV_CPU_ENABLE_INTERPOLATE_JIT");
        if (!flag) {
            return false;
        }
        return !(flag[0] == '0' && flag[1] == '\0');
    }();
    return enabled;
#else
    return false;
#endif
}

class InterpolateRefWrapperExecutor : public Executor {
public:
    InterpolateRefWrapperExecutor(const InterpolateAttrs& attrs,
                                  const MemoryArgs& memory,
                                  const ExecutorContext::CPtr& context)
        : m_attrs(attrs), m_context(context) {
        update(memory);
    }

    bool update(const MemoryArgs& memory) override {
        m_memory = memory;
        auto src = m_memory.at(ARG_SRC);
        auto dst = m_memory.at(ARG_DST);

        const auto& srcDesc = src->getDescPtr();
        const auto& dstDesc = dst->getDescPtr();

        const auto srcDims = srcDesc->getShape().getDims();
        const auto dstDims = dstDesc->getShape().getDims();
        if (srcDims != m_lastSrcDims || dstDims != m_lastDstDims || !m_refExecutor) {
            m_refExecutor = std::make_shared<InterpolateRefExecutor>(m_context);
            if (!m_refExecutor->init(m_attrs, {srcDesc}, {dstDesc}, {})) {
                m_refExecutor.reset();
                return false;
            }
            m_lastSrcDims = srcDims;
            m_lastDstDims = dstDims;
        }
        return true;
    }

    void execute(const MemoryArgs& memory) override {
        if (!m_refExecutor) {
            update(memory);
        }
        std::vector<MemoryCPtr> srcVec{memory.at(ARG_SRC)};
        std::vector<MemoryPtr> dstVec{memory.at(ARG_DST)};
        m_refExecutor->exec(srcVec, dstVec, nullptr);
    }

    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::ref;
    }

private:
    InterpolateAttrs m_attrs;
    ExecutorContext::CPtr m_context;
    MemoryArgs m_memory;
    std::shared_ptr<InterpolateRefExecutor> m_refExecutor;
    VectorDims m_lastSrcDims;
    VectorDims m_lastDstDims;
};

}  // namespace

static ExecutorImplementation<InterpolateAttrs> make_interpolate_ref_impl() {
    auto supports = [](const executor::Config<InterpolateAttrs>& config) {
        const auto& descs = config.descs;
        auto itSrc = descs.find(ARG_SRC);
        auto itDst = descs.find(ARG_DST);
        if (itSrc == descs.end() || itDst == descs.end()) {
            return false;
        }
        const size_t rIn = itSrc->second->getShape().getRank();
        const size_t rOut = itDst->second->getShape().getRank();
        return (rIn >= 3 && rIn <= 5) && (rOut >= 3 && rOut <= 5);
    };

    auto createOptimal = [](const executor::Config<InterpolateAttrs>&)
        -> std::optional<executor::Config<InterpolateAttrs>> { return {}; };

    auto accepts = [](const InterpolateAttrs&, const MemoryArgs&) { return true; };

    auto create = [](const InterpolateAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) {
        return std::make_shared<InterpolateRefWrapperExecutor>(attrs, memory, context);
    };

    return ExecutorImplementation<InterpolateAttrs>(
        "interpolate_ref", ExecutorType::Reference, OperationType::Interpolate, supports, createOptimal, accepts, create);
}

static ExecutorImplementation<InterpolateAttrs> make_interpolate_jit_stub(const std::string& name) {
    auto supports = [](const executor::Config<InterpolateAttrs>&) { return false; };
    auto createOptimal = [](const executor::Config<InterpolateAttrs>&)
        -> std::optional<executor::Config<InterpolateAttrs>> { return {}; };
    auto accepts = [](const InterpolateAttrs&, const MemoryArgs&) { return false; };
    auto create = [](const InterpolateAttrs&, const MemoryArgs&, const ExecutorContext::CPtr&)
        -> std::shared_ptr<Executor> { return {}; };
    return ExecutorImplementation<InterpolateAttrs>(name.c_str(),
                                                    ExecutorType::Jit,
                                                    OperationType::Interpolate,
                                                    supports,
                                                    createOptimal,
                                                    accepts,
                                                    create);
}

#if defined(OV_CPU_WITH_ACL)
static ExecutorImplementation<InterpolateAttrs> make_interpolate_acl_impl() {
    auto supports = [](const executor::Config<InterpolateAttrs>& config) {
        auto itSrc = config.descs.find(ARG_SRC);
        auto itDst = config.descs.find(ARG_DST);
        if (itSrc == config.descs.end() || itDst == config.descs.end()) {
            return false;
        }
        std::vector<MemoryDescPtr> srcDescs{itSrc->second};
        std::vector<MemoryDescPtr> dstDescs{itDst->second};
        return acl_interpolate_is_supported(config.attrs, srcDescs, dstDescs);
    };

    auto createOptimal = [](const executor::Config<InterpolateAttrs>&)
        -> std::optional<executor::Config<InterpolateAttrs>> { return {}; };

    auto accepts = [](const InterpolateAttrs&, const MemoryArgs&) { return true; };

    struct ACLWrapper final : Executor {
        ACLWrapper(std::shared_ptr<ACLInterpolateExecutor> exec, ExecutorContext::CPtr ctx)
            : impl(std::move(exec)), context(std::move(ctx)) {}

        bool update(const MemoryArgs& memory) override {
            m_memory = memory;
            return true;
        }

        void execute(const MemoryArgs& memory) override {
            std::vector<MemoryCPtr> src{memory.at(ARG_SRC)};
            std::vector<MemoryPtr> dst{memory.at(ARG_DST)};
            impl->exec(src, dst, nullptr);
        }

        [[nodiscard]] impl_desc_type implType() const override {
            return impl_desc_type::acl;
        }

        std::shared_ptr<ACLInterpolateExecutor> impl;
        ExecutorContext::CPtr context;
        MemoryArgs m_memory;
    };

    auto create = [](const InterpolateAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) {
        auto exec = std::make_shared<ACLInterpolateExecutor>(context);
        std::vector<MemoryDescPtr> srcDescs{memory.at(ARG_SRC)->getDescPtr()};
        std::vector<MemoryDescPtr> dstDescs{memory.at(ARG_DST)->getDescPtr()};
        if (!exec->init(attrs, srcDescs, dstDescs, {})) {
            OPENVINO_THROW("ACL Interpolate init failed");
        }
        auto wrapper = std::make_shared<ACLWrapper>(exec, context);
        wrapper->update(memory);
        return std::static_pointer_cast<Executor>(wrapper);
    };

    return ExecutorImplementation<InterpolateAttrs>("interpolate_acl",
                                                    ExecutorType::Acl,
                                                    OperationType::Interpolate,
                                                    supports,
                                                    createOptimal,
                                                    accepts,
                                                    create);
}
#endif

template <>
const std::vector<ExecutorImplementation<InterpolateAttrs>>& getImplementations<InterpolateAttrs>() {
    static const std::vector<ExecutorImplementation<InterpolateAttrs>> impls = [] {
        std::vector<ExecutorImplementation<InterpolateAttrs>> result;

        if (is_interpolate_jit_feature_enabled()) {
            result.push_back(make_interpolate_jit_stub("interpolate_jit_x64_nearest"));
            result.push_back(make_interpolate_jit_stub("interpolate_jit_x64_linear"));
            result.push_back(make_interpolate_jit_stub("interpolate_jit_x64_linear_onnx"));
        }

#if defined(OV_CPU_WITH_ACL)
        result.push_back(make_interpolate_acl_impl());
#endif

        result.push_back(make_interpolate_ref_impl());
        return result;
    }();

    return impls;
}

}  // namespace ov::intel_cpu
