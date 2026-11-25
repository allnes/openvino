// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>

#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include "utils/precision_support.h"

#if defined(OPENVINO_ARCH_X86_64)
#    include "config.h"
#    include "cpu/x64/cpu_isa_traits.hpp"
#    include "cpu_types.h"
#    include "graph.h"
#    include "graph_context.h"
#    include "nodes/executors/jit/eltwise.h"
#endif

namespace ov::intel_cpu::test {
namespace {

TEST(FP16EltwiseConvert, FusesWideningInputAtEitherPortAndNarrowingOutput) {
#if !defined(OPENVINO_ARCH_X86_64)
    GTEST_SKIP();
#endif
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    using namespace ov::opset13;
    ov::Core core;
    for (const bool second_port : {false, true}) {
        auto input = std::make_shared<Parameter>(ov::element::f16, ov::PartialShape{-1, 17});
        auto widen = std::make_shared<Convert>(input, ov::element::f32);
        widen->set_friendly_name("widen");
        ov::disable_conversion(widen, ov::element::f16);
        auto constant = Constant::create(ov::element::f32, {}, {0.0006F});
        auto subtract =
            second_port ? std::make_shared<Subtract>(constant, widen) : std::make_shared<Subtract>(widen, constant);
        ov::disable_conversion(subtract, ov::element::f16);
        auto narrow = std::make_shared<Convert>(subtract, ov::element::f16);
        narrow->set_friendly_name("narrow");
        auto model = std::make_shared<ov::Model>(ov::OutputVector{narrow}, ov::ParameterVector{input});
        auto compiled = core.compile_model(
            model,
            "CPU",
            {ov::hint::inference_precision(ov::element::f16), ov::num_streams(1), ov::inference_num_threads(2)});
        auto request = compiled.create_infer_request();
        for (const size_t rows : {1U, 31U, 1U}) {
            ov::Tensor tensor(ov::element::f16, {rows, 17});
            for (size_t i = 0; i < tensor.get_size(); ++i) {
                tensor.data<ov::float16>()[i] = ov::float16((static_cast<int>(i % 29) - 14) * 0.03125F);
            }
            request.set_input_tensor(tensor);
            request.infer();
            const auto result = request.get_output_tensor();
            for (size_t i = 0; i < tensor.get_size(); ++i) {
                const float value = tensor.data<const ov::float16>()[i];
                EXPECT_EQ(result.data<const ov::float16>()[i],
                          ov::float16(second_port ? 0.0006F - value : value - 0.0006F));
            }
            size_t conversions = 0;
            for (const auto& node : compiled.get_runtime_model()->get_ordered_ops()) {
                const auto& info = node->get_rt_info();
                if (info.count("layerType") && info.at("layerType").as<std::string>() == "Convert") {
                    ++conversions;
                }
            }
            EXPECT_EQ(conversions, 0U);
        }
    }
}

TEST(FP16EltwiseConvert, PreservesIntermediateFP16Rounding) {
#if !defined(OPENVINO_ARCH_X86_64)
    GTEST_SKIP();
#endif
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    using namespace ov::opset13;
    auto input = std::make_shared<Parameter>(ov::element::f16, ov::Shape{17});
    const ov::float16 half_constant(0.0006F);
    auto subtract = std::make_shared<Subtract>(input, Constant::create(ov::element::f16, {}, {0.0006F}));
    auto widen = std::make_shared<Convert>(subtract, ov::element::f32);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{widen}, ov::ParameterVector{input});
    ov::Core core;
    auto compiled = core.compile_model(
        model,
        "CPU",
        {ov::hint::inference_precision(ov::element::f16), ov::num_streams(1), ov::inference_num_threads(2)});
    auto request = compiled.create_infer_request();
    ov::Tensor tensor(ov::element::f16, {17});
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        tensor.data<ov::float16>()[i] = ov::float16(1.F + i * 0.03125F);
    }
    request.set_input_tensor(tensor);
    request.infer();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        const float expected =
            ov::float16(static_cast<float>(tensor.data<const ov::float16>()[i]) - static_cast<float>(half_constant));
        EXPECT_EQ(request.get_output_tensor().data<const float>()[i], expected);
    }
}

TEST(FP16EltwiseConvert, PreservesSharedWideningConvert) {
#if !defined(OPENVINO_ARCH_X86_64)
    GTEST_SKIP();
#endif
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    using namespace ov::opset13;
    auto input = std::make_shared<Parameter>(ov::element::f16, ov::Shape{17});
    auto widen = std::make_shared<Convert>(input, ov::element::f32);
    ov::disable_conversion(widen, ov::element::f16);
    auto subtract = std::make_shared<Subtract>(widen, Constant::create(ov::element::f32, {}, {0.0006F}));
    ov::disable_conversion(subtract, ov::element::f16);
    auto narrow = std::make_shared<Convert>(subtract, ov::element::f16);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{narrow, widen}, ov::ParameterVector{input});
    ov::Core core;
    auto compiled = core.compile_model(
        model,
        "CPU",
        {ov::hint::inference_precision(ov::element::f16), ov::num_streams(1), ov::inference_num_threads(2)});
    auto request = compiled.create_infer_request();
    ov::Tensor tensor(ov::element::f16, {17});
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        tensor.data<ov::float16>()[i] = ov::float16(1.F + i * 0.03125F);
    }
    request.set_input_tensor(tensor);
    request.infer();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        const float value = tensor.data<const ov::float16>()[i];
        EXPECT_EQ(request.get_output_tensor(0).data<const ov::float16>()[i], ov::float16(value - 0.0006F));
        EXPECT_EQ(request.get_output_tensor(1).data<const float>()[i], value);
    }
}

TEST(FP16EltwiseConvert, PreservesConvertsBelowAVX2) {
#if defined(OPENVINO_ARCH_X86_64)
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
        GTEST_SKIP();
    }
    using namespace ov::opset13;
    auto input = std::make_shared<Parameter>(ov::element::f16, ov::Shape{17});
    auto widen = std::make_shared<Convert>(input, ov::element::f32);
    ov::disable_conversion(widen, ov::element::f16);
    auto subtract = std::make_shared<Subtract>(Constant::create(ov::element::f32, {}, {0.0006F}), widen);
    ov::disable_conversion(subtract, ov::element::f16);
    auto narrow = std::make_shared<Convert>(subtract, ov::element::f16);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{narrow}, ov::ParameterVector{input});
    // Compile the backend graph directly so front-end precision promotion cannot hide
    // an unsupported FP16 load/store under the ISA cap.
    Config config;
    auto context = std::make_shared<GraphContext>(config, nullptr, false);
    Graph graph;
    ASSERT_NO_THROW(graph.Init(model, context));
    size_t conversions = 0;
    for (const auto& node : graph.GetNodes()) {
        conversions += node->getType() == Type::Convert;
    }
    EXPECT_EQ(conversions, 2U);
    ov::Core core;
    auto compiled = core.compile_model(
        model,
        "CPU",
        {ov::hint::inference_precision(ov::element::f16), ov::num_streams(1), ov::inference_num_threads(2)});
    auto request = compiled.create_infer_request();
    ov::Tensor tensor(ov::element::f16, {17});
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        tensor.data<ov::float16>()[i] = ov::float16(1.F + i * 0.03125F);
    }
    request.set_input_tensor(tensor);
    request.infer();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        const float value = tensor.data<const ov::float16>()[i];
        EXPECT_EQ(request.get_output_tensor().data<const ov::float16>()[i], ov::float16(0.0006F - value));
    }
#else
    GTEST_SKIP();
#endif
}

TEST(EltwiseReducedPrecisionFallback, RejectsUnsupportedLoadAndStorePrecisions) {
#if defined(OPENVINO_ARCH_X86_64)
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ||
        dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni_2)) {
        GTEST_SKIP();
    }
    EltwiseAttrs attrs;
    attrs.data.algo = Algorithm::EltwiseSubtract;
    EXPECT_FALSE(EltwiseJitExecutor::supports(attrs, 2, {ov::element::f32, ov::element::f32}, {ov::element::bf16}));
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
        for (const auto precision : {ov::element::f16, ov::element::bf16}) {
            EXPECT_FALSE(EltwiseJitExecutor::supports(attrs, 2, {precision, ov::element::f32}, {ov::element::f32}));
        }
        EXPECT_FALSE(EltwiseJitExecutor::supports(attrs, 2, {ov::element::f32, ov::element::f32}, {ov::element::f16}));
    }
#else
    GTEST_SKIP();
#endif
}

}  // namespace
}  // namespace ov::intel_cpu::test
