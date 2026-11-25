// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/cpu_opset/common/pass/move_fp16_convert_after_view.hpp"
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

TEST(FP16EltwiseConvert, FusesWideningThroughShapeOperations) {
#if !defined(OPENVINO_ARCH_X86_64)
    GTEST_SKIP();
#endif
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    using namespace ov::opset13;
    auto input = std::make_shared<Parameter>(ov::element::f16, ov::PartialShape{-1, 8});
    auto widen = std::make_shared<Convert>(input, ov::element::f32);
    auto reshape = std::make_shared<Reshape>(widen, Constant::create(ov::element::i32, {3}, {-1, 2, 4}), false);
    auto unsqueeze = std::make_shared<Unsqueeze>(reshape, Constant::create(ov::element::i32, {1}, {1}));
    auto squeeze = std::make_shared<Squeeze>(unsqueeze, Constant::create(ov::element::i32, {1}, {1}));
    auto subtract = std::make_shared<Subtract>(squeeze, Constant::create(ov::element::f32, {}, {0.0006F}));
    for (const auto& node : ov::NodeVector{widen, reshape, unsqueeze, squeeze, subtract}) {
        ov::disable_conversion(node, ov::element::f16);
    }
    auto narrow = std::make_shared<Convert>(subtract, ov::element::f16);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{narrow}, ov::ParameterVector{input});
    ov::Core core;
    auto compiled = core.compile_model(
        model,
        "CPU",
        {ov::hint::inference_precision(ov::element::f16), ov::num_streams(1), ov::inference_num_threads(2)});
    auto request = compiled.create_infer_request();
    for (const size_t rows : {1U, 5U, 1U}) {
        ov::Tensor tensor(ov::element::f16, {rows, 8});
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            tensor.data<ov::float16>()[i] = ov::float16(static_cast<float>(static_cast<int>(i % 9) - 4) * 0.03125F);
        }
        request.set_input_tensor(tensor);
        request.infer();
        const auto output = request.get_output_tensor();
        EXPECT_EQ(output.get_shape(), (ov::Shape{rows, 2, 4}));
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            EXPECT_EQ(output.data<const ov::float16>()[i], ov::float16(tensor.data<const ov::float16>()[i] - 0.0006F));
        }
    }
    size_t conversions = 0;
    for (const auto& node : compiled.get_runtime_model()->get_ordered_ops()) {
        const auto& info = node->get_rt_info();
        conversions += info.count("layerType") && info.at("layerType").as<std::string>() == "Convert";
    }
    EXPECT_EQ(conversions, 0U);
}

TEST(FP16EltwiseConvert, KeepsSplitOutputsInFP16UntilArithmetic) {
#if !defined(OPENVINO_ARCH_X86_64)
    GTEST_SKIP();
#endif
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    using namespace ov::opset13;
    auto input = std::make_shared<Parameter>(ov::element::f16, ov::PartialShape{-1, 8});
    auto widen = std::make_shared<Convert>(input, ov::element::f32);
    auto split = std::make_shared<VariadicSplit>(widen,
                                                 Constant::create(ov::element::i32, {}, {1}),
                                                 Constant::create(ov::element::i32, {2}, {3, 5}));
    auto first = std::make_shared<Convert>(split->output(0), ov::element::f16);
    auto reshape = std::make_shared<Reshape>(split->output(1), Constant::create(ov::element::i32, {2}, {-1, 5}), false);
    auto subtract = std::make_shared<Subtract>(reshape, Constant::create(ov::element::f32, {}, {0.0006F}));
    for (const auto& node : ov::NodeVector{widen, split, reshape, subtract}) {
        ov::disable_conversion(node, ov::element::f16);
    }
    auto second = std::make_shared<Convert>(subtract, ov::element::f16);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{first, second}, ov::ParameterVector{input});
    ov::Core core;
    auto compiled = core.compile_model(
        model,
        "CPU",
        {ov::hint::inference_precision(ov::element::f16), ov::num_streams(1), ov::inference_num_threads(2)});
    auto request = compiled.create_infer_request();
    for (const size_t rows : {1U, 5U, 1U}) {
        ov::Tensor tensor(ov::element::f16, {rows, 8});
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            tensor.data<ov::float16>()[i] = ov::float16(static_cast<float>(static_cast<int>(i % 9) - 4) * 0.03125F);
        }
        request.set_input_tensor(tensor);
        request.infer();
        for (size_t row = 0; row < rows; ++row) {
            for (size_t channel = 0; channel < 8; ++channel) {
                const auto value = tensor.data<const ov::float16>()[row * 8 + channel];
                if (channel < 3) {
                    EXPECT_EQ(request.get_output_tensor(0).data<const ov::float16>()[row * 3 + channel], value);
                } else {
                    EXPECT_EQ(request.get_output_tensor(1).data<const ov::float16>()[row * 5 + channel - 3],
                              ov::float16(value - 0.0006F));
                }
            }
        }
    }
    size_t conversions = 0;
    for (const auto& node : compiled.get_runtime_model()->get_ordered_ops()) {
        const auto& info = node->get_rt_info();
        conversions += info.count("layerType") && info.at("layerType").as<std::string>() == "Convert";
    }
    EXPECT_EQ(conversions, 0U);
}

TEST(FP16ViewConvert, PreservesPrecisionAndConstantFoldingMarkers) {
#if !defined(OPENVINO_ARCH_X86_64)
    GTEST_SKIP();
#endif
    using namespace ov::opset13;
    auto input = std::make_shared<Parameter>(ov::element::f16, ov::Shape{8});
    auto widen = std::make_shared<Convert>(input, ov::element::f32);
    auto reshape = std::make_shared<Reshape>(widen, Constant::create(ov::element::i32, {2}, {2, 4}), false);
    reshape->output(0).get_tensor().set_names({"view_output"});
    for (const auto& node : ov::NodeVector{widen, reshape}) {
        ov::disable_conversion(node, ov::element::f16);
        ov::pass::disable_constant_folding(node);
    }
    auto model = std::make_shared<ov::Model>(ov::OutputVector{reshape}, ov::ParameterVector{input});
    ov::pass::Manager manager;
    manager.register_pass<MoveFP16ConvertAfterView>();
    manager.run_passes(model);
    const auto after = model->get_results()[0]->get_input_node_shared_ptr(0);
#if defined(OPENVINO_ARCH_X86_64)
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
        EXPECT_EQ(after, reshape);
        EXPECT_EQ(reshape->get_input_node_shared_ptr(0), widen);
        for (const auto& node : ov::NodeVector{widen, reshape}) {
            EXPECT_TRUE(ov::is_conversion_disabled(node, ov::element::f16));
            EXPECT_TRUE(ov::pass::constant_folding_is_disabled(node));
        }
        EXPECT_TRUE(model->output(0).get_names().count("view_output"));
        return;
    }
#endif
    ASSERT_TRUE(ov::is_type<Convert>(after));
    const auto view = after->get_input_node_shared_ptr(0);
    ASSERT_TRUE(ov::is_type<Reshape>(view));
    EXPECT_EQ(view->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(after->get_output_element_type(0), ov::element::f32);
    for (const auto& node : ov::NodeVector{view, after}) {
        EXPECT_TRUE(ov::is_conversion_disabled(node, ov::element::f16));
        EXPECT_TRUE(ov::pass::constant_folding_is_disabled(node));
    }
    EXPECT_TRUE(model->output(0).get_names().count("view_output"));
}

TEST(FP16ViewConvert, PreservesSharedFP32WideningOutput) {
    using namespace ov::opset13;
    auto input = std::make_shared<Parameter>(ov::element::f16, ov::Shape{8});
    auto widen = std::make_shared<Convert>(input, ov::element::f32);
    auto reshape = std::make_shared<Reshape>(widen, Constant::create(ov::element::i32, {2}, {2, 4}), false);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{reshape, widen}, ov::ParameterVector{input});
    ov::pass::Manager manager;
    manager.register_pass<MoveFP16ConvertAfterView>();
    manager.run_passes(model);
    EXPECT_EQ(model->get_results()[0]->get_input_node_shared_ptr(0), reshape);
    EXPECT_EQ(model->get_results()[1]->get_input_node_shared_ptr(0), widen);
    EXPECT_EQ(reshape->get_output_element_type(0), ov::element::f32);
}

TEST(FP16ViewConvert, PreservesAllFP16RoundtripEncodings) {
#if !defined(OPENVINO_ARCH_X86_64)
    GTEST_SKIP();
#endif
    using namespace ov::opset13;
    constexpr size_t encodings = 65536;
    constexpr size_t elements = encodings + 1;
    auto make_model = [](size_t view_rank) {
        auto input = std::make_shared<Parameter>(ov::element::f16, ov::Shape{elements});
        auto widen = std::make_shared<Convert>(input, ov::element::f32);
        ov::disable_conversion(widen, ov::element::f16);
        ov::Output<ov::Node> data = widen;
        if (view_rank != 0) {
            ov::Shape view_shape(view_rank, 1);
            view_shape.back() = elements;
            auto view =
                std::make_shared<Reshape>(widen, Constant::create(ov::element::i32, {view_rank}, view_shape), false);
            ov::disable_conversion(view, ov::element::f16);
            data = view;
        }
        auto narrow = std::make_shared<Convert>(data, ov::element::f16);
        ov::OutputVector outputs{narrow};
        if (view_rank == 0) {
            // Keep the original widening observable so it cannot be eliminated.
            outputs.push_back(widen);
        }
        return std::make_shared<ov::Model>(outputs, ov::ParameterVector{input});
    };
    ov::Core core;
    const ov::AnyMap properties{ov::hint::inference_precision(ov::element::f16),
                                ov::num_streams(1),
                                ov::inference_num_threads(2)};
    auto reference = core.compile_model(make_model(0), "CPU", properties).create_infer_request();
    ov::Tensor tensor(ov::element::f16, {elements});
    for (size_t bits = 0; bits < encodings; ++bits) {
        tensor.data<ov::float16>()[bits] = ov::float16::from_bits(static_cast<uint16_t>(bits));
    }
    tensor.data<ov::float16>()[encodings] = ov::float16::from_bits(0xfc01);
    reference.set_input_tensor(tensor);
    reference.infer();
    const auto expected = reference.get_output_tensor(0);
#if defined(OPENVINO_ARCH_X86_64)
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
        EXPECT_EQ(expected.data<const ov::float16>()[0x7c01].to_bits(), 0x7e01);
        EXPECT_EQ(expected.data<const ov::float16>()[encodings].to_bits(), 0xfe01);
    }
#endif
    for (const size_t view_rank : {2U, 13U}) {
        auto optimized = core.compile_model(make_model(view_rank), "CPU", properties).create_infer_request();
        optimized.set_input_tensor(tensor);
        optimized.infer();
        const auto actual = optimized.get_output_tensor(0);
        for (size_t bits = 0; bits < elements; ++bits) {
            EXPECT_EQ(actual.data<const ov::float16>()[bits].to_bits(),
                      expected.data<const ov::float16>()[bits].to_bits())
                << "rank=" << view_rank << ", input bits=" << bits;
        }
    }
}

}  // namespace
}  // namespace ov::intel_cpu::test
