// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/cpu_opset/common/pass/move_fc_scales_before_split.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include "utils/precision_support.h"

namespace ov::intel_cpu::test {
namespace {
std::string layer_type(const std::shared_ptr<ov::Node>& node) {
    const auto& info = node->get_rt_info();
    return info.count("layerType") ? info.at("layerType").as<std::string>() : "";
}

TEST(FP16SplitScales, FusesOutputScalesIntoCompressedFC) {
#if !defined(OPENVINO_ARCH_X86_64)
    GTEST_SKIP();
#endif
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    using namespace ov::opset13;
    constexpr size_t k = 64;
    constexpr size_t n = 17;
    ov::Core core;
    for (const auto weight_type : {ov::element::i4, ov::element::i8}) {
        for (const size_t dq_group : {0U, 32U}) {
            auto input = std::make_shared<Parameter>(ov::element::f16, ov::PartialShape{-1, k});
            std::vector<int32_t> values(n * k);
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] = static_cast<int32_t>(i * 7 % 16) - 8;
            }
            auto weights =
                std::make_shared<Convert>(Constant::create(weight_type, ov::Shape{n, 2, 32}, values), ov::element::f16);
            auto dequantized =
                std::make_shared<Multiply>(weights, Constant::create(ov::element::f16, ov::Shape{n, 2, 1}, {0.0625F}));
            auto flat = std::make_shared<Reshape>(dequantized, Constant::create(ov::element::i64, {2}, {n, k}), false);
            auto fc = std::make_shared<MatMul>(input, flat, false, true);
            auto split = std::make_shared<VariadicSplit>(fc,
                                                         Constant::create(ov::element::i64, {}, {-1}),
                                                         Constant::create(ov::element::i64, {2}, {5, 12}));
            auto first = std::make_shared<Multiply>(split->output(0), Constant::create(ov::element::f16, {}, {2.F}));
            auto second = std::make_shared<Multiply>(split->output(1), Constant::create(ov::element::f16, {}, {4.F}));
            auto model = std::make_shared<ov::Model>(ov::OutputVector{first, second}, ov::ParameterVector{input});
            auto compile = [&](ov::element::Type precision) {
                return core.compile_model(model->clone(),
                                          "CPU",
                                          {ov::hint::inference_precision(precision),
                                           ov::hint::dynamic_quantization_group_size(dq_group),
                                           ov::inference_num_threads(2),
                                           ov::num_streams(1)});
            };
            auto compiled = compile(ov::element::f16);
            auto reference = compile(ov::element::f32);
            auto request = compiled.create_infer_request();
            auto control = reference.create_infer_request();
            for (const size_t rows : {1U, 31U, 1U}) {
                ov::Tensor tensor(ov::element::f16, ov::Shape{rows, k});
                for (size_t i = 0; i < tensor.get_size(); ++i) {
                    tensor.data<ov::float16>()[i] =
                        ov::float16(static_cast<float>(static_cast<int>(i * 17 % 31) - 15) * 0.125F);
                }
                request.set_input_tensor(tensor);
                control.set_input_tensor(tensor);
                request.infer();
                control.infer();
                for (size_t port = 0; port < 2; ++port) {
                    const auto actual = request.get_output_tensor(port);
                    const auto expected = control.get_output_tensor(port);
                    ASSERT_EQ(actual.get_shape(), expected.get_shape());
                    for (size_t i = 0; i < actual.get_size(); ++i) {
                        EXPECT_EQ(actual.data<const ov::float16>()[i], expected.data<const ov::float16>()[i]);
                    }
                }
                size_t fc_count = 0, scales_after_split = 0;
                for (const auto& node : compiled.get_runtime_model()->get_ordered_ops()) {
                    if (layer_type(node) == "FullyConnected") {
                        ++fc_count;
                        EXPECT_TRUE(node->input_value(1).get_element_type().is_integral_number());
                    }
                    if (layer_type(node) == "Eltwise") {
                        for (const auto& value : node->input_values()) {
                            if (layer_type(value.get_node_shared_ptr()) == "Split") {
                                ++scales_after_split;
                            }
                        }
                    }
                }
                EXPECT_EQ(fc_count, 1U);
                EXPECT_EQ(scales_after_split, 0U);
            }
        }
    }
}

TEST(FP16SplitScales, PreservesSharedAndUnsupportedOutputs) {
    using namespace ov::opset13;
    struct Case {
        const char* name;
        float scale;
        int64_t axis;
        bool raw_consumer = false;
        bool different_scales = false;
        bool shared_fc = false;
        bool f32 = false;
        bool disable = false;
    };
    const std::vector<Case> cases = {
        {"negative", -2.F, -1},
        {"zero", 0.F, -1},
        {"infinite", std::numeric_limits<float>::infinity(), -1},
        {"nan", std::numeric_limits<float>::quiet_NaN(), -1},
        {"non_channel_axis", 2.F, 0},
        {"shared_raw_output", 2.F, -1, true},
        {"different_scales", 2.F, -1, false, true},
        {"shared_fc", 2.F, -1, false, false, true},
        {"f32", 2.F, -1, false, false, false, true},
        {"callback", 2.F, -1, false, false, false, false, true},
    };
    for (const auto& item : cases) {
        SCOPED_TRACE(item.name);
        const auto precision = item.f32 ? ov::element::f32 : ov::element::f16;
        auto input = std::make_shared<Parameter>(precision, ov::Shape{2, 4});
        auto fc = std::make_shared<MatMul>(input, Constant::create(precision, {6, 4}, {0.25F}), false, true);
        auto split = std::make_shared<Split>(fc, Constant::create(ov::element::i64, {}, {item.axis}), 2);
        const ov::float16 half_scale(item.scale);
        auto scale = precision == ov::element::f16 ? std::make_shared<Constant>(precision, ov::Shape{}, &half_scale)
                                                   : Constant::create(precision, {}, {item.scale});
        auto first = std::make_shared<Multiply>(split->output(0), scale);
        ov::OutputVector outputs{first, split->output(1)};
        if (item.raw_consumer) {
            outputs.push_back(split->output(0));
        }
        if (item.different_scales) {
            outputs.push_back(std::make_shared<Multiply>(split->output(0), Constant::create(precision, {}, {4.F})));
        }
        if (item.shared_fc) {
            outputs.push_back(fc);
        }
        auto model = std::make_shared<ov::Model>(outputs, ov::ParameterVector{input});
        ov::pass::Manager manager;
        manager.register_pass<MoveFCScalesBeforeSplit>();
        if (item.disable) {
            manager.get_pass_config()->set_callback<MoveFCScalesBeforeSplit>(
                [](const std::shared_ptr<const ov::Node>&) {
                    return true;
                });
        }
        manager.run_passes(model);
        EXPECT_EQ(split->input_value(0), fc->output(0));
        EXPECT_EQ(first->input_value(0), split->output(0));
    }
}

TEST(FP16SplitScales, CombinesDuplicateScalesAndPreservesUnscaledChannels) {
    using namespace ov::opset13;
    for (const int64_t axis : {-1, 1}) {
        auto input = std::make_shared<Parameter>(ov::element::f16, ov::PartialShape{-1, 4});
        auto fc = std::make_shared<MatMul>(input, Constant::create(ov::element::f16, {6, 4}, {0.25F}), false, true);
        auto split = std::make_shared<VariadicSplit>(fc,
                                                     Constant::create(ov::element::i64, {}, {axis}),
                                                     Constant::create(ov::element::i64, {2}, {2, -1}));
        auto scale = Constant::create(ov::element::f16, {}, {2.F});
        auto first = std::make_shared<Multiply>(split->output(0), scale);
        auto duplicate = std::make_shared<Multiply>(scale, split->output(0));
        auto model = std::make_shared<ov::Model>(ov::OutputVector{first, duplicate, split->output(1)},
                                                 ov::ParameterVector{input});
        ov::pass::Manager manager;
        manager.register_pass<MoveFCScalesBeforeSplit>();
        manager.run_passes(model);
        const auto multiply = ov::as_type_ptr<Multiply>(split->get_input_node_shared_ptr(0));
        ASSERT_NE(multiply, nullptr);
        const auto constant = ov::as_type_ptr<Constant>(multiply->get_input_node_shared_ptr(1));
        ASSERT_NE(constant, nullptr);
        EXPECT_EQ(constant->cast_vector<float>(), (std::vector<float>{2.F, 2.F, 1.F, 1.F, 1.F, 1.F}));
        EXPECT_EQ(model->output(0).get_node()->input_value(0), split->output(0));
        EXPECT_EQ(model->output(1).get_node()->input_value(0), split->output(0));
        EXPECT_EQ(model->output(2).get_node()->input_value(0), split->output(1));
    }
}

TEST(FP16SplitScales, PreservesScalarPrecisionAndPrecisionMarkers) {
    using namespace ov::opset13;
    for (const bool fp32_scalar : {false, true}) {
        auto input = std::make_shared<Parameter>(ov::element::f16, ov::Shape{2, 4});
        auto fc = std::make_shared<MatMul>(input, Constant::create(ov::element::f16, {6, 4}, {0.25F}), false, true);
        auto split = std::make_shared<Split>(fc, Constant::create(ov::element::i64, {}, {-1}), 2);
        const auto precision = fp32_scalar ? ov::element::f32 : ov::element::f16;
        auto scale = Constant::create(precision, {}, {fp32_scalar ? 65536.F : 2.F});
        auto first = std::make_shared<ov::op::TypeRelaxed<Multiply>>();
        first->set_arguments(ov::OutputVector{split->output(0), scale});
        first->set_origin_input_type(precision, 0);
        first->set_origin_input_type(precision, 1);
        first->set_overridden_output_type(ov::element::f16);
        first->validate_and_infer_types();
        if (!fp32_scalar) {
            ov::disable_conversion(first, ov::element::f16);
        }
        auto model = std::make_shared<ov::Model>(ov::OutputVector{first, split->output(1)}, ov::ParameterVector{input});
        ov::pass::Manager manager;
        manager.register_pass<MoveFCScalesBeforeSplit>();
        manager.run_passes(model);
        EXPECT_EQ(split->input_value(0), fc->output(0));
        EXPECT_EQ(first->input_value(0), split->output(0));
        EXPECT_EQ(first->input_value(1), scale->output(0));
        if (!fp32_scalar) {
            EXPECT_TRUE(ov::is_conversion_disabled(first, ov::element::f16));
        }
    }
}

TEST(FP16SplitScales, RoundsScaledAccumulatorAtFinalStore) {
#if !defined(OPENVINO_ARCH_X86_64)
    GTEST_SKIP();
#endif
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    using namespace ov::opset13;
    ov::Core core;
    for (const bool subnormal : {false, true}) {
        constexpr size_t k = 32;
        auto input = std::make_shared<Parameter>(ov::element::f16, ov::Shape{1, k});
        std::vector<int32_t> values(2 * k, 0);
        for (size_t channel = 0; channel < 2; ++channel) {
            values[channel * k] = 1;
            values[channel * k + 1] = subnormal ? 0 : 1;
        }
        const float weight_scale = subnormal ? 0.5F : 1.F;
        const float output_scale = subnormal ? 4.F : 3.F;
        auto weights = std::make_shared<Multiply>(
            std::make_shared<Convert>(Constant::create(ov::element::i8, {2, k}, values), ov::element::f16),
            Constant::create(ov::element::f16, {}, {weight_scale}));
        auto fc = std::make_shared<MatMul>(input, weights, false, true);
        auto split = std::make_shared<Split>(fc, Constant::create(ov::element::i64, {}, {-1}), 2);
        auto scaled =
            std::make_shared<Multiply>(split->output(0), Constant::create(ov::element::f16, {}, {output_scale}));
        auto model =
            std::make_shared<ov::Model>(ov::OutputVector{scaled, split->output(1)}, ov::ParameterVector{input});
        auto compiled = core.compile_model(model,
                                           "CPU",
                                           {ov::hint::inference_precision(ov::element::f16),
                                            ov::hint::dynamic_quantization_group_size(0),
                                            ov::num_streams(1),
                                            ov::inference_num_threads(2)});
        auto request = compiled.create_infer_request();
        ov::Tensor tensor(ov::element::f16, {1, k});
        std::fill_n(tensor.data<ov::float16>(), k, ov::float16(0.F));
        tensor.data<ov::float16>()[0] = ov::float16(subnormal ? 0x1p-24F : 1.F);
        tensor.data<ov::float16>()[1] = ov::float16(subnormal ? 0.F : 0.0006F);
        const float accumulator =
            (static_cast<float>(tensor.data<ov::float16>()[0]) + static_cast<float>(tensor.data<ov::float16>()[1])) *
            weight_scale;
        request.set_input_tensor(tensor);
        request.infer();
        EXPECT_EQ(request.get_output_tensor(0).data<const ov::float16>()[0], ov::float16(accumulator * output_scale));
        EXPECT_EQ(request.get_output_tensor(1).data<const ov::float16>()[0], ov::float16(accumulator));
    }
}
}  // namespace
}  // namespace ov::intel_cpu::test
