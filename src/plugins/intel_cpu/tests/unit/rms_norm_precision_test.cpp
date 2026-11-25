// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "graph.h"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "ov_ops/rms.hpp"
#include "transformations/cpu_opset/common/op/power_static.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include "utils/precision_support.h"

namespace ov::intel_cpu::test {
namespace {
struct RMSPrecisionCase {
    ov::element::Type source;
    ov::element::Type destination;
    size_t channels;
    size_t gamma_size;
};

float read_value(const ov::Tensor& tensor, size_t index) {
    if (tensor.get_element_type() == ov::element::f16) {
        return tensor.data<const ov::float16>()[index];
    }
    if (tensor.get_element_type() == ov::element::bf16) {
        return tensor.data<const ov::bfloat16>()[index];
    }
    return tensor.data<const float>()[index];
}

float rounded(float value, const ov::element::Type& precision) {
    if (precision == ov::element::f16) {
        return ov::float16(value);
    }
    if (precision == ov::element::bf16) {
        return ov::bfloat16(value);
    }
    return value;
}

class RMSNormPrecision : public testing::TestWithParam<RMSPrecisionCase> {};

TEST_P(RMSNormPrecision, PreservesIndependentOutputPrecision) {
    if (!ov::with_cpu_x86_avx2()) {
        GTEST_SKIP();
    }
    const auto& p = GetParam();
    constexpr float epsilon = 1e-5F;
    std::vector<float> gamma(p.gamma_size);
    for (size_t i = 0; i < gamma.size(); ++i) {
        // Keep gamma exact under the plugin's independent constant-precision conversion.
        gamma[i] = 1.25F + static_cast<float>(i % 7) * 0.03125F;
    }
    auto input =
        std::make_shared<ov::op::v0::Parameter>(p.source, ov::PartialShape{-1, -1, static_cast<int64_t>(p.channels)});
    auto rms = p.gamma_size == 0 ? std::make_shared<ov::op::internal::RMS>(input, epsilon, p.destination)
                                 : std::make_shared<ov::op::internal::RMS>(
                                       input,
                                       ov::op::v0::Constant::create(ov::element::f32, ov::Shape{p.gamma_size}, gamma),
                                       epsilon,
                                       p.destination);
    for (const auto source : {ov::element::f32, ov::element::f16, ov::element::bf16}) {
        for (const auto destination : {ov::element::f32, ov::element::f16, ov::element::bf16}) {
            ov::disable_conversion(rms, source, destination);
        }
    }
    auto model = std::make_shared<ov::Model>(ov::OutputVector{rms}, ov::ParameterVector{input});
    ov::Core core;
    auto compiled =
        core.compile_model(model,
                           "CPU",
                           {ov::hint::inference_precision(p.source), ov::inference_num_threads(2), ov::num_streams(1)});
    auto request = compiled.create_infer_request();
    for (const size_t rows : {1U, 5U, 2U}) {
        ov::Tensor tensor(p.source, ov::Shape{1, rows, p.channels});
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            const float value = static_cast<float>(static_cast<int>(i * 17 % 31) - 15) * 0.0625F;
            if (p.source == ov::element::f16) {
                tensor.data<ov::float16>()[i] = ov::float16(value);
            } else if (p.source == ov::element::bf16) {
                tensor.data<ov::bfloat16>()[i] = ov::bfloat16(value);
            } else {
                tensor.data<float>()[i] = value;
            }
        }
        request.set_input_tensor(tensor);
        request.infer();
        const auto output = request.get_output_tensor();
        ASSERT_EQ(output.get_element_type(), p.destination);
        ASSERT_EQ(output.get_shape(), tensor.get_shape());
        for (size_t row = 0; row < rows; ++row) {
            float sum = 0.F;
            for (size_t channel = 0; channel < p.channels; ++channel) {
                const float value = read_value(tensor, row * p.channels + channel);
                sum += value * value;
            }
            const float norm = 1.F / std::sqrt(sum / static_cast<float>(p.channels) + epsilon);
            for (size_t channel = 0; channel < p.channels; ++channel) {
                const float scale = gamma.empty() ? 1.F : gamma[channel % gamma.size()];
                const float expected =
                    rounded(read_value(tensor, row * p.channels + channel) * norm * scale, p.destination);
                const float tolerance = p.destination == ov::element::bf16 ? std::abs(expected) * 0.008F : 2e-5F;
                EXPECT_NEAR(read_value(output, row * p.channels + channel), expected, tolerance)
                    << "row=" << row << " channel=" << channel;
            }
        }
        size_t count = 0;
        for (const auto& node : compiled.get_runtime_model()->get_ordered_ops()) {
            const auto& info = node->get_rt_info();
            if (info.count("layerType") && info.at("layerType").as<std::string>() == "RMS") {
                ++count;
                EXPECT_EQ(node->input_value(0).get_element_type(),
                          hasHardwareSupport(p.source) ? p.source : ov::element::f32);
                EXPECT_EQ(node->get_output_element_type(0), p.destination);
            }
        }
        EXPECT_EQ(count, 1U);
    }
}

std::vector<RMSPrecisionCase> precision_cases() {
    std::vector<RMSPrecisionCase> cases;
    for (const auto source : {ov::element::f32, ov::element::f16, ov::element::bf16}) {
        for (const auto destination : {ov::element::f32, ov::element::f16, ov::element::bf16}) {
            for (const size_t channels : {9U, 64U}) {
                for (const size_t gamma_size : {size_t{0}, size_t{1}, channels}) {
                    cases.push_back({source, destination, channels, gamma_size});
                }
            }
        }
    }
    return cases;
}

INSTANTIATE_TEST_SUITE_P(MixedPrecision, RMSNormPrecision, testing::ValuesIn(precision_cases()));

TEST(RMSNormFusion, ScalarScaleKeepsFp32UntilFp16Store) {
    if (!ov::with_cpu_x86_avx2()) {
        GTEST_SKIP();
    }
    constexpr float epsilon = 1e-5F;
    for (const size_t channels : {9U, 64U}) {
        for (const size_t gamma_size : {size_t{1}, channels}) {
            for (const float scale : {1.F / 128.F, -0.5F}) {
                for (const float gamma : {1.25F, 65536.F}) {
                    SCOPED_TRACE("channels=" + std::to_string(channels) + " gamma_size=" + std::to_string(gamma_size) +
                                 " scale=" + std::to_string(scale) + " gamma=" + std::to_string(gamma));
                    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, channels});
                    auto rms = std::make_shared<ov::op::internal::RMS>(
                        input,
                        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{gamma_size}, {gamma}),
                        epsilon,
                        ov::element::f32);
                    auto scaled = std::make_shared<PowerStaticNode>(rms, 1.F, scale, 0.F, ov::element::f16);
                    auto model = std::make_shared<ov::Model>(ov::OutputVector{scaled}, ov::ParameterVector{input});
                    ov::Core core;
                    auto compiled = core.compile_model(model,
                                                       "CPU",
                                                       {ov::hint::inference_precision(ov::element::f32),
                                                        ov::inference_num_threads(2),
                                                        ov::num_streams(1)});
                    auto request = compiled.create_infer_request();
                    auto tensor = request.get_input_tensor();
                    for (size_t row = 0; row < 2; ++row) {
                        for (size_t channel = 0; channel < channels; ++channel) {
                            tensor.data<float>()[row * channels + channel] = row == 0 ? 1.F : 2.F;
                        }
                    }
                    request.infer();
                    const auto output = request.get_output_tensor();
                    for (size_t row = 0; row < 2; ++row) {
                        const float value = row == 0 ? 1.F : 2.F;
                        const float norm = 1.F / std::sqrt(value * value + epsilon);
                        const auto expected = ov::float16(value * (norm * gamma) * scale);
                        for (size_t channel = 0; channel < channels; ++channel) {
                            EXPECT_EQ(output.data<const ov::float16>()[row * channels + channel], expected);
                        }
                    }
                    size_t rms_count = 0, eltwise_count = 0;
                    for (const auto& node : compiled.get_runtime_model()->get_ordered_ops()) {
                        const auto& info = node->get_rt_info();
                        if (!info.count("layerType")) {
                            continue;
                        }
                        const auto type = info.at("layerType").as<std::string>();
                        if (type == "RMS") {
                            ++rms_count;
                            EXPECT_EQ(node->get_output_element_type(0), ov::element::f16);
                        } else if (type == "Eltwise") {
                            ++eltwise_count;
                        }
                    }
                    EXPECT_EQ(rms_count, 1U);
                    EXPECT_EQ(eltwise_count, 0U);
                }
            }
        }
    }
}

TEST(RMSNormFusion, SignedZeroMatchesSeparatePowerStatic) {
    if (!ov::with_cpu_x86_avx2()) {
        GTEST_SKIP();
    }
    for (const size_t channels : {9U, 64U}) {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, channels});
        auto rms = std::make_shared<ov::op::internal::RMS>(input, 1e-5, ov::element::f32);
        auto scaled = std::make_shared<PowerStaticNode>(rms, 1.F, 0.5F, 0.F, ov::element::f16);
        auto model = std::make_shared<ov::Model>(ov::OutputVector{scaled}, ov::ParameterVector{input});
        ov::Core core;
        auto compiled = core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32));
        auto request = compiled.create_infer_request();
        auto tensor = request.get_input_tensor();
        for (size_t i = 0; i < channels; ++i) {
            tensor.data<float>()[i] = -0.F;
        }
        request.infer();
        const auto output = request.get_output_tensor();
        for (size_t i = 0; i < channels; ++i) {
            const float value = output.data<const ov::float16>()[i];
            EXPECT_EQ(value, 0.F);
            EXPECT_FALSE(std::signbit(value));
        }
    }
}

TEST(RMSNormFusion, CacheDistinguishesOutputScaleAndPrecision) {
    if (!ov::with_cpu_x86_avx2()) {
        GTEST_SKIP();
    }
    ov::ParameterVector inputs;
    ov::OutputVector outputs;
    const std::vector<float> scales{1.F / 128.F, 1.F / 64.F, 1.F / 128.F};
    const ov::element::TypeVector precisions{ov::element::f16, ov::element::f16, ov::element::f32};
    for (size_t i = 0; i < scales.size(); ++i) {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 9});
        inputs.push_back(input);
        auto rms = std::make_shared<ov::op::internal::RMS>(input, 1e-5, ov::element::f32);
        outputs.push_back(std::make_shared<PowerStaticNode>(rms, 1.F, scales[i], 0.F, precisions[i]));
    }
    auto model = std::make_shared<ov::Model>(outputs, inputs);
    ov::Core core;
    auto compiled = core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32));
    auto request = compiled.create_infer_request();
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto tensor = request.get_input_tensor(i);
        for (size_t j = 0; j < tensor.get_size(); ++j) {
            tensor.data<float>()[j] = 1.F;
        }
    }
    request.infer();
    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto tensor = request.get_output_tensor(i);
        const float expected = rounded((1.F / std::sqrt(1.F + 1e-5F)) * scales[i], precisions[i]);
        for (size_t j = 0; j < tensor.get_size(); ++j) {
            EXPECT_FLOAT_EQ(read_value(tensor, j), expected);
        }
    }
}

TEST(RMSNormFusion, RetainsRoundingNonlinearOperationsAndOtherConsumers) {
    if (!ov::with_cpu_x86_avx2()) {
        GTEST_SKIP();
    }
    struct RejectedCase {
        const char* name;
        ov::element::Type rms_precision;
        float power;
        float scale;
        float shift;
        bool other_consumer;
    };
    const std::vector<RejectedCase> cases{
        {"IntermediateFP16Rounding", ov::element::f16, 1.F, 0.5F, 0.F, false},
        {"NonlinearPower", ov::element::f32, 2.F, 0.5F, 0.F, false},
        {"NonzeroShift", ov::element::f32, 1.F, 0.5F, 1.F, false},
        {"OtherConsumer", ov::element::f32, 1.F, 0.5F, 0.F, true},
        {"NegativeZeroShift", ov::element::f32, 1.F, 0.5F, -0.F, false},
        {"ZeroScale", ov::element::f32, 1.F, 0.F, 0.F, false},
        {"InfiniteScale", ov::element::f32, 1.F, std::numeric_limits<float>::infinity(), 0.F, false}};
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 9});
        auto rms = std::make_shared<ov::op::internal::RMS>(input, 1e-5, test_case.rms_precision);
        auto scaled =
            std::make_shared<PowerStaticNode>(rms, test_case.power, test_case.scale, test_case.shift, ov::element::f16);
        ov::OutputVector outputs{scaled};
        if (test_case.other_consumer) {
            outputs.push_back(rms);
        }
        auto model = std::make_shared<ov::Model>(outputs, ov::ParameterVector{input});
        Config config;
        auto context = std::make_shared<GraphContext>(config, nullptr, false);
        Graph graph;
        graph.Init(model, context);
        size_t count = 0;
        for (const auto& node : graph.GetNodes()) {
            if (node->getType() == Type::RMS) {
                ++count;
                EXPECT_TRUE(node->getFusedWith().empty());
            }
        }
        EXPECT_EQ(count, 1U);
    }
}
}  // namespace
}  // namespace ov::intel_cpu::test
