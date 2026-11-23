// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/core/visibility.hpp"

#if defined(OPENVINO_ARCH_X86_64)
#    include <cstddef>
#    include <cstdint>
#    include <memory>
#    include <set>
#    include <string>
#    include <vector>

#    include "internal_properties.hpp"
#    include "openvino/openvino.hpp"
#    include "openvino/opsets/opset13.hpp"
#    include "openvino/runtime/intel_cpu/properties.hpp"
#    include "utils/precision_support.h"

namespace ov::intel_cpu::test {
namespace {
void check_tensor_parallel(ov::element::Type weight_type,
                           bool kn,
                           bool grouped,
                           bool per_channel,
                           size_t n = 64,
                           size_t k = 64) {
    using namespace ov::opset13;
    size_t groups = 1;
    if (grouped) {
        groups = k == 33 ? 3 : 2;
    }
    const size_t group_size = k / groups;
    const size_t parameter_channels = per_channel ? n : 1;
    const bool signed_weights = weight_type == ov::element::i4 || weight_type == ov::element::i8;
    std::vector<int32_t> quantized(n * k);
    for (size_t channel = 0; channel < n; ++channel) {
        for (size_t feature = 0; feature < k; ++feature) {
            quantized[kn ? feature * n + channel : channel * k + feature] =
                static_cast<int32_t>((channel + feature) % 8) - (signed_weights ? 4 : 0);
        }
    }
    std::vector<float> scales(parameter_channels * groups);
    std::vector<float> zero_points(scales.size());
    for (size_t channel = 0; channel < parameter_channels; ++channel) {
        for (size_t group = 0; group < groups; ++group) {
            const size_t index = kn ? group * parameter_channels + channel : channel * groups + group;
            scales[index] = static_cast<float>(1 + group + channel % 2) * 0.25F;
            zero_points[index] = static_cast<float>(group + channel % 3) - (signed_weights ? 2.F : 0.F);
        }
    }
    const ov::Shape weight_shape = grouped ? (kn ? ov::Shape{groups, group_size, n} : ov::Shape{n, groups, group_size})
                                           : (kn ? ov::Shape{k, n} : ov::Shape{n, k});
    const ov::Shape parameter_shape =
        grouped ? (kn ? ov::Shape{groups, 1, parameter_channels} : ov::Shape{parameter_channels, groups, 1})
                : (kn ? ov::Shape{1, parameter_channels} : ov::Shape{parameter_channels, 1});
    auto weights = Constant::create(weight_type, weight_shape, quantized);
    auto scale = Constant::create(ov::element::f16, parameter_shape, scales);
    auto zero_point = Constant::create(ov::element::f16, parameter_shape, zero_points);
    std::shared_ptr<ov::Node> dequantized = std::make_shared<Multiply>(
        std::make_shared<Subtract>(std::make_shared<Convert>(weights, ov::element::f16), zero_point),
        scale);
    if (grouped) {
        dequantized = std::make_shared<Reshape>(
            dequantized,
            Constant::create(ov::element::i64,
                             ov::Shape{2},
                             kn ? std::vector<int64_t>{static_cast<int64_t>(k), static_cast<int64_t>(n)}
                                : std::vector<int64_t>{static_cast<int64_t>(n), static_cast<int64_t>(k)}),
            false);
    }
    auto input = std::make_shared<Parameter>(ov::element::f16, ov::PartialShape{-1, static_cast<int64_t>(k)});
    auto output = std::make_shared<MatMul>(input, dequantized, false, !kn);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{input});
    ov::Core core;
    auto compiled = core.compile_model(model,
                                       "CPU",
                                       {ov::hint::inference_precision(ov::element::f16),
                                        ov::hint::model_distribution_policy(std::set<ov::hint::ModelDistributionPolicy>{
                                            ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL}),
                                        ov::intel_cpu::enable_tensor_parallel(true),
                                        ov::inference_num_threads(1),
                                        ov::num_streams(1)});
    auto request = compiled.create_infer_request();
    for (const size_t rows : {1U, 4U, 1U}) {
        ov::Tensor source(ov::element::f16, ov::Shape{rows, k});
        auto* values = source.data<ov::float16>();
        for (size_t row = 0; row < rows; ++row) {
            // One-hot rows select weights from both K groups without a second GEMM reference.
            const size_t feature = 1 + (row % groups) * group_size;
            for (size_t column = 0; column < k; ++column) {
                values[row * k + column] = ov::float16(column == feature ? 1.F : 0.F);
            }
        }
        request.set_input_tensor(source);
        request.infer();
        const auto result = request.get_output_tensor();
        ASSERT_EQ(result.get_shape(), (ov::Shape{rows, n}));
        ASSERT_NE(result.data<ov::float16>(), nullptr);
        for (size_t row = 0; row < rows; ++row) {
            const size_t feature = 1 + (row % groups) * group_size;
            const size_t group = feature / group_size;
            for (size_t channel = 0; channel < n; ++channel) {
                const size_t parameter_channel = per_channel ? channel : 0;
                const float value = static_cast<float>((channel + feature) % 8) - (signed_weights ? 4.F : 0.F);
                const float zero = static_cast<float>(group + parameter_channel % 3) - (signed_weights ? 2.F : 0.F);
                const float scale_value = static_cast<float>(1 + group + parameter_channel % 2) * 0.25F;
                EXPECT_EQ(result.data<ov::float16>()[row * n + channel], ov::float16((value - zero) * scale_value))
                    << "rows=" << rows << " row=" << row << " channel=" << channel;
            }
        }
    }
}
}  // namespace

TEST(FP16CompressedFCTensorParallel, BroadcastScalesPreserveInference) {
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    for (const auto type : {ov::element::i4, ov::element::u4, ov::element::i8, ov::element::u8}) {
        for (const bool kn : {false, true}) {
            SCOPED_TRACE(type.to_string() + (kn ? "/KN" : "/NK"));
            check_tensor_parallel(type, kn, false, false);
        }
    }
}

TEST(FP16CompressedFCTensorParallel, GroupedBroadcastScalesAndZeroPointsPreserveInference) {
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    for (const auto type : {ov::element::i4, ov::element::u4, ov::element::i8, ov::element::u8}) {
        for (const bool kn : {false, true}) {
            SCOPED_TRACE(type.to_string() + (kn ? "/KN" : "/NK"));
            check_tensor_parallel(type, kn, true, false);
        }
    }
}

TEST(FP16CompressedFCTensorParallel, PerChannelGroupedParametersPreserveInference) {
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    for (const auto type : {ov::element::i4, ov::element::u4, ov::element::i8, ov::element::u8}) {
        for (const bool kn : {false, true}) {
            SCOPED_TRACE(type.to_string() + (kn ? "/KN" : "/NK"));
            for (const size_t n : {64U, 65U}) {
                SCOPED_TRACE("N=" + std::to_string(n));
                check_tensor_parallel(type, kn, true, true, n);
            }
        }
    }
}

TEST(FP16CompressedFCTensorParallel, OddKPreservesPackedPartitionBoundaries) {
    if (!hasFp16WeightsDecompressionSupport()) {
        GTEST_SKIP();
    }
    for (const auto type : {ov::element::i4, ov::element::u4}) {
        for (const bool kn : {false, true}) {
            SCOPED_TRACE(type.to_string() + (kn ? "/KN" : "/NK"));
            check_tensor_parallel(type, kn, true, true, 18, 33);
        }
    }
}
}  // namespace ov::intel_cpu::test
#endif
