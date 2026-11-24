// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/kernels/paged_causal_conv1d.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "cpu_parallel.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/paged_causal_conv1d.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"
#include "utils/precision_support.h"

#if OV_THREAD_USE_TBB
#    include <oneapi/tbb/task_arena.h>
#endif

namespace ov::intel_cpu::test {
namespace {

float read_element(const ov::Tensor& tensor, size_t index) {
    if (tensor.get_element_type() == ov::element::f32) {
        return tensor.data<const float>()[index];
    }
    if (tensor.get_element_type() == ov::element::f16) {
        return static_cast<float>(tensor.data<const ov::float16>()[index]);
    }
    return static_cast<float>(tensor.data<const ov::bfloat16>()[index]);
}

void write_element(ov::Tensor& tensor, size_t index, float value) {
    if (tensor.get_element_type() == ov::element::f32) {
        tensor.data<float>()[index] = value;
    } else if (tensor.get_element_type() == ov::element::f16) {
        tensor.data<ov::float16>()[index] = ov::float16(value);
    } else {
        tensor.data<ov::bfloat16>()[index] = ov::bfloat16(value);
    }
}

// Public inference exercises the runtime ISA dispatcher; direct execution also covers
// independent data/state precisions without graph precision conversion of Parameters.
using ConvParams = std::tuple<ov::element::Type, ov::element::Type, size_t, size_t, size_t, bool, bool>;

class PagedCausalConv1DTest : public testing::TestWithParam<ConvParams> {};

TEST_P(PagedCausalConv1DTest, DynamicSequencesOutputAndCache) {
    const auto [data_type, state_type, hidden, taps, dilation, has_bias, public_api] = GetParam();
    if (public_api && !hasHardwareSupport(data_type)) {
        GTEST_SKIP() << "Data precision is not supported natively.";
    }
    using namespace ov::opset13;
    const size_t window = (taps - 1) * dilation + 1;
    const size_t stride = hidden * window;
    constexpr size_t blocks = 36;
    const std::vector<int32_t> block_begins{0, 12, 24, 36};
    std::vector<int32_t> block_indices(blocks);
    for (size_t i = 0; i < blocks; i++) {
        block_indices[i] = static_cast<int32_t>((i + 5) % blocks);
    }
    const std::vector<int32_t> intervals{16, 0, 1};
    const std::vector<int32_t> past{15, 0, 3};
    ov::Tensor weight(data_type, {hidden, 1, taps});
    ov::Tensor bias(data_type, {has_bias ? hidden : 0});
    for (size_t i = 0; i < weight.get_size(); i++) {
        write_element(weight, i, static_cast<float>(static_cast<int>(i * 7 % 23) - 11) / 32.F);
    }
    for (size_t i = 0; i < bias.get_size(); i++) {
        write_element(bias, i, static_cast<float>(static_cast<int>(i % 9) - 4) / 16.F);
    }
    if (has_bias) {
        // Small terms added to a large bias must accumulate in f32 before output rounding.
        write_element(bias, 0, 1024.F);
        for (size_t k = 0; k < taps; k++) {
            write_element(weight, k, 1.F);
        }
    }

    ov::InferRequest request;
    if (public_api) {
        auto input = std::make_shared<Parameter>(data_type, ov::PartialShape{-1, static_cast<int64_t>(hidden)});
        auto state = std::make_shared<Parameter>(state_type, ov::Shape{blocks, hidden, window});
        auto begins = std::make_shared<Parameter>(ov::element::i32, ov::Shape{4});
        auto conv = std::make_shared<ov::op::internal::PagedCausalConv1D>(
            input,
            state,
            std::make_shared<Constant>(weight),
            std::make_shared<Constant>(bias),
            begins,
            Constant::create(ov::element::i32, {blocks}, block_indices),
            Constant::create(ov::element::i32, {4}, block_begins),
            Constant::create(ov::element::i32, {3}, past),
            Constant::create(ov::element::i32, {3}, intervals),
            dilation);
        auto model = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input, state, begins});
        ov::Core core;
        auto compiled = core.compile_model(
            model,
            "CPU",
            {ov::hint::inference_precision(data_type), ov::inference_num_threads(2), ov::num_streams(1)});
        request = compiled.create_infer_request();
    }
    auto cpu_parallel = std::make_shared<CpuParallel>(ov::intel_cpu::TbbPartitioner::STATIC);
    std::vector<float> scratch(static_cast<size_t>(cpu_parallel->get_num_worker_threads()) * stride);
    for (const auto& lengths : {std::vector<int32_t>{3, 1, 0}, {1, 5, 3}, {257, 2, 1}, {1, 1, 1}}) {
        std::vector<int32_t> begins{0};
        for (auto length : lengths) {
            begins.push_back(begins.back() + length);
        }
        const auto tokens = static_cast<size_t>(begins.back());
        ov::Tensor input(data_type, {tokens, hidden});
        ov::Tensor state(state_type, {blocks, hidden, window});
        ov::Tensor expected_state(state_type, state.get_shape());
        ov::Tensor expected_output(data_type, input.get_shape());
        for (size_t i = 0; i < input.get_size(); i++) {
            write_element(input, i, static_cast<float>(static_cast<int>(i * 17 % 61) - 30) / 32.F);
        }
        if (has_bias) {
            for (size_t t = 0; t < tokens; t++) {
                write_element(input, t * hidden, 0.25F);
            }
        }
        for (size_t i = 0; i < state.get_size(); i++) {
            const auto value = static_cast<float>(static_cast<int>(i * 13 % 37) - 18) / 32.F;
            write_element(state, i, value);
            write_element(expected_state, i, value);
        }
        for (size_t s = 0; s < lengths.size(); s++) {
            std::vector<float> local(stride);
            const auto read_block = static_cast<size_t>(block_indices[block_begins[s]]);
            for (size_t i = 0; i < stride; i++) {
                local[i] = read_element(state, read_block * stride + i);
            }
            for (int32_t t = 0; t < lengths[s]; t++) {
                const auto token = static_cast<size_t>(begins[s]) + static_cast<size_t>(t);
                for (size_t h = 0; h < hidden; h++) {
                    for (size_t k = 0; k + 1 < window; k++) {
                        local[h * window + k] = local[h * window + k + 1];
                    }
                    local[h * window + window - 1] = read_element(input, token * hidden + h);
                    float sum = has_bias ? read_element(bias, h) : 0.F;
                    for (size_t k = 0; k < taps; k++) {
                        sum += local[h * window + k * dilation] * read_element(weight, h * taps + k);
                    }
                    write_element(expected_output, token * hidden + h, sum);
                }
                const int32_t cached = (intervals[s] ? past[s] % intervals[s] : 0) + t + 1;
                if ((intervals[s] && cached % intervals[s] == 0) || t + 1 == lengths[s]) {
                    const auto slot = intervals[s] ? 1 + (cached - 1) / intervals[s] : 1;
                    if (slot < block_begins[s + 1] - block_begins[s]) {
                        const auto physical = static_cast<size_t>(block_indices[block_begins[s] + slot]);
                        for (size_t i = 0; i < stride; i++) {
                            write_element(expected_state, physical * stride + i, local[i]);
                        }
                    }
                }
            }
        }
        ov::Tensor output(data_type, input.get_shape());
        if (public_api) {
            ov::Tensor begins_tensor(ov::element::i32, {begins.size()});
            std::copy(begins.begin(), begins.end(), begins_tensor.data<int32_t>());
            request.set_input_tensor(0, input);
            request.set_input_tensor(1, state);
            request.set_input_tensor(2, begins_tensor);
            request.infer();
            output = request.get_output_tensor();
        } else {
            const auto execute = [&] {
                ov::Extensions::Cpu::XARCH::paged_causal_conv1d_exec(input.data(),
                                                                     state.data(),
                                                                     weight.data(),
                                                                     has_bias ? bias.data() : nullptr,
                                                                     has_bias,
                                                                     begins.data(),
                                                                     block_indices.data(),
                                                                     block_begins.data(),
                                                                     past.data(),
                                                                     intervals.data(),
                                                                     output.data(),
                                                                     tokens,
                                                                     hidden,
                                                                     taps,
                                                                     dilation,
                                                                     lengths.size(),
                                                                     data_type,
                                                                     state_type,
                                                                     scratch.data(),
                                                                     cpu_parallel);
            };
#if OV_THREAD_USE_TBB
            // Match the CPU executor's arena even when a single channel block
            // takes the serial branch of parallel_for2d.
            tbb::task_arena arena;
            arena.execute(execute);
#else
            execute();
#endif
        }
        for (size_t i = 0; i < output.get_size(); i++) {
            if (data_type == ov::element::bf16) {
                // Existing SIMD stores round to nearest, whereas the scalar bfloat16
                // constructor can truncate a value with an even low mantissa bit.
                const auto actual = static_cast<int>(output.data<const ov::bfloat16>()[i].to_bits());
                const auto expected = static_cast<int>(expected_output.data<const ov::bfloat16>()[i].to_bits());
                ASSERT_LE(std::abs(actual - expected), 1) << "output element " << i;
            } else {
                ASSERT_EQ(read_element(output, i), read_element(expected_output, i)) << "output element " << i;
            }
        }
        for (size_t i = 0; i < state.get_size(); i++) {
            ASSERT_EQ(read_element(state, i), read_element(expected_state, i)) << "state element " << i;
        }
    }
}

std::vector<ConvParams> conv_cases() {
    std::vector<ConvParams> cases;
    for (auto data : {ov::element::f32, ov::element::f16, ov::element::bf16}) {
        for (auto state : {ov::element::f32, ov::element::f16, ov::element::bf16}) {
            for (bool public_api : {false, true}) {
                if (public_api && data != state) {
                    continue;
                }
                cases.emplace_back(data, state, 67, 4, 1, true, public_api);
                cases.emplace_back(data, state, 17, 3, 1, false, public_api);
                cases.emplace_back(data, state, 7, 3, 2, true, public_api);
                cases.emplace_back(data, state, 67, 4, 3, false, public_api);
                cases.emplace_back(data, state, 1, 2, 2, false, public_api);
            }
        }
    }
    return cases;
}

INSTANTIATE_TEST_SUITE_P(CausalWindow, PagedCausalConv1DTest, testing::ValuesIn(conv_cases()));

}  // namespace
}  // namespace ov::intel_cpu::test
