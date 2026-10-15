// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "fuse_conv_dequantization.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "utils/rt_info/convolution_dequantization.hpp"

namespace ov::intel_cpu {
namespace {

bool is_channel_constant(const ov::Shape& shape, size_t channels) {
    if (shape_size(shape) == 1) {
        return true;
    }
    if (shape.size() > 4) {
        return false;
    }
    ov::Shape aligned(4 - shape.size(), 1);
    aligned.insert(aligned.end(), shape.begin(), shape.end());
    return aligned == ov::Shape{1, channels, 1, 1};
}

}  // namespace

FuseConvolutionDequantization::FuseConvolutionDequantization() {
    using namespace ov::pass::pattern;
    auto convolution = wrap_type<ov::op::v1::Convolution>();
    auto multiply = wrap_type<ov::op::v1::Multiply>({convolution, wrap_type<ov::op::v0::Constant>()});
    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& matcher) {
        const auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(matcher.get_match_root());
        const auto root = matcher.get_match_root();
        const auto conv = ov::as_type_ptr<ov::op::v1::Convolution>(mul ? mul->get_input_node_shared_ptr(0) : root);
        if (conv->get_rt_info().count(ConvolutionInputZeroPoint) != 0) {
            return false;
        }
        const auto consumers = conv->output(0).get_target_inputs();
        if (!mul && (conv->get_output_element_type(0) != ov::element::f32 ||
                     !std::all_of(consumers.begin(), consumers.end(), [](const auto& input) {
                         return ov::is_type<ov::op::v0::FakeQuantize>(input.get_node());
                     }))) {
            return false;
        }
        const auto weights = ov::as_type_ptr<ov::op::v0::Constant>(conv->get_input_node_shared_ptr(1));
        const auto scales = mul ? ov::as_type_ptr<ov::op::v0::Constant>(mul->get_input_node_shared_ptr(1)) : nullptr;
        if (!weights || weights->get_element_type() != ov::element::i8 || weights->get_shape().size() != 4 ||
            conv->get_input_partial_shape(0).is_dynamic() || conv->get_input_shape(0).size() != 4 ||
            conv->get_strides()[0] != conv->get_strides()[1] || conv->get_dilations() != ov::Strides{1, 1} ||
            ov::shape_size(weights->get_shape()) == 0 || ov::shape_size(conv->get_input_shape(0)) == 0 ||
            std::any_of(conv->get_pads_begin().begin(),
                        conv->get_pads_begin().end(),
                        [](auto pad) {
                            return pad < 0;
                        }) ||
            std::any_of(conv->get_pads_end().begin(),
                        conv->get_pads_end().end(),
                        [](auto pad) {
                            return pad < 0;
                        }) ||
            (root->get_output_element_type(0) != ov::element::f16 &&
             root->get_output_element_type(0) != ov::element::f32) ||
            (scales && !is_channel_constant(scales->get_shape(), weights->get_shape()[0]))) {
            return false;
        }
        auto data = conv->input_value(0);
        const auto subtract = ov::as_type_ptr<ov::op::v1::Subtract>(data.get_node_shared_ptr());
        if (subtract) {
            data = subtract->input_value(0);
        }
        if (const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(data.get_node_shared_ptr())) {
            data = convert->input_value(0);
        }
        const auto type = data.get_element_type();
        if ((type != ov::element::u8 && type != ov::element::i8) || (!subtract && type == ov::element::i8)) {
            return false;
        }
        int32_t zero_point = 0;
        if (subtract) {
            const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(subtract->get_input_node_shared_ptr(1));
            if (!constant || !is_channel_constant(constant->get_shape(), conv->get_input_shape(0)[1])) {
                return false;
            }
            const auto values = constant->cast_vector<float>();
            const auto value = values.front();
            const int minimum = type == ov::element::u8 ? 0 : -128;
            const int maximum = type == ov::element::u8 ? 255 : 127;
            if (!std::isfinite(value) || value < minimum || value > maximum || value != std::nearbyint(value) ||
                !std::all_of(values.begin(), values.end(), [value](float item) {
                    return item == value;
                })) {
                return false;
            }
            zero_point = static_cast<int32_t>(value);
        }
        auto values = scales ? scales->cast_vector<float>() : std::vector<float>{1.f};
        if (!std::all_of(values.begin(), values.end(), [](float value) {
                return std::isfinite(value) && value > 0;
            })) {
            return false;
        }
        auto replacement = conv->clone_with_new_inputs({data, conv->input_value(1)});
        auto relaxed = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(replacement);
        if (!relaxed) {
            return false;
        }
        relaxed->set_overridden_output_type(ov::element::f32, 0);
        replacement->validate_and_infer_types();
        ov::copy_runtime_info({conv, root}, replacement);
        std::ostringstream serialized_scales;
        serialized_scales << std::setprecision(std::numeric_limits<float>::max_digits10);
        for (size_t i = 0; i < values.size(); ++i) {
            if (i != 0) {
                serialized_scales << ' ';
            }
            serialized_scales << values[i];
        }
        replacement->get_rt_info()[ConvolutionInputZeroPoint] = zero_point;
        replacement->get_rt_info()[ConvolutionDequantizationScales] = serialized_scales.str();
        replacement->set_friendly_name(conv->get_friendly_name());
        if (root->get_output_element_type(0) == ov::element::f16) {
            const auto convert = std::make_shared<ov::op::v0::Convert>(replacement, ov::element::f16);
            ov::copy_runtime_info(root, convert);
            convert->set_friendly_name(root->get_friendly_name());
            ov::replace_node(root, convert);
        } else {
            ov::replace_node(root, replacement);
        }
        return true;
    };
    const auto root = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{multiply, convolution});
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(root, "FuseConvolutionDequantization"), callback);
}

}  // namespace ov::intel_cpu
