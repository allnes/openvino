// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "move_fc_scales_before_split.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_cpu {

MoveFCScalesBeforeSplit::MoveFCScalesBeforeSplit() {
    MATCHER_SCOPE(MoveFCScalesBeforeSplit);
    auto split_m = ov::pass::pattern::wrap_type<ov::op::v1::Split, ov::op::v1::VariadicSplit>();
    ov::matcher_pass_callback callback = [this](ov::pass::pattern::Matcher& matcher) {
        const auto split = matcher.get_match_root();
        const auto data = split->input_value(0);
        const auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(data.get_node_shared_ptr());
        const auto& shape = data.get_partial_shape();
        const auto axis = ov::as_type_ptr<ov::op::v0::Constant>(split->get_input_node_shared_ptr(1));
        if (transformation_callback(split) || !matmul || matmul->get_transpose_a() ||
            !ov::op::util::is_on_path<ov::op::v0::Constant>(matmul->input_value(1)) ||
            data.get_element_type() != ov::element::f16 || data.get_target_inputs().size() != 1 ||
            shape.rank().is_dynamic() || shape.size() < 2 || shape[shape.size() - 1].is_dynamic() || !axis ||
            ov::shape_size(axis->get_shape()) != 1) {
            return false;
        }
        const auto rank = static_cast<int64_t>(shape.size());
        const auto split_axis = axis->cast_vector<int64_t>()[0];
        if (split_axis != -1 && split_axis != rank - 1) {
            return false;
        }

        std::vector<float> scales;
        ov::NodeVector multiplies;
        for (const auto& output : split->outputs()) {
            const auto& output_shape = output.get_partial_shape();
            if (output_shape.rank().is_dynamic() || output_shape.size() != shape.size() ||
                output_shape[shape.size() - 1].is_dynamic()) {
                return false;
            }
            float scale = 1.F;
            bool scaled = false;
            bool unscaled = false;
            for (const auto& input : output.get_target_inputs()) {
                const auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(input.get_node()->shared_from_this());
                const auto constant = multiply ? ov::as_type_ptr<ov::op::v0::Constant>(
                                                     multiply->get_input_node_shared_ptr(1 - input.get_index()))
                                               : nullptr;
                if (!constant || ov::shape_size(constant->get_shape()) != 1 ||
                    constant->get_element_type() != data.get_element_type() ||
                    multiply->get_output_element_type(0) != data.get_element_type() ||
                    multiply->get_output_partial_shape(0) != output_shape ||
                    multiply->get_autob().m_type != ov::op::AutoBroadcastType::NUMPY) {
                    unscaled = true;
                    continue;
                }
                if (multiply->get_rt_info().count(ov::DisablePrecisionConversion::get_type_info_static()) != 0U) {
                    return false;
                }
                if (const auto relaxed = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(multiply)) {
                    for (size_t port = 0; port < 2; ++port) {
                        const auto& type = relaxed->get_origin_input_type(port);
                        if (type != ov::element::dynamic && type != data.get_element_type()) {
                            return false;
                        }
                    }
                }
                const float value = constant->cast_vector<float>()[0];
                if (!std::isfinite(value) || value <= 0.F || (scaled && value != scale)) {
                    return false;
                }
                scale = value;
                scaled = true;
                multiplies.push_back(multiply);
            }
            // A shared raw output cannot be changed when only some consumers scale it.
            if (scaled && unscaled) {
                return false;
            }
            scales.insert(scales.end(), output_shape[shape.size() - 1].get_length(), scale);
        }
        if (multiplies.empty() || scales.size() != static_cast<size_t>(shape[shape.size() - 1].get_length())) {
            return false;
        }

        auto constant = ov::op::v0::Constant::create(data.get_element_type(), ov::Shape{scales.size()}, scales);
        auto multiply = std::make_shared<ov::op::v1::Multiply>(data, constant);
        multiply->set_friendly_name(matmul->get_friendly_name() + "/output_scales");
        ov::copy_runtime_info(multiplies, multiply);
        split->input(0).replace_source_output(multiply);
        for (const auto& node : multiplies) {
            const auto port = ov::is_type<ov::op::v0::Constant>(node->get_input_node_shared_ptr(0)) ? 1 : 0;
            node->output(0).replace(node->input_value(port));
        }
        return true;
    };
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(split_m, matcher_name), callback);
}

}  // namespace ov::intel_cpu
