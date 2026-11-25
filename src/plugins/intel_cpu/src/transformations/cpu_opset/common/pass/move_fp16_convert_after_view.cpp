// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "move_fp16_convert_after_view.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>

#include "cpu_types.h"
#include "nodes/executors/eltwise_config.hpp"
#include "nodes/executors/jit/eltwise.h"
#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node_input.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/cpu_opset/common/op/power_static.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_cpu {

MoveFP16ConvertAfterView::MoveFP16ConvertAfterView() {
    MATCHER_SCOPE(MoveFP16ConvertAfterView);
    auto pattern = ov::pass::pattern::wrap_type<ov::op::v1::Reshape,
                                                ov::op::v0::Squeeze,
                                                ov::op::v0::Unsqueeze,
                                                ov::op::v1::Split,
                                                ov::op::v1::VariadicSplit>();
    matcher_pass_callback callback = [this](ov::pass::pattern::Matcher& matcher) {
        const auto view = matcher.get_match_root();
        const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(view->get_input_node_shared_ptr(0));
        if (!convert || transformation_callback(view) || convert->get_input_element_type(0) != ov::element::f16 ||
            convert->get_destination_type() != ov::element::f32 || convert->output(0).get_target_inputs().size() != 1 ||
            ov::is_type<ov::op::v0::Constant>(convert->get_input_node_shared_ptr(0))) {
            return false;
        }
        auto narrows_to_fp16 = [](const ov::Input<ov::Node>& input) {
            const auto* narrow = ov::as_type<const ov::op::v0::Convert>(input.get_node());
            return narrow != nullptr && narrow->get_destination_type() == ov::element::f16;
        };
        // Splitting the widening must not add materialized conversions: all but at most
        // one output must immediately return to FP16 (or have no consumers).
        size_t fp16_outputs = 0;
        for (const auto& output : view->outputs()) {
            const auto consumers = output.get_target_inputs();
            if (std::all_of(consumers.begin(), consumers.end(), narrows_to_fp16)) {
                ++fp16_outputs;
            }
        }
        if (fp16_outputs + 1 < view->get_output_size()) {
            return false;
        }
        // The roundtrip requires an FP16 JIT load/store. Reference execution can
        // change NaN payloads or signed zero, and may materialize FP32 tensors.
        const auto rank = view->get_output_partial_shape(0).rank();
        EltwiseAttrs roundtrip_attrs;
        roundtrip_attrs.data.algo = Algorithm::EltwisePowerStatic;
        roundtrip_attrs.data.alpha = 1.F;
        roundtrip_attrs.data.beta = 1.F;
        if (rank.is_dynamic() ||
            !EltwiseJitExecutor::supports(roundtrip_attrs, rank.get_length(), {ov::element::f16}, {ov::element::f16})) {
            return false;
        }
        auto inputs = view->input_values();
        inputs[0] = convert->input_value(0);
        auto new_view = view->clone_with_new_inputs(inputs);
        for (const auto& output : new_view->outputs()) {
            if (output.get_element_type() != ov::element::f16) {
                return false;
            }
        }
        auto preserve_precision = [](const std::shared_ptr<ov::Node>& from, const std::shared_ptr<ov::Node>& to) {
            const auto& info = from->get_rt_info();
            const auto it = info.find(ov::DisablePrecisionConversion::get_type_info_static());
            if (it != info.end()) {
                to->get_rt_info().insert(*it);
            }
        };
        new_view->set_friendly_name(view->get_friendly_name());
        ov::copy_runtime_info(view, new_view);
        preserve_precision(view, new_view);
        if (ov::pass::constant_folding_is_disabled(view)) {
            ov::pass::disable_constant_folding(new_view);
        }
        ov::OutputVector outputs;
        for (size_t port = 0; port < new_view->get_output_size(); ++port) {
            auto after = std::make_shared<ov::op::v0::Convert>(new_view->output(port), ov::element::f32);
            after->set_friendly_name(convert->get_friendly_name() + "/view_" + std::to_string(port));
            ov::copy_runtime_info(convert, after);
            preserve_precision(convert, after);
            if (ov::pass::constant_folding_is_disabled(convert)) {
                ov::pass::disable_constant_folding(after);
            }
            outputs.emplace_back(after);
        }
        ov::replace_node(view, outputs);
        // A fused FP16 load/store preserves the original roundtrip's NaN quieting
        // without materializing an FP32 tensor. Identity PowerStatic performs no
        // arithmetic between those conversions, preserving signed zero as well.
        for (size_t port = 0; port < outputs.size(); ++port) {
            const auto consumers = outputs[port].get_target_inputs();
            for (const auto& input : consumers) {
                if (narrows_to_fp16(input)) {
                    const auto narrow = input.get_node()->shared_from_this();
                    auto roundtrip =
                        std::make_shared<PowerStaticNode>(new_view->output(port), 1.F, 1.F, 0.F, ov::element::f16);
                    roundtrip->set_friendly_name(narrow->get_friendly_name());
                    ov::copy_runtime_info(narrow, roundtrip);
                    preserve_precision(narrow, roundtrip);
                    if (ov::pass::constant_folding_is_disabled(narrow)) {
                        ov::pass::disable_constant_folding(roundtrip);
                    }
                    ov::replace_node(narrow, roundtrip);
                }
            }
        }
        return true;
    };
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, matcher_name), callback);
}

}  // namespace ov::intel_cpu
