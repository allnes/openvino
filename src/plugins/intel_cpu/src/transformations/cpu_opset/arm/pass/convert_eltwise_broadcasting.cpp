// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "convert_eltwise_broadcasting.hpp"

#include <numeric>

#include <ngraph/rt_info.hpp>
#include <ngraph/opsets/opset1.hpp>
#include "nodes/executors/acl/acl_utils.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"


NGRAPH_RTTI_DEFINITION(ov::intel_cpu::ConvertEltwiseBase, "ConvertEltwiseBase", 0);
template <class T>
ngraph::matcher_pass_callback ov::intel_cpu::ConvertEltwiseBase::convert_eltwise() {
    return [&](ngraph::pattern::Matcher& m) {
        auto eltwise = m.get_match_root();
        if (!std::dynamic_pointer_cast<T>(eltwise)) {
            return false;
        }

        if (eltwise->get_input_shape(0).size() == eltwise->get_input_shape(1).size() &&
            eltwise->get_input_shape(0) != eltwise->get_input_shape(1) &&
            eltwise->get_input_shape(0).size() < 5) {
            auto counter_elem = [](const ov::Shape& _shape) -> size_t {
                size_t count_elems = 1;
                for (auto elem_dims : _shape) { count_elems *= elem_dims; }
                return count_elems;
            };

            int originId = counter_elem(eltwise->get_input_shape(0)) > counter_elem(eltwise->get_input_shape(1)) ? 0 : 1;
            int broadcastedId = counter_elem(eltwise->get_input_shape(0)) > counter_elem(eltwise->get_input_shape(1)) ? 1 : 0;
            auto&& broadcastedInput = eltwise->input_value(broadcastedId);
            auto shape_node = std::make_shared<ov::op::v0::Constant>(ngraph::element::i64,
                                                                     ngraph::Shape{eltwise->get_input_shape(originId).size()},
                                                                     eltwise->get_input_shape(originId).data());
            auto reshape = std::make_shared<ov::op::v3::Broadcast>(broadcastedInput, shape_node);
            eltwise->set_argument(broadcastedId, reshape);
            return true;
        }
        return false;
    };
}

NGRAPH_RTTI_DEFINITION(ov::intel_cpu::ConvertAdd, "ConvertAdd", 0);
ov::intel_cpu::ConvertAdd::ConvertAdd() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<ov::op::v1::Add>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                    ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                   ngraph::pattern::has_static_shape()), "ConvertAdd");
    register_matcher(m, convert_eltwise<ov::op::v1::Add>());
}

NGRAPH_RTTI_DEFINITION(ov::intel_cpu::ConvertSubtract, "ConvertSubtract", 0);
ov::intel_cpu::ConvertSubtract::ConvertSubtract() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<ov::op::v1::Subtract>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                         ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                        ngraph::pattern::has_static_shape()), "ConvertSubtract");
    register_matcher(m, convert_eltwise<ov::op::v1::Subtract>());
}

NGRAPH_RTTI_DEFINITION(ov::intel_cpu::ConvertMultiply, "ConvertMultiply", 0);
ov::intel_cpu::ConvertMultiply::ConvertMultiply() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<ov::op::v1::Multiply>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                         ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                        ngraph::pattern::has_static_shape()), "ConvertMultiply");
    register_matcher(m, convert_eltwise<ov::op::v1::Multiply>());
}

NGRAPH_RTTI_DEFINITION(ov::intel_cpu::ConvertMinimum, "ConvertMinimum", 0);
ov::intel_cpu::ConvertMinimum::ConvertMinimum() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<ov::op::v1::Minimum>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                        ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                       ngraph::pattern::has_static_shape()), "ConvertMinimum");
    register_matcher(m, convert_eltwise<ov::op::v1::Minimum>());
}

NGRAPH_RTTI_DEFINITION(ov::intel_cpu::ConvertMaximum, "ConvertMaximum", 0);
ov::intel_cpu::ConvertMaximum::ConvertMaximum() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<ov::op::v1::Maximum>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                        ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                       ngraph::pattern::has_static_shape()), "ConvertMaximum");
    register_matcher(m, convert_eltwise<ov::op::v1::Maximum>());
}