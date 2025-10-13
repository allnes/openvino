// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "deconv_3d_force_fp32.hpp"

#include <memory>

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/depth_to_space.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_cpu {

Deconv3DForceFP32::Deconv3DForceFP32() {
    using namespace ov::pass::pattern;

    auto input = any_input(rank_equals(5));
    auto weights = any_input(rank_equals(5));
    auto output_shape = any_input();
    auto deconv2 = wrap_type<ov::op::v1::ConvolutionBackpropData>({input, weights});
    auto deconv3 = wrap_type<ov::op::v1::ConvolutionBackpropData>({input, weights, output_shape});
    auto deconv_any = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{deconv2, deconv3});

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();
        auto deconv = ov::as_type_ptr<ov::op::v1::ConvolutionBackpropData>(node);
        if (!deconv)
            return false;

        // Only rank-5 (3D) and f16 tensors
        if (deconv->get_input_partial_shape(0).rank().get_length() != 5)
            return false;
        if (deconv->get_output_element_type(0) != ov::element::f16)
            return false;

        auto data = deconv->input_value(0);
        auto wei = deconv->input_value(1);
        auto out_et = deconv->get_output_element_type(0);

        // Convert inputs to f32
        auto data_f32 = std::make_shared<ov::op::v0::Convert>(data, ov::element::f32);
        auto wei_f32 = std::make_shared<ov::op::v0::Convert>(wei, ov::element::f32);

        std::shared_ptr<ov::Node> new_deconv;
        if (deconv->get_input_size() == 3) {
            new_deconv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(data_f32,
                                                                               wei_f32,
                                                                               deconv->input_value(2),
                                                                               deconv->get_strides(),
                                                                               deconv->get_pads_begin(),
                                                                               deconv->get_pads_end(),
                                                                               deconv->get_dilations(),
                                                                               deconv->get_auto_pad(),
                                                                               deconv->get_output_padding());
        } else {
            new_deconv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(data_f32,
                                                                               wei_f32,
                                                                               deconv->get_strides(),
                                                                               deconv->get_pads_begin(),
                                                                               deconv->get_pads_end(),
                                                                               deconv->get_dilations(),
                                                                               deconv->get_auto_pad(),
                                                                               deconv->get_output_padding());
        }

        // Convert back to f16 for graph consistency
        auto back_to_f16 = std::make_shared<ov::op::v0::Convert>(new_deconv, out_et);

        back_to_f16->set_friendly_name(deconv->get_friendly_name());
        ov::copy_runtime_info(deconv, {data_f32, wei_f32, new_deconv, back_to_f16});
        deconv->output(0).replace(back_to_f16->output(0));
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(deconv_any, "Deconv3DForceFP32");
    register_matcher(m, callback);
}

}  // namespace ov::intel_cpu
