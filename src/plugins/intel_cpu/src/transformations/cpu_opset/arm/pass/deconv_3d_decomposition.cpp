// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deconv_3d_decomposition.hpp"

#include <array>
#include <memory>
#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/reverse.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_cpu {

namespace {
inline std::shared_ptr<ov::op::v0::Constant> c0_i32() {
    return ov::op::v0::Constant::create(ov::element::i32, {}, {0});
}
inline std::shared_ptr<ov::op::v0::Constant> c1_i32() {
    return ov::op::v0::Constant::create(ov::element::i32, {}, {1});
}
inline std::shared_ptr<ov::op::v0::Constant> c2_i32() {
    return ov::op::v0::Constant::create(ov::element::i32, {}, {2});
}
inline std::shared_ptr<ov::op::v0::Constant> c4_i32() {
    return ov::op::v0::Constant::create(ov::element::i32, {}, {4});
}
inline std::shared_ptr<ov::op::v0::Constant> c5_i32() {
    return ov::op::v0::Constant::create(ov::element::i32, {}, {5});
}
inline std::shared_ptr<ov::op::v0::Constant> c_1_i32() {
    return ov::op::v0::Constant::create(ov::element::i32, {}, {-1});
}
inline ov::Output<ov::Node> make1d(const ov::Output<ov::Node>& scalar) {
    return std::make_shared<ov::op::v0::Unsqueeze>(scalar, c0_i32());
}
inline ov::Output<ov::Node> concat0(const std::vector<ov::Output<ov::Node>>& dims) {
    if (dims.size() == 1)
        return dims[0];
    return std::make_shared<ov::op::v0::Concat>(dims, 0);
}
inline ov::Output<ov::Node> shape_of_i32(const ov::Output<ov::Node>& node) {
    return std::make_shared<ov::op::v3::ShapeOf>(node, ov::element::i32);
}
inline ov::Output<ov::Node> gather_dim(const ov::Output<ov::Node>& shape, int idx) {
    auto idx_c = ov::op::v0::Constant::create(ov::element::i32, {}, {idx});
    return std::make_shared<ov::op::v8::Gather>(shape, idx_c, c0_i32());
}
}  // namespace

Deconv3DDecomposition::Deconv3DDecomposition() {
    auto input_pattern = ov::pass::pattern::any_input(ov::pass::pattern::rank_equals(5));
    auto weights_pattern = ov::pass::pattern::any_input(ov::pass::pattern::rank_equals(5));

    auto deconv_3d =
        ov::pass::pattern::wrap_type<ov::op::v1::ConvolutionBackpropData>({input_pattern, weights_pattern});

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto deconv = ov::as_type_ptr<ov::op::v1::ConvolutionBackpropData>(m.get_match_root());
        if (!deconv)
            return false;

        // Temporary safety: disable transformation until zero-insertion path is fully validated for 3D
        // (Conv3D ACL is already enabled by default and improves latency; enabling Deconv3D decomposition will follow.)
        return false;

        // Only NCDHW tensors (rank 5)
        if (deconv->get_input_partial_shape(0).rank().get_length() != 5) {
            return false;
        }

        // Strides
        const auto& strides = deconv->get_strides();
        if (strides.size() != 3)
            return false;

        // output_padding is supported for EXPLICIT/NOTSET; for SAME_* + non-zero output_padding skip below

        auto auto_pad = deconv->get_auto_pad();
        auto pads_begin = deconv->get_pads_begin();
        auto pads_end = deconv->get_pads_end();
        const auto& dilations = deconv->get_dilations();

        // Prepare input
        auto data = deconv->input_value(0);
        auto weights = deconv->input_value(1);
        ov::NodeVector decomp_nodes;

        ov::Output<ov::Node> result = data;

        // Compute padding per axis
        std::array<ov::Output<ov::Node>, 3> pad_l_dyn;
        std::array<ov::Output<ov::Node>, 3> pad_r_dyn;

        auto two = c2_i32();
        auto zero_scalar = c0_i32();

        // Effective kernel: insert zeros in weights by stride factors to avoid data upsampling.
        // Guards: support only dilations == 1 for now; skip stride>1 for stability until fully verified.
        if (dilations[0] != 1 || dilations[1] != 1 || dilations[2] != 1)
            return false;

        // Derive K_minus_1 from weights shape
        auto weights_shape = shape_of_i32(weights);
        auto Kd = gather_dim(weights_shape, 2);
        auto Kh = gather_dim(weights_shape, 3);
        auto Kw = gather_dim(weights_shape, 4);
        auto minus_one = c_1_i32();
        auto Kd_minus_1 = std::make_shared<ov::op::v1::Add>(Kd, minus_one);
        auto Kh_minus_1 = std::make_shared<ov::op::v1::Add>(Kh, minus_one);
        auto Kw_minus_1 = std::make_shared<ov::op::v1::Add>(Kw, minus_one);

        auto out_padding = deconv->get_output_padding();
        int opad_d = (out_padding.size() > 0) ? static_cast<int>(out_padding[0]) : 0;
        int opad_h = (out_padding.size() > 1) ? static_cast<int>(out_padding[1]) : 0;
        int opad_w = (out_padding.size() > 2) ? static_cast<int>(out_padding[2]) : 0;

        // Determine pad_left/right for EXPLICIT/NOTSET/SAME_*
        auto make_pad_pair = [&](const ov::Output<ov::Node>& K_minus_1,
                                 int32_t p_begin,
                                 int32_t p_end,
                                 int out_pad_axis,
                                 bool same_upper,
                                 bool same_lower) {
            ov::Output<ov::Node> pl, pr;
            if (same_upper || same_lower) {
                // SAME split
                auto two = c2_i32();
                if (same_upper) {
                    // pad_left = floor((K-1)/2), pad_right = K-1 - pad_left
                    pl = std::make_shared<ov::op::v1::Divide>(K_minus_1, two, true);
                    pr = std::make_shared<ov::op::v1::Subtract>(K_minus_1, pl);
                } else {
                    // SAME_LOWER: pad_left = ceil((K-1)/2) = floor(K/2)
                    auto K = std::make_shared<ov::op::v1::Add>(K_minus_1, c1_i32());
                    pl = std::make_shared<ov::op::v1::Divide>(K, two, true);
                    pr = std::make_shared<ov::op::v1::Subtract>(K_minus_1, pl);
                }
            } else {
                // EXPLICIT/NOTSET
                auto pb = ov::op::v0::Constant::create(ov::element::i32, {}, {p_begin});
                auto pe = ov::op::v0::Constant::create(ov::element::i32, {}, {p_end});
                pl = std::make_shared<ov::op::v1::Subtract>(K_minus_1, pb);
                auto tmp = std::make_shared<ov::op::v1::Subtract>(K_minus_1, pe);
                auto opd = ov::op::v0::Constant::create(ov::element::i32, {}, {out_pad_axis});
                pr = std::make_shared<ov::op::v1::Add>(tmp, opd);
            }
            // non-negative
            pl = std::make_shared<ov::op::v1::Maximum>(pl, zero_scalar);
            pr = std::make_shared<ov::op::v1::Maximum>(pr, zero_scalar);
            return std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>(pl, pr);
        };

        bool same_upper = (auto_pad == ov::op::PadType::SAME_UPPER);
        bool same_lower = (auto_pad == ov::op::PadType::SAME_LOWER);
        if (same_upper || same_lower) {
            // For SAME_* use symmetric base (effectively pb=pe=0 in formulas)
            pads_begin = {0, 0, 0};
            pads_end = {0, 0, 0};
        }

        auto pb0 = static_cast<int32_t>(pads_begin.size() > 0 ? pads_begin[0] : 0);
        auto pb1 = static_cast<int32_t>(pads_begin.size() > 1 ? pads_begin[1] : 0);
        auto pb2 = static_cast<int32_t>(pads_begin.size() > 2 ? pads_begin[2] : 0);
        auto pe0 = static_cast<int32_t>(pads_end.size() > 0 ? pads_end[0] : 0);
        auto pe1 = static_cast<int32_t>(pads_end.size() > 1 ? pads_end[1] : 0);
        auto pe2 = static_cast<int32_t>(pads_end.size() > 2 ? pads_end[2] : 0);

        auto pr_d = make_pad_pair(Kd_minus_1, pb0, pe0, opad_d, same_upper, same_lower);
        auto pr_h = make_pad_pair(Kh_minus_1, pb1, pe1, opad_h, same_upper, same_lower);
        auto pr_w = make_pad_pair(Kw_minus_1, pb2, pe2, opad_w, same_upper, same_lower);
        pad_l_dyn = {pr_d.first, pr_h.first, pr_w.first};
        pad_r_dyn = {pr_d.second, pr_h.second, pr_w.second};

        // Apply padding on D,H,W
        auto zero_1d = ov::op::v0::Constant::create(ov::element::i32, {1}, {0});
        auto pad_d = make1d(pad_l_dyn[0]);
        auto pad_h = make1d(pad_l_dyn[1]);
        auto pad_w = make1d(pad_l_dyn[2]);
        auto pad_d_r = make1d(pad_r_dyn[0]);
        auto pad_h_r = make1d(pad_r_dyn[1]);
        auto pad_w_r = make1d(pad_r_dyn[2]);
        auto pads_begin_dyn = concat0({zero_1d, zero_1d, pad_d, pad_h, pad_w});
        auto pads_end_dyn = concat0({zero_1d, zero_1d, pad_d_r, pad_h_r, pad_w_r});
        auto pad_value = ov::op::v0::Constant::create(result.get_element_type(), {}, {0});
        auto pad_op = std::make_shared<ov::op::v1::Pad>(result,
                                                        pads_begin_dyn,
                                                        pads_end_dyn,
                                                        pad_value,
                                                        ov::op::PadMode::CONSTANT);
        decomp_nodes.push_back(pad_op);
        result = pad_op;

        // Prepare weights: transpose channels (use cross-correlation semantics)
        auto transpose_order = ov::op::v0::Constant::create(ov::element::i32, {5}, {1, 0, 2, 3, 4});
        auto weights_t = std::make_shared<ov::op::v1::Transpose>(weights, transpose_order);
        decomp_nodes.push_back(weights_t);

        ov::Strides conv_strides = {1, 1, 1};
        ov::CoordinateDiff conv_pads_begin = {0, 0, 0};
        ov::CoordinateDiff conv_pads_end = {0, 0, 0};
        ov::Strides conv_dilations = {1, 1, 1};

        auto conv3d = std::make_shared<ov::op::v1::Convolution>(result,
                                                                weights_t,
                                                                conv_strides,
                                                                conv_pads_begin,
                                                                conv_pads_end,
                                                                conv_dilations,
                                                                ov::op::PadType::EXPLICIT);
        decomp_nodes.push_back(conv3d);
        result = conv3d;

        result.get_node()->set_friendly_name(deconv->get_friendly_name());
        ov::copy_runtime_info(deconv, decomp_nodes);
        deconv->output(0).replace(result);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(deconv_3d, "Deconv3DDecomposition");
    register_matcher(m, callback);
}

}  // namespace ov::intel_cpu
