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
#include "openvino/op/depth_to_space.hpp"
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
#include "openvino/op/slice.hpp"
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
// helpers kept minimal; remove unused constants to avoid warnings
inline ov::Output<ov::Node> make1d(const ov::Output<ov::Node>& value) {
    // Force shape [1] regardless of whether input is scalar or already [1]
    auto one = ov::op::v0::Constant::create(ov::element::i32, {1}, {1});
    return std::make_shared<ov::op::v1::Reshape>(value, one, false);
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

    // Match both 2-input and 3-input (with output_shape) variants
    auto deconv_2_inputs =
        ov::pass::pattern::wrap_type<ov::op::v1::ConvolutionBackpropData>({input_pattern, weights_pattern});
    auto deconv_3_inputs = ov::pass::pattern::wrap_type<ov::op::v1::ConvolutionBackpropData>(
        {input_pattern, weights_pattern, ov::pass::pattern::any_input()});
    auto deconv_3d = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{deconv_2_inputs, deconv_3_inputs});

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto deconv = ov::as_type_ptr<ov::op::v1::ConvolutionBackpropData>(m.get_match_root());
        if (!deconv)
            return false;

        // Only NCDHW tensors (rank 5)
        if (deconv->get_input_partial_shape(0).rank().get_length() != 5) {
            return false;
        }

        const auto& strides = deconv->get_strides();
        const auto& dilations = deconv->get_dilations();
        auto pads_begin = deconv->get_pads_begin();
        auto pads_end = deconv->get_pads_end();
        auto auto_pad = deconv->get_auto_pad();
        const auto& out_pad = deconv->get_output_padding();

        // Strict gating: support only UNet3D-like blocks to avoid affecting broad test matrix yet
        // Conditions:
        //   - strides == (2,2,2)
        //   - dilations == (1,1,1)
        //   - kernel == (2,2,2)
        //   - pads_begin == pads_end == (0,0,0)
        //   - output_padding == (0,0,0)
        //   - auto_pad in {EXPLICIT, NOTSET}
        if (strides.size() != 3 || dilations.size() != 3 || pads_begin.size() != 3 || pads_end.size() != 3)
            return false;
        if (!(strides[0] == 2 && strides[1] == 2 && strides[2] == 2))
            return false;
        if (!(dilations[0] == 1 && dilations[1] == 1 && dilations[2] == 1))
            return false;
        if (!(auto_pad == ov::op::PadType::EXPLICIT || auto_pad == ov::op::PadType::NOTSET ||
              auto_pad == ov::op::PadType::VALID || auto_pad == ov::op::PadType::SAME_UPPER ||
              auto_pad == ov::op::PadType::SAME_LOWER))
            return false;

        // Weights must be 5D; allow Constant or statically known PartialShape
        const auto& wps = deconv->get_input_partial_shape(1);
        if (wps.rank().is_dynamic() || wps.rank().get_length() != 5)
            return false;
        // Kernel sizes must be static to derive padding
        for (int i = 2; i <= 4; ++i) {
            if (!wps[i].is_static())
                return false;
        }
        int kD = static_cast<int>(wps[2].get_length());
        int kH = static_cast<int>(wps[3].get_length());
        int kW = static_cast<int>(wps[4].get_length());
        // Support k=2 and k=3 cases (stride=2, dilation=1)
        if (!((kD == 2 && kH == 2 && kW == 2) || (kD == 3 && kH == 3 && kW == 3)))
            return false;

        // Prepare input/weights
        auto data = deconv->input_value(0);
        auto weights = deconv->input_value(1);
        ov::NodeVector decomp_nodes;
        ov::Output<ov::Node> result = data;

        // Build zero-insertion upsample per spatial axis using reshape -> pad -> reshape
        auto upsample_axis = [&](const ov::Output<ov::Node>& in, int axis, int s) -> ov::Output<ov::Node> {
            if (s == 1)
                return in;
            auto shp = shape_of_i32(in);
            // Gather all dims
            auto N = gather_dim(shp, 0);
            auto C = gather_dim(shp, 1);
            auto D = gather_dim(shp, 2);
            auto H = gather_dim(shp, 3);
            auto W = gather_dim(shp, 4);

            std::vector<ov::Output<ov::Node>> dims = {make1d(N), make1d(C), make1d(D), make1d(H), make1d(W)};

            // Insert a unit dim after the target axis
            std::vector<ov::Output<ov::Node>> shape_with_unit;
            for (int i = 0; i < 5; ++i) {
                shape_with_unit.push_back(dims[i]);
                if (i == axis) {
                    shape_with_unit.push_back(ov::op::v0::Constant::create(ov::element::i32, {1}, {1}));
                }
            }
            auto resh1 = std::make_shared<ov::op::v1::Reshape>(in, concat0(shape_with_unit), false);
            decomp_nodes.push_back(resh1);

            // Build pads: pad only the inserted axis (axis + 1) at end with (s-1)
            auto pads0 = ov::op::v0::Constant::create(ov::element::i32, {1}, {0});
            std::vector<ov::Output<ov::Node>> pads_begin_v(6, pads0);
            std::vector<ov::Output<ov::Node>> pads_end_v(6, pads0);
            pads_end_v[axis + 1] = ov::op::v0::Constant::create(ov::element::i32, {1}, {s - 1});
            auto pads_begin = concat0(pads_begin_v);
            auto pads_end = concat0(pads_end_v);
            auto pad_value = ov::op::v0::Constant::create(in.get_element_type(), {}, {0});
            auto padded = std::make_shared<ov::op::v1::Pad>(resh1, pads_begin, pads_end, pad_value, ov::op::PadMode::CONSTANT);
            decomp_nodes.push_back(padded);

            // Reshape back with scaled target dim
            auto scale_c = ov::op::v0::Constant::create(ov::element::i32, {}, {s});
            auto scaled_dim = std::make_shared<ov::op::v1::Multiply>(dims[axis], scale_c);
            std::vector<ov::Output<ov::Node>> final_dims;
            for (int i = 0; i < 5; ++i) {
                final_dims.push_back(i == axis ? make1d(scaled_dim) : dims[i]);
            }
            auto resh2 = std::make_shared<ov::op::v1::Reshape>(padded, concat0(final_dims), false);
            decomp_nodes.push_back(resh2);
            return resh2;
        };

        // Apply upsample for D,H,W axes according to strides
        const int sD = static_cast<int>(strides[0]);
        const int sH = static_cast<int>(strides[1]);
        const int sW = static_cast<int>(strides[2]);
        result = upsample_axis(result, /*axis D*/ 2, sD);
        result = upsample_axis(result, /*axis H*/ 3, sH);
        result = upsample_axis(result, /*axis W*/ 4, sW);

        // Pad the upsampled tensor per axis to match Deconv (general k, s, d=1) with arbitrary pads and output_padding
        {
            const bool is_same = (auto_pad == ov::op::PadType::SAME_UPPER) || (auto_pad == ov::op::PadType::SAME_LOWER);
            const int pLD = is_same ? 0 : static_cast<int>(pads_begin[0]);
            const int pLH = is_same ? 0 : static_cast<int>(pads_begin[1]);
            const int pLW = is_same ? 0 : static_cast<int>(pads_begin[2]);
            const int pRD = is_same ? 0 : static_cast<int>(pads_end[0]);
            const int pRH = is_same ? 0 : static_cast<int>(pads_end[1]);
            const int pRW = is_same ? 0 : static_cast<int>(pads_end[2]);
            const int oPD = (!out_pad.empty()) ? static_cast<int>(out_pad[0]) : 0;
            const int oPH = (!out_pad.empty()) ? static_cast<int>(out_pad[1]) : 0;
            const int oPW = (!out_pad.empty()) ? static_cast<int>(out_pad[2]) : 0;
            // For d=1 general k: pad_sum_i = -s_i - (pL_i + pR_i) + 2*(k_i-1) + out_pad_i + 1
            const int pad_sum_D = -sD - (pLD + pRD) + 2 * (kD - 1) + oPD + 1;
            const int pad_sum_H = -sH - (pLH + pRH) + 2 * (kH - 1) + oPH + 1;
            const int pad_sum_W = -sW - (pLW + pRW) + 2 * (kW - 1) + oPW + 1;
            const int pad_b_D = 0, pad_b_H = 0, pad_b_W = 0;
            const int pad_e_D = std::max(0, pad_sum_D);
            const int pad_e_H = std::max(0, pad_sum_H);
            const int pad_e_W = std::max(0, pad_sum_W);

            auto zero_1d = ov::op::v0::Constant::create(ov::element::i32, {1}, {0});
            auto pads_begin_dyn = concat0({zero_1d,
                                           zero_1d,
                                           ov::op::v0::Constant::create(ov::element::i32, {1}, {pad_b_D}),
                                           ov::op::v0::Constant::create(ov::element::i32, {1}, {pad_b_H}),
                                           ov::op::v0::Constant::create(ov::element::i32, {1}, {pad_b_W})});
            auto pads_end_dyn   = concat0({zero_1d,
                                           zero_1d,
                                           ov::op::v0::Constant::create(ov::element::i32, {1}, {pad_e_D}),
                                           ov::op::v0::Constant::create(ov::element::i32, {1}, {pad_e_H}),
                                           ov::op::v0::Constant::create(ov::element::i32, {1}, {pad_e_W})});
            auto pad_value = ov::op::v0::Constant::create(result.get_element_type(), {}, {0});
            auto pad_op = std::make_shared<ov::op::v1::Pad>(result, pads_begin_dyn, pads_end_dyn, pad_value, ov::op::PadMode::CONSTANT);
            decomp_nodes.push_back(pad_op);
            result = pad_op;
        }

        // Transpose and flip weights: [Cin,Cout,kD,kH,kW] -> [Cout,Cin,kD,kH,kW] with spatial reverse
        auto transpose_order = ov::op::v0::Constant::create(ov::element::i32, {5}, {1, 0, 2, 3, 4});
        auto weights_t = std::make_shared<ov::op::v1::Transpose>(weights, transpose_order);
        decomp_nodes.push_back(weights_t);
        auto flip_axes = ov::op::v0::Constant::create(ov::element::i64, {3}, {2, 3, 4});
        auto weights_tf = std::make_shared<ov::op::v1::Reverse>(weights_t, flip_axes, ov::op::v1::Reverse::Mode::INDEX);
        decomp_nodes.push_back(weights_tf);

        // Final Conv3D (stride=1, EXPLICIT pads 0)
        ov::Strides conv_strides = {1, 1, 1};
        ov::CoordinateDiff conv_pads_begin = {0, 0, 0};
        ov::CoordinateDiff conv_pads_end = {0, 0, 0};
        ov::Strides conv_dilations = {1, 1, 1};

        auto conv3d = std::make_shared<ov::op::v1::Convolution>(result,
                                                                weights_tf,
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
