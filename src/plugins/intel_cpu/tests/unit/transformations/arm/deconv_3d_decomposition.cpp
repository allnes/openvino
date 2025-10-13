// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/arm/pass/deconv_3d_decomposition.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace ov::intel_cpu;

TEST(Deconv3DDecomposition, Stride2_Kernel2_NoPads_TransformsAndShape) {
    // Build model: Deconv3D with stride=2, kernel=2, explicit pads=0
    ov::Shape input_shape{1, 4, 8, 10, 12};   // N,C,D,H,W
    ov::Shape weights_shape{4, 8, 2, 2, 2};   // [Cin, Cout, kD,kH,kW]

    auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, input_shape);
    auto weights = ov::opset1::Constant::create(ov::element::f32, weights_shape, {1});

    ov::Strides strides{2, 2, 2};
    ov::CoordinateDiff pads_begin{0, 0, 0};
    ov::CoordinateDiff pads_end{0, 0, 0};
    ov::Strides dilations{1, 1, 1};

    auto deconv = std::make_shared<ov::opset1::ConvolutionBackpropData>(param,
        weights, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);
    auto res = std::make_shared<ov::opset1::Result>(deconv);
    auto f = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "Deconv3D_S2K2");

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<Deconv3DDecomposition>();
    manager.run_passes(f);

    // Expect that the root op is now Convolution (3D)
    auto root = f->get_results().front()->input_value(0).get_node_shared_ptr();
    auto conv = std::dynamic_pointer_cast<ov::opset1::Convolution>(root);
    ASSERT_NE(conv, nullptr);

    // And output shape equals doubled spatial dims
    auto out_pshape = conv->get_output_partial_shape(0);
    ASSERT_TRUE(out_pshape.rank().is_static());
    auto out_shape = out_pshape.get_shape();
    EXPECT_EQ(out_shape[2], input_shape[2] * 2);
    EXPECT_EQ(out_shape[3], input_shape[3] * 2);
    EXPECT_EQ(out_shape[4], input_shape[4] * 2);
}

TEST(Deconv3DDecomposition, ExplicitPads_Stride1_NoTransform) {
    // Build model: Deconv3D with stride=1, explicit pads => pass must not trigger
    ov::Shape input_shape{1, 8, 8, 8, 8};
    ov::Shape weights_shape{8, 16, 3, 3, 3};
    auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, input_shape);
    auto weights = ov::opset1::Constant::create(ov::element::f32, weights_shape, {1});
    ov::Strides strides{1, 1, 1};
    ov::CoordinateDiff pads_begin{1, 1, 1};
    ov::CoordinateDiff pads_end{1, 1, 1};
    ov::Strides dilations{1, 1, 1};
    auto deconv = std::make_shared<ov::opset1::ConvolutionBackpropData>(param, weights, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);
    auto res = std::make_shared<ov::opset1::Result>(deconv);
    auto f = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "Deconv3D_noXform");

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<Deconv3DDecomposition>();
    manager.run_passes(f);

    auto root = f->get_results().front()->input_value(0).get_node_shared_ptr();
    auto conv = std::dynamic_pointer_cast<ov::opset1::Convolution>(root);
    ASSERT_EQ(conv, nullptr);
}

TEST(Deconv3DDecomposition, SameUpper_Stride1_WithOutputPadding_NoTransform) {
    ov::Shape input_shape{1, 2, 5, 5, 5};
    ov::Shape weights_shape{2, 2, 3, 3, 3};
    auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, input_shape);
    auto weights = ov::opset1::Constant::create(ov::element::f32, weights_shape, {1});
    ov::Strides strides{1, 1, 1};
    ov::CoordinateDiff pads_begin{0, 0, 0};
    ov::CoordinateDiff pads_end{0, 0, 0};
    ov::Strides dilations{1, 1, 1};
    ov::CoordinateDiff out_pad{1, 0, 1};
    auto deconv = std::make_shared<ov::opset1::ConvolutionBackpropData>(param,
        weights, strides, pads_begin, pads_end, dilations, ov::op::PadType::SAME_UPPER, out_pad);
    auto res = std::make_shared<ov::opset1::Result>(deconv);
    auto f = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "Deconv3D_SAME");
    ov::pass::Manager manager; manager.register_pass<ov::pass::InitNodeInfo>(); manager.register_pass<Deconv3DDecomposition>(); manager.run_passes(f);
    auto root = f->get_results().front()->input_value(0).get_node_shared_ptr();
    auto conv = std::dynamic_pointer_cast<ov::opset1::Convolution>(root);
    ASSERT_EQ(conv, nullptr);
}
