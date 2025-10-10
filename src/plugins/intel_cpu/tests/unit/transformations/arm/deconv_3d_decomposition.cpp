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

TEST(Deconv3DDecomposition, ExplicitPads_Stride1_StaticShapes) {
    ::setenv("OV_CPU_ENABLE_DECONV3D_DECOMPOSITION", "1", 1);
    // Build model: Deconv3D with stride=1, explicit pads
    ov::Shape input_shape{1, 8, 8, 8, 8};
    ov::Shape weights_shape{8, 16, 3, 3, 3};  // [Cin, Cout, kD,kH,kW]

    auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, input_shape);
    auto weights = ov::opset1::Constant::create(ov::element::f32, weights_shape, {1});

    ov::Strides strides{1, 1, 1};
    ov::CoordinateDiff pads_begin{1, 1, 1};
    ov::CoordinateDiff pads_end{1, 1, 1};
    ov::Strides dilations{1, 1, 1};

    auto deconv = std::make_shared<ov::opset1::ConvolutionBackpropData>(param, weights, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);
    auto res = std::make_shared<ov::opset1::Result>(deconv);
    auto f = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "Deconv3D");

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<Deconv3DDecomposition>();
    manager.run_passes(f);

    // Expect that the root op is now Convolution (3D)
    auto root = f->get_results().front()->input_value(0).get_node_shared_ptr();
    auto conv = std::dynamic_pointer_cast<ov::opset1::Convolution>(root);
    ASSERT_NE(conv, nullptr);
}

TEST(Deconv3DDecomposition, NotSetPads_Stride2_StaticShapes) {
    ::setenv("OV_CPU_ENABLE_DECONV3D_DECOMPOSITION", "1", 1);
    GTEST_SKIP() << "Stride>1 cases are currently disabled in the pass to avoid latency regressions with Interpolate-based upsampling.";
}

TEST(Deconv3DDecomposition, SameUpper_Stride1_WithOutputPadding) {
    ::setenv("OV_CPU_ENABLE_DECONV3D_DECOMPOSITION", "1", 1);
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
    ASSERT_NE(conv, nullptr);
}
