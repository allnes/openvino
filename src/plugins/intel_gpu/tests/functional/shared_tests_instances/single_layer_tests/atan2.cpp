// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_constants.hpp"
#include "openvino/opsets/opset15.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {

class PortableEltwiseFamilyAtan2Test : public ov::test::SubgraphBaseTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        init_input_shapes(ov::test::static_shapes_to_test_representation({ov::Shape{1, 8, 4, 4}, ov::Shape{1, 8, 4, 4}}));

        auto y = std::make_shared<ov::opset15::Parameter>(ov::element::f32, inputDynamicShapes[0]);
        auto x = std::make_shared<ov::opset15::Parameter>(ov::element::f32, inputDynamicShapes[1]);
        auto divide = std::make_shared<ov::opset15::Divide>(y, x);
        auto atan = std::make_shared<ov::opset15::Atan>(divide);

        auto pi = ov::opset15::Constant::create(ov::element::f32, {}, {3.14159265f});
        auto neg_pi = ov::opset15::Constant::create(ov::element::f32, {}, {-3.14159265f});
        auto half_pi = ov::opset15::Constant::create(ov::element::f32, {}, {1.57079633f});
        auto neg_half_pi = ov::opset15::Constant::create(ov::element::f32, {}, {-1.57079633f});
        auto zero = ov::opset15::Constant::create(ov::element::f32, {}, {0.0f});

        auto atan_plus_pi = std::make_shared<ov::opset15::Add>(atan, pi);
        auto atan_minus_pi = std::make_shared<ov::opset15::Add>(atan, neg_pi);
        auto add_pi_condition =
            std::make_shared<ov::opset15::LogicalAnd>(std::make_shared<ov::opset15::Less>(x, zero), std::make_shared<ov::opset15::GreaterEqual>(y, zero));
        auto quadrant_select = std::make_shared<ov::opset15::Select>(add_pi_condition, atan_plus_pi, atan_minus_pi);
        auto regular_select = std::make_shared<ov::opset15::Select>(std::make_shared<ov::opset15::Greater>(x, zero), atan, quadrant_select);

        auto x_is_zero = std::make_shared<ov::opset15::Equal>(x, zero);
        auto positive_half_pi = std::make_shared<ov::opset15::LogicalAnd>(x_is_zero, std::make_shared<ov::opset15::Greater>(y, zero));
        auto negative_half_pi = std::make_shared<ov::opset15::LogicalAnd>(x_is_zero, std::make_shared<ov::opset15::Less>(y, zero));
        auto special_condition = std::make_shared<ov::opset15::LogicalOr>(positive_half_pi, negative_half_pi);
        auto special_value = std::make_shared<ov::opset15::Select>(positive_half_pi, half_pi, neg_half_pi);
        auto axis_result = std::make_shared<ov::opset15::Select>(special_condition, special_value, regular_select);
        auto both_zero = std::make_shared<ov::opset15::LogicalAnd>(x_is_zero, std::make_shared<ov::opset15::Equal>(y, zero));
        auto atan2 = std::make_shared<ov::opset15::Select>(both_zero, zero, axis_result);
        atan2->set_friendly_name("Atan2");

        function = std::make_shared<ov::Model>(ov::OutputVector{atan2}, ov::ParameterVector{y, x}, "Atan2");
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& parameters = function->get_parameters();
        ASSERT_EQ(parameters.size(), 2u);
        ASSERT_EQ(target_input_static_shapes.size(), 2u);

        for (size_t input_idx = 0; input_idx < parameters.size(); ++input_idx) {
            ov::Tensor tensor(ov::element::f32, target_input_static_shapes[input_idx]);
            auto* data = tensor.data<float>();
            for (size_t i = 0; i < tensor.get_size(); ++i) {
                data[i] = input_idx == 0 ? (static_cast<int>(i % 17) - 8) * 0.25f : (static_cast<int>(i % 11) - 5) * 0.5f;
            }
            inputs.emplace(parameters[input_idx], tensor);
        }
    }
};

TEST_F(PortableEltwiseFamilyAtan2Test, Inference) {
    run();
}

}  // namespace
