// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <tuple>
#include <vector>

#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/subtract.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov::test {

using AsymmetricConvParams = std::tuple<element::Type, int, bool, bool, size_t, element::Type>;

class AsymmetricConv : public testing::WithParamInterface<AsymmetricConvParams>,
                       virtual public SubgraphBaseStaticTest,
                       public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<AsymmetricConvParams>& obj) {
        const auto& [precision, zeroPoint, perChannel, withBias, stride, hint] = obj.param;
        std::ostringstream result;
        result << "InPRC=" << precision << "_ZeroPoint=" << zeroPoint << "_PerChannel=" << perChannel
               << "_WithBias=" << withBias << "_Stride=" << stride << "_Hint=" << hint;
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [precision, zeroPoint, perChannel, withBias, stride, hint] = GetParam();
        targetDevice = utils::DEVICE_CPU;
        configuration = {ov::hint::inference_precision(hint), ov::num_streams(1), ov::inference_num_threads(1)};
        if (hint == element::f16) {
            const auto capabilities = core->get_property(targetDevice, ov::device::capabilities);
            if (std::find(capabilities.begin(), capabilities.end(), "FP16") == capabilities.end()) {
                GTEST_SKIP() << "CPU does not support FP16 inference";
            }
        }
        selectedType = "acl_I8";
        ov::ParameterVector params{std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3, 7, 8})};
        const float low = static_cast<float>((precision == element::i8 ? -128 : 0) - zeroPoint) * 0.25f;
        const float high = low + 255.f * 0.25f;
        const auto input = utils::make_fake_quantize(params[0], element::f32, 256, {}, {low}, {high}, {low}, {high});

        const size_t kernel = stride == 1 ? 2 : 3;
        std::vector<int8_t> weightsData(4 * 3 * kernel * kernel);
        for (size_t i = 0; i < weightsData.size(); ++i) {
            weightsData[i] = static_cast<int8_t>(static_cast<int>((i * 73) % 255) - 127);
        }
        const auto weights = op::v0::Constant::create(element::i8, Shape{4, 3, kernel, kernel}, weightsData);
        const auto weightsConvert = std::make_shared<op::v0::Convert>(weights, element::f32);
        ov::disable_constant_folding(weightsConvert);
        const auto scale =
            perChannel
                ? op::v0::Constant::create(element::f32, {4, 1, 1, 1}, {1.f / 1024, 2.f / 1024, 3.f / 1024, 4.f / 1024})
                : op::v0::Constant::create(element::f32, {}, {1.f / 1024});
        const auto scaledWeights = std::make_shared<op::v1::Multiply>(weightsConvert, scale);
        const auto padding = static_cast<ptrdiff_t>(stride - 1);
        std::shared_ptr<Node> output = std::make_shared<op::v1::Convolution>(input,
                                                                             scaledWeights,
                                                                             Strides{stride, stride},
                                                                             CoordinateDiff{padding, padding},
                                                                             CoordinateDiff{padding, padding},
                                                                             Strides{1, 1});
        if (withBias) {
            output = std::make_shared<op::v1::Add>(
                output,
                op::v0::Constant::create(element::f32, {1, 4, 1, 1}, {-0.375f, 0.125f, 0.25f, 0.625f}));
        }
        const auto fq = utils::make_fake_quantize(output, element::f32, 256, {}, {-32}, {31.75f}, {-32}, {31.75f});
        function = std::make_shared<Model>(OutputVector{fq}, params, "AsymmetricConv");
        // One final FQ quantum covers f16 rounding near a quantization boundary.
        abs_threshold = hint == element::f16 ? 0.25001f : 1e-5f;
        rel_threshold = 0;
    }

    void generate_inputs(const std::vector<Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& parameter = function->get_parameters().front();
        const auto precision = std::get<0>(GetParam());
        Tensor tensor(element::f32, targetInputStaticShapes[0]);
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            const auto value = static_cast<int>((i * 73) % 256);
            tensor.data<float>()[i] =
                static_cast<float>((precision == element::u8 ? value : value - 128) - std::get<1>(GetParam())) * 0.25f;
        }
        inputs.insert({parameter, tensor});
    }
};

TEST_P(AsymmetricConv, CompareWithRefs) {
    run();
    std::stringstream stream;
    compiledModel.export_model(stream);
    auto importedModel = core->import_model(stream, targetDevice, configuration);
    auto importedRequest = importedModel.create_infer_request();
    importedRequest.set_input_tensor(inferRequest.get_input_tensor());
    importedRequest.infer();
    // Zero points and scales must survive compiled-model export/import without rounding.
    const auto expected = inferRequest.get_output_tensor();
    const auto actual = importedRequest.get_output_tensor();
    ASSERT_EQ(expected.get_size(), actual.get_size());
    EXPECT_TRUE(std::equal(expected.data<const float>(),
                           expected.data<const float>() + expected.get_size(),
                           actual.data<const float>()));
    CheckPluginRelatedResults(importedModel, "Convolution");
    CheckNumberOfNodesWithType(compiledModel, "Convolution", 1);
    CheckPluginRelatedResults(compiledModel, "Convolution");
    const auto convolutions = GetNodesWithTypes(compiledModel.get_runtime_model(), {"Convolution"});
    for (const auto& node : convolutions) {
        EXPECT_EQ(node->get_input_element_type(0), std::get<0>(GetParam()));
        EXPECT_EQ(node->get_input_element_type(1), element::i8);
        EXPECT_EQ(node->get_output_element_type(0), element::f32);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_AsymmetricConv_Unsigned_CPU,
                         AsymmetricConv,
                         testing::Combine(testing::Values(element::u8),
                                          testing::Values(0, 1, 120, 255),
                                          testing::Bool(),
                                          testing::Bool(),
                                          testing::Values(1, 2),
                                          testing::Values(element::f32, element::f16)),
                         AsymmetricConv::getTestCaseName);

}  // namespace ov::test
