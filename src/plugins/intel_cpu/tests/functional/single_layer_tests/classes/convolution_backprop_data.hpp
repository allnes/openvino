// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common_test_utils/ov_tensor_utils.hpp>
#include <shared_test_classes/single_layer/convolution_backprop_data.hpp>

#include "cpu_shape.h"
#include "ngraph_functions/builders.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/convolution_params.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "gtest/gtest.h"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

using DeconvSpecParams = LayerTestsDefinitions::convBackpropDataSpecificParams;

using DeconvInputData = std::tuple<InputShape, // data shape
        ngraph::helpers::InputLayerType,       // 'output_shape' input type
        std::vector<std::vector<int32_t>>>;    // values for 'output_shape'

using DeconvLayerCPUTestParamsSet = std::tuple<DeconvSpecParams,
        DeconvInputData,
        ElementType,
        fusingSpecificParams,
        CPUSpecificParams,
        std::map<std::string, std::string>>;

class DeconvolutionLayerCPUTest : public testing::WithParamInterface<DeconvLayerCPUTestParamsSet>,
                                  virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<DeconvLayerCPUTestParamsSet> obj);
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override;
    void init_ref_function(std::shared_ptr<ov::Model> &funcRef, const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void validate() override;
    void configure_model() override;
    std::shared_ptr<ov::Model> createGraph(const std::vector<ov::PartialShape>& inShapes, ngraph::helpers::InputLayerType outShapeType);

protected:
    InferenceEngine::SizeVector kernel, stride;
    void SetUp() override;

private:
    ElementType prec;
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector dilation;
    std::vector<ptrdiff_t> padBegin, padEnd, outPadding;
    size_t convOutChannels;
    std::vector<std::vector<int32_t>> outShapeData;
    size_t inferRequestNum = 0;
};

namespace Deconvolution {

/* COMMON PARAMS */
const std::map<std::string, std::string> cpuEmptyPluginConfig;
const std::map<std::string, std::string>cpuBF16PluginConfig = { { InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16,
                                                                  InferenceEngine::PluginConfigParams::YES } };
const std::vector<std::vector<ptrdiff_t>> emptyOutputPadding = { {} };

/* ============= Deconvolution params (planar layout) ============= */
const InferenceEngine::SizeVector numOutChannels_Planar = { 6 };

/* ============= Deconvolution params (blocked layout) ============= */
const InferenceEngine::SizeVector numOutChannels_Blocked = { 64 };

/* ============= Deconvolution params (2D) ============= */
const std::vector<InferenceEngine::SizeVector> kernels2d = { {3, 3}, {1, 1} };
const std::vector<InferenceEngine::SizeVector> strides2d = { {1, 1}, {2, 2} };
const std::vector<std::vector<ptrdiff_t>> padBegins2d = { {0, 0} };
const std::vector<std::vector<ptrdiff_t>> padEnds2d = { {0, 0} };
const std::vector<InferenceEngine::SizeVector> dilations2d = { {1, 1} };


const std::vector<InferenceEngine::SizeVector> deconvAmxKernels2d = { {3, 3}, {2, 2}};
const std::vector<InferenceEngine::SizeVector> deconvAmxStrides2d = { {2, 2}};

/* ============= Deconvolution params (3D) ============= */
const std::vector<InferenceEngine::SizeVector> kernels3d = { {3, 3, 3}, {1, 1, 1} };
const std::vector<InferenceEngine::SizeVector> strides3d = { {1, 1, 1}, {2, 2, 2} };
const std::vector<std::vector<ptrdiff_t>> padBegins3d = { {0, 0, 0} };
const std::vector<std::vector<ptrdiff_t>> padEnds3d = { {0, 0, 0} };
const std::vector<InferenceEngine::SizeVector> dilations3d = { {1, 1, 1} };

const std::vector<InferenceEngine::SizeVector> deconvAmxKernels3d = { {3, 3, 3}, {2, 2, 2} };
const std::vector<InferenceEngine::SizeVector> deconvAmxStrides3d = {  {2, 2, 2} };

/* ============= */
} // namespace Deconvolution

} // namespace CPULayerTestsDefinitions