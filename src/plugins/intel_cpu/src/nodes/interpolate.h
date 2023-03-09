// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executors/interpolate.hpp"
#include "executors/interpolate_list.hpp"
#include <string>
#include <memory>
#include <vector>

#define MAX_INPUT_INTERPOLATE 8

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

class Interpolate : public Node {
public:
    static constexpr size_t DATA_ID = 0;
    static constexpr size_t TARGET_SHAPE_ID = 1;
    static constexpr size_t SCALES_ID = 2;
    static constexpr size_t AXES_ID = 3;
    static constexpr int CUBIC_GRID_LEN = 4;

public:
    Interpolate(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;
    bool canBeInPlace() const override {
        return false;
    }
    bool canFuse(const NodePtr& node) const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

    bool needShapeInfer() const override;
    bool needPrepareParams() const override;
    void prepareParams() override;

private:
    InterpolateAttrs interpAttrs;
    std::shared_ptr<InterpolateExecutor> execPtr = nullptr;

    void setPostOps(dnnl::primitive_attr &attr, const VectorDims &dims);

    std::vector<float> getScales(const VectorDims &srcDimPad, const VectorDims &dstDim);
    InterpolateShapeCalcMode shapeCalcMode;

    bool isAxesSpecified = false;
    std::vector<int> axes;
    std::vector<float> scales;
    bool isScaleConstant = false;

    // 6 ptrs for each quantization, 2 ptrs for each depth_wise
    std::vector<const void*> postOpsDataPtrs;

    std::vector<float> lastScales;
    std::vector<int32_t> lastSizes;

    VectorDims lastOutputDims;

    std::string errorPrefix;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
