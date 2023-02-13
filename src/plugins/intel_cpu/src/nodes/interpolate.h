// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executors/interpolate.hpp"
#include <string>
#include <memory>
#include <vector>

#define MAX_INPUT_INTERPOLATE 8

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

struct jit_interpolate_config_params {
    InterpolateLayoutType layout;
    InterpolateMode mode;
    InferenceEngine::Precision src_prc;
    InferenceEngine::Precision dst_prc;
    int src_data_size;
    int dst_data_size;
    int indices_size;
    int spatial_dim_size;
    int C, ID, IH, IW, OD, OH, OW;
};

struct jit_interpolate_call_args {
    const void *src_ptr[MAX_INPUT_INTERPOLATE];
    const void *weight_ptr[MAX_INPUT_INTERPOLATE];
    const int *index;
    void *dst;
    size_t work_amount;
    size_t oc_off;
    //ptr to array of post op inputs pointers (flat list)
    const void* post_op_data;
};

struct jit_uni_interpolate_kernel {
    void (*ker_)(const jit_interpolate_call_args *);

    void operator()(const jit_interpolate_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_interpolate_kernel(jit_interpolate_config_params jcp, const dnnl_primitive_attr &attr) : ker_(nullptr), jcp_(jcp), attr_(attr) {}
    virtual ~jit_uni_interpolate_kernel() {}

    virtual void create_ker() = 0;

    jit_interpolate_config_params jcp_;
    const dnnl_primitive_attr &attr_;
};


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

    class InterpolateJitExecutor : public InterpolateExecutor {
        public:
            InterpolateJitExecutor(const InterpolateAttrs& interpAttrs,
                                   const VectorDims &srcDims,
                                   const VectorDims &dstDims,
                                   const std::vector<float> &dataScales,
                                   const dnnl::primitive_attr &attr);
            bool init(const InterpolateAttrs& reduceAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr &attr) override { return true; };
            void exec(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_) override;

        private:
            // nearest neighbor
            void NNPlanar(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_,
                int B, int C, int ID, int IH, int IW, int OD, int OH, int OW);
            void NNCGathered(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_,
                int B, int C, int ID, int IH, int IW, int OD, int OH, int OW);

            // onnx linear
            void linearOnnxPlanar(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_,
                int B, int C, int ID, int IH, int IW, int OD, int OH, int OW);
            void linearOnnxCGathered(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_,
                int B, int C, int ID, int IH, int IW, int OD, int OH, int OW);

            // cubic
            void cubicPlanar(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_,
                int B, int C, int IH, int IW, int OH, int OW);
            void cubicCGathered(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_,
                int B, int C, int IH, int IW, int OH, int OW);

        private:
            std::shared_ptr<jit_uni_interpolate_kernel> interpolateKernel = nullptr;
    };

    void setPostOps(dnnl::primitive_attr &attr, const VectorDims &dims);

    std::vector<float> getScales(const VectorDims &srcDimPad, const VectorDims &dstDim);

    bool hasPad = false;
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
