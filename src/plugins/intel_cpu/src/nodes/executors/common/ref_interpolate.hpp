// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/interpolate_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "post_ops.hpp"

namespace ov {
namespace intel_cpu {

using PostOpsPtr = std::shared_ptr<PostOps>;

class RefInterpolateExecutor : public Executor {
public:
    RefInterpolateExecutor(const InterpolateAttrs& attrs,
                          const PostOpsPtr& postOps,
                          const MemoryArgs& memory,
                          const ExecutorContext::CPtr context);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    
    impl_desc_type implType() const override {
        return impl_desc_type::ref;
    }

private:
    InterpolateAttrs m_attrs;
    PostOpsPtr m_postOps;
    
    // Interpolation helper methods
    void buildTblNN(const VectorDims& srcDimPad5d, const VectorDims& dstDim5d, 
                    const std::vector<float>& dataScales);
    void buildTblLinearOnnx(const VectorDims& srcDimPad5d, const VectorDims& dstDim5d, 
                           const std::vector<float>& dataScales);
    void buildTblCubic(const VectorDims& srcDimPad5d, const VectorDims& dstDim5d,
                      const std::vector<float>& dataScales);
    void buildTblPillow(const VectorDims& srcDimPad5d, const VectorDims& dstDim5d,
                       const std::vector<float>& dataScales);
    
    void NNRef(const uint8_t* in_ptr_, uint8_t* out_ptr_, 
               int B, int C, int ID, int IH, int IW, int OD, int OH, int OW);
    void linearOnnxRef(const uint8_t* in_ptr_, uint8_t* out_ptr_,
                      int B, int C, int ID, int IH, int IW, int OD, int OH, int OW);
    void cubicRef(const uint8_t* in_ptr_, uint8_t* out_ptr_,
                  int B, int C, int IH, int IW, int OH, int OW);
    void pillowRef(const uint8_t* in_ptr_, uint8_t* out_ptr_,
                   int B, int C, int IH, int IW, int OH, int OW);
    void pillowRefNCHWAsNHWC(const uint8_t* in_ptr_, uint8_t* out_ptr_,
                             int B, int C, int IH, int IW, int OH, int OW);
    
    float coordTransToInput(int outCoord, float scale, int inShape, int outShape) const;
    int nearestRound(float origin, bool isDownsample, InterpolateNearestMode nearestMode);
    void linearOnnxCF(int outCoord, float scale, int inShape, int outShape,
                     int& index0, int& index1, float& weight0, float& weight1);
    static std::vector<float> getCubicCoeffs(float mantissa, float a);
    static float getPillowBilinearCoeffs(float m);
    static float getPillowBicubicCoeffs(float m);
    
    std::vector<int> auxTable;
    VectorDims srcDimPad5d, dstDim5d;
    int spatialDimSize;
    size_t dataRank;
    size_t srcDataSize;
    
    // Padding helper
    const uint8_t* padPreprocess(const MemoryCPtr& src, const MemoryPtr& dst);
    void preprocess(const float* in_ptr_, const VectorDims& inDims, const VectorDims& inDimsPad,
                   const std::vector<int>& padBegin, const std::vector<int>& padEnd,
                   std::vector<float>& outPadded);
    
    std::vector<uint8_t> m_padded_input;
    std::vector<uint8_t> pillow_working_buf;
    size_t m_threads_num = 0lu;
};

}  // namespace intel_cpu
}  // namespace ov