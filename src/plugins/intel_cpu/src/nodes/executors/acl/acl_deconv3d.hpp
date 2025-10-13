// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <arm_compute/runtime/NEON/functions/NEDeconvolutionLayer.h>
#include <memory>
#include <vector>

#include "nodes/executors/deconv.hpp"
#include "nodes/executors/executor.hpp"

namespace ov::intel_cpu {

// ACL-based 3D Deconvolution executor using multiple 2D NEDeconvolutionLayer passes per depth kernel slice.
// Narrow, UNet3D-focused support:
//  - Rank-5 tensors (N, C, D, H, W) or nspc (N, D, H, W, C)
//  - Batch N can be >1 at init-time checks (dummy shapes); runtime path is validated on N=1
//  - No groups
//  - f16 or f32
//  - Strides (2,2,2), dilations (1,1,1)
//  - Pads (0,0,0), output_padding (0,0,0)
//  - Kernel D in {2} (kH,kW arbitrary >=1)
//  - Layout nspc preferred to enable contiguous per-depth planes
class AclDeconv3DExecutor : public DeconvExecutor {
public:
    explicit AclDeconv3DExecutor(ExecutorContext::CPtr context);

    bool init(const DeconvAttrs& deconvAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr& attr) override;

    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void* post_ops_data_) override;

    [[nodiscard]] impl_desc_type getImplType() const override { return impl_desc_type::acl; }

    static bool customIsSupported3D(const DeconvAttrs& deconvAttrs,
                                    const std::vector<MemoryDescPtr>& srcDescs,
                                    const std::vector<MemoryDescPtr>& dstDescs);

private:
    // Per-kD function and weights
    std::vector<std::unique_ptr<arm_compute::NEDeconvolutionLayer>> m_layers;
    std::vector<std::shared_ptr<arm_compute::Tensor>> m_weiTensors;
    std::vector<std::vector<uint8_t>> m_weiStorage;  // raw transposed weights per kD

    // Shared src/dst tensors (shape-constant); import pointers per invocation
    std::shared_ptr<arm_compute::Tensor> m_srcTensor;
    std::shared_ptr<arm_compute::Tensor> m_dstTensor;
    std::shared_ptr<arm_compute::Tensor> m_biasTensor;
    // Temporary NHWC planes for NCSP fallback and for output_padding handling
    std::vector<uint8_t> m_srcPlaneTmp;
    std::vector<uint8_t> m_dstPlaneTmp;
    std::vector<uint8_t> m_dstSmallPlaneTmp;

    // Cached geometry
    bool m_nspc = true;
    bool m_hasBias = false;
    size_t m_N = 1, m_Cin = 0, m_Cout = 0;
    size_t m_Din = 0, m_Hin = 0, m_Win = 0;
    size_t m_Dout = 0, m_Hout = 0, m_Wout = 0;
    size_t m_effHout = 0, m_effWout = 0;  // actual H/W configured for ACL (Hout - output_padding)
    size_t m_kD = 0, m_kH = 0, m_kW = 0;
    size_t m_sD = 0, m_sH = 0, m_sW = 0;
    // Cache
    const uint8_t* m_lastWeiBasePtr = nullptr;
};

class AclDeconv3DExecutorBuilder : public DeconvExecutorBuilder {
public:
    [[nodiscard]] bool isSupported(const DeconvAttrs& deconvAttrs,
                                   const std::vector<MemoryDescPtr>& srcDescs,
                                   const std::vector<MemoryDescPtr>& dstDescs) const override {
        return AclDeconv3DExecutor::customIsSupported3D(deconvAttrs, srcDescs, dstDescs);
    }
    [[nodiscard]] DeconvExecutorPtr makeExecutor(ExecutorContext::CPtr context) const override {
        return std::make_shared<AclDeconv3DExecutor>(context);
    }
};

}  // namespace ov::intel_cpu
