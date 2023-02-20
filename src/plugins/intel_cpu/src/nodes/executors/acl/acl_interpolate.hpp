// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../interpolate.hpp"

namespace ov {
namespace intel_cpu {

class ACLInterpolateExecutor : public InterpolateExecutor {
public:
    ACLInterpolateExecutor(const ExecutorContext::CPtr context) : InterpolateExecutor(context) {}

    bool init(const InterpolateAttrs& interpolateAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;

    void exec(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_) override;

    impl_desc_type getImplType() const override {
        return implType;
    }

private:
    impl_desc_type implType = impl_desc_type::acl;
    InterpolateAttrs aclInterpolateAttrs;
    arm_compute::SamplingPolicy acl_coord;
    arm_compute::InterpolationPolicy acl_policy;
    bool antialias{};
    arm_compute::Tensor srcTensor, dstTensor;
};

class ACLInterpolateExecutorBuilder : public InterpolateExecutorBuilder {
public:
    bool isSupported(const InterpolateAttrs& interpolateAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override;

    InterpolateExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<ACLInterpolateExecutor>(context);
    }
private:
    static bool isSupportedConfiguration(const InterpolateAttrs& interpolateAttrs,
                                  const std::vector<MemoryDescPtr>& srcDescs,
                                  const std::vector<MemoryDescPtr>& dstDescs) ;
};
}
}