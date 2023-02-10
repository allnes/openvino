// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#if defined(OV_CPU_WITH_ACL)

#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace ov {
namespace intel_cpu {

/**
* @brief Return ComputeLibrary DataType that corresponds to the given precision
* @param precision
* precision for convert
* @return ComputeLibrary DataType or UNKNOWN if precision is not mapped to DataType
*/
inline arm_compute::DataType precisionToAclDataType(InferenceEngine::Precision precision) {
    switch (precision) {
        case InferenceEngine::Precision::I8:    return arm_compute::DataType::S8;
        case InferenceEngine::Precision::U8:    return arm_compute::DataType::U8;
        case InferenceEngine::Precision::I16:   return arm_compute::DataType::S16;
        case InferenceEngine::Precision::U16:   return arm_compute::DataType::U16;
        case InferenceEngine::Precision::I32:   return arm_compute::DataType::S32;
        case InferenceEngine::Precision::U32:   return arm_compute::DataType::U32;
        case InferenceEngine::Precision::FP16:  return arm_compute::DataType::F16;
        case InferenceEngine::Precision::FP32:  return arm_compute::DataType::F32;
        case InferenceEngine::Precision::FP64:  return arm_compute::DataType::F64;
        case InferenceEngine::Precision::I64:   return arm_compute::DataType::S64;
        case InferenceEngine::Precision::BF16:  return arm_compute::DataType::BFLOAT16;
        default:                                return arm_compute::DataType::UNKNOWN;
    }
}

/**
* @brief Return ComputeLibrary DataType that corresponds to the given precision
* @param precision
* precision for convert
* @return ComputeLibrary DataType or UNKNOWN if precision is not mapped to DataType
*/
inline arm_compute::DataLayout getAclDataLayoutByMemoryDesc(MemoryDescCPtr desc) {

    if (desc->hasLayoutType(LayoutType::ncsp)) {
        if (desc->getShape().getRank() == 4) return arm_compute::DataLayout::NCHW;
        if (desc->getShape().getRank() == 5) return arm_compute::DataLayout::NCDHW;
        return arm_compute::DataLayout::UNKNOWN;
    } else {
        if (desc->getShape().getRank() == 4) return arm_compute::DataLayout::NHWC;
        if (desc->getShape().getRank() == 5) return arm_compute::DataLayout::NDHWC;
        return arm_compute::DataLayout::UNKNOWN;
    }
}

}   // namespace intel_cpu
}   // namespace ov

#endif