// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ov::intel_cpu {

// User-data attributes survive CPU compiled-model export/import through the IR serializer.
inline constexpr char ConvolutionInputZeroPoint[] = "user_dataCPUConvolutionInputZeroPoint";
inline constexpr char ConvolutionDequantizationScales[] = "user_dataCPUConvolutionDequantizationScales";

}  // namespace ov::intel_cpu
