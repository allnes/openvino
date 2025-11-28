// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nodes/fake_quantize.h"
#include "utils/cpu_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>

// ARM JIT FakeQuantize kernel lives under kernels/aarch64 to keep JIT code colocated.

#if defined(OPENVINO_ARCH_AARCH64) || defined(OPENVINO_ARCH_ARM64)

// oneDNN's jit_generator.hpp defines ABI macros that can clash, keep same guards as other AArch64 kernels
#undef abi_param1
#undef abi_param2
#undef abi_param3
#undef abi_param4
#undef abi_param5
#undef abi_param6
#undef abi_param7
#undef abi_param8
#undef abi_not_param1

#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>

namespace ov::intel_cpu::node {

namespace aarch64 = dnnl::impl::cpu::aarch64;

struct jit_uni_quantization_kernel_arm : public jit_uni_quantize_kernel, public aarch64::jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_quantization_kernel_arm)
    explicit jit_uni_quantization_kernel_arm(const jit_quantize_params& jqp);

    void create_ker() override;
    void generate() override;

private:
    static float saturate_f32_to_u8(float v);
    static float saturate_f32_to_i8(float v);
    static void kernel_scalar(const jit_quantize_call_args* args);

    bool can_jit_ = false;
};

}  // namespace ov::intel_cpu::node

#endif  // OPENVINO_ARCH_AARCH64 || OPENVINO_ARCH_ARM64
