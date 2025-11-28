// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "nodes/kernels/aarch64/jit_uni_quantization.hpp"

#if defined(OPENVINO_ARCH_AARCH64) || defined(OPENVINO_ARCH_ARM64)

namespace ov::intel_cpu::node {

using namespace Xbyak_aarch64;

jit_uni_quantization_kernel_arm::jit_uni_quantization_kernel_arm(const jit_quantize_params& jqp)
    : jit_uni_quantize_kernel(jqp),
      can_jit_(jqp.src_prc == ov::element::f32 && jqp.dst_prc == ov::element::f32) {}

float jit_uni_quantization_kernel_arm::saturate_f32_to_u8(float v) {
    v = std::max(0.f, std::min(255.f, v));
    return std::nearbyintf(v);
}

float jit_uni_quantization_kernel_arm::saturate_f32_to_i8(float v) {
    v = std::max(-128.f, std::min(127.f, v));
    return std::nearbyintf(v);
}

void jit_uni_quantization_kernel_arm::kernel_scalar(const jit_quantize_call_args* args) {
    const auto& p = *args;
    const size_t work = p.work_amount;
    const size_t bs = p.block_size;
    const size_t src_step = p.src_step;
    const size_t dst_step = p.dst_step;

    const size_t src_elem_size = bs ? p.src_step / bs : sizeof(float);
    const size_t dst_elem_size = bs ? p.dst_step / bs : sizeof(float);
    const bool dst_f32 = dst_elem_size == sizeof(float);

    for (size_t w = 0; w < work; ++w) {
        const uint8_t* src_base = p.from + w * src_step;
        uint8_t* dst_base = const_cast<uint8_t*>(p.to + w * dst_step);

        const float* cl = p.crop_low;
        const float* ch = p.crop_high;
        const float* isc = p.input_scale;
        const float* ish = p.input_shift;
        const float* osc = p.output_scale;
        const float* osh = p.output_shift;

        for (size_t i = 0; i < bs; ++i) {
            float src_val = 0.f;
            if (src_elem_size == sizeof(float)) {
                src_val = reinterpret_cast<const float*>(src_base)[i];
            } else if (src_elem_size == sizeof(uint8_t)) {
                src_val = static_cast<float>(reinterpret_cast<const uint8_t*>(src_base)[i]);
            } else {
                src_val = static_cast<float>(reinterpret_cast<const int8_t*>(src_base)[i]);
            }
            float val = src_val * isc[i] + ish[i];
            val = std::nearbyintf(val);
            val = std::min(std::max(val, cl[i]), ch[i]);
            val = val * osc[i] + osh[i];

            if (dst_f32) {
                reinterpret_cast<float*>(dst_base)[i] = val;
            } else if (dst_elem_size == sizeof(uint8_t)) {
                reinterpret_cast<uint8_t*>(dst_base)[i] = static_cast<uint8_t>(saturate_f32_to_u8(val));
            } else {
                reinterpret_cast<int8_t*>(dst_base)[i] = static_cast<int8_t>(saturate_f32_to_i8(val));
            }
        }
    }
}

void jit_uni_quantization_kernel_arm::create_ker() {
    const bool disable_jit = std::getenv("OV_ARM_DISABLE_FQ_JIT") != nullptr;
    if (can_jit_ && !disable_jit) {
        jit_generator::create_kernel();
        ker_ = ov::intel_cpu::jit_kernel_cast<decltype(ker_)>(jit_ker());
    } else {
        ker_ = &kernel_scalar;
    }
}

void jit_uni_quantization_kernel_arm::generate() {
    // JIT only for planar float src->float dst path; broadcast is handled via explicit loads
    jit_generator::preamble();

    XReg reg_from = x1;
    XReg reg_to = x2;
    XReg reg_cl = x3;
    XReg reg_ch = x4;
    XReg reg_isc = x5;
    XReg reg_ish = x6;
    XReg reg_osc = x7;
    XReg reg_osh = x8;
    XReg reg_src_step = x9;
    XReg reg_dst_step = x10;
    XReg reg_bs = x11;
    XReg reg_work = x12;
    XReg reg_w = x13;
    XReg reg_idx = x14;
    XReg reg_src_ptr = x15;
    XReg reg_dst_ptr = x16;

    // load call args
    ldr(reg_from, ptr(dnnl::impl::cpu::aarch64::abi_param1, static_cast<int32_t>(offsetof(jit_quantize_call_args, from))));
    ldr(reg_to, ptr(dnnl::impl::cpu::aarch64::abi_param1, static_cast<int32_t>(offsetof(jit_quantize_call_args, to))));
    ldr(reg_cl, ptr(dnnl::impl::cpu::aarch64::abi_param1, static_cast<int32_t>(offsetof(jit_quantize_call_args, crop_low))));
    ldr(reg_ch, ptr(dnnl::impl::cpu::aarch64::abi_param1, static_cast<int32_t>(offsetof(jit_quantize_call_args, crop_high))));
    ldr(reg_isc, ptr(dnnl::impl::cpu::aarch64::abi_param1, static_cast<int32_t>(offsetof(jit_quantize_call_args, input_scale))));
    ldr(reg_ish, ptr(dnnl::impl::cpu::aarch64::abi_param1, static_cast<int32_t>(offsetof(jit_quantize_call_args, input_shift))));
    ldr(reg_osc, ptr(dnnl::impl::cpu::aarch64::abi_param1, static_cast<int32_t>(offsetof(jit_quantize_call_args, output_scale))));
    ldr(reg_osh, ptr(dnnl::impl::cpu::aarch64::abi_param1, static_cast<int32_t>(offsetof(jit_quantize_call_args, output_shift))));
    ldr(reg_src_step, ptr(dnnl::impl::cpu::aarch64::abi_param1, static_cast<int32_t>(offsetof(jit_quantize_call_args, src_step))));
    ldr(reg_dst_step, ptr(dnnl::impl::cpu::aarch64::abi_param1, static_cast<int32_t>(offsetof(jit_quantize_call_args, dst_step))));
    ldr(reg_bs, ptr(dnnl::impl::cpu::aarch64::abi_param1, static_cast<int32_t>(offsetof(jit_quantize_call_args, block_size))));
    ldr(reg_work, ptr(dnnl::impl::cpu::aarch64::abi_param1, static_cast<int32_t>(offsetof(jit_quantize_call_args, work_amount))));

    mov(reg_w, 0);

    Label l_work_loop;
    Label l_work_done;
    L(l_work_loop);
    cmp(reg_w, reg_work);
    b(Cond::GE, l_work_done);

    // compute src/dst base for this work item
    mul(reg_src_ptr, reg_w, reg_src_step);
    add(reg_src_ptr, reg_from, reg_src_ptr);
    mul(reg_dst_ptr, reg_w, reg_dst_step);
    add(reg_dst_ptr, reg_to, reg_dst_ptr);

    mov(reg_idx, 0);
    Label l_vec_loop;
    Label l_tail;
    Label l_tail_done;

    L(l_vec_loop);
    cmp(reg_idx, reg_bs);
    b(Cond::GE, l_tail);

    // if remaining <4 -> tail
    sub(x17, reg_bs, reg_idx);
    cmp(x17, 4);
    b(Cond::LT, l_tail);

    // vector body: load 4 floats
    lsl(x18, reg_idx, 2);
    add(x18, reg_src_ptr, x18);  // src offset bytes
    lsl(x19, reg_idx, 2);
    add(x19, reg_dst_ptr, x19);  // dst offset bytes
    lsl(x20, reg_idx, 2);
    add(x20, reg_cl, x20);
    lsl(x21, reg_idx, 2);
    add(x21, reg_ch, x21);
    lsl(x22, reg_idx, 2);
    add(x22, reg_isc, x22);
    lsl(x23, reg_idx, 2);
    add(x23, reg_ish, x23);
    lsl(x24, reg_idx, 2);
    add(x24, reg_osc, x24);
    lsl(x25, reg_idx, 2);
    add(x25, reg_osh, x25);

    ld1(v0.s, ptr(x18));  // src
    ld1(v1.s, ptr(x20));  // cl
    ld1(v2.s, ptr(x21));  // ch
    ld1(v3.s, ptr(x22));  // isc
    ld1(v4.s, ptr(x23));  // ish
    ld1(v5.s, ptr(x24));  // osc
    ld1(v6.s, ptr(x25));  // osh

    // val = src*isc + ish
    fmul(v0.s, v0.s, v3.s);
    fadd(v0.s, v0.s, v4.s);
    frintn(v0.s, v0.s);
    fmax(v0.s, v0.s, v1.s);
    fmin(v0.s, v0.s, v2.s);
    fmul(v7.s, v0.s, v5.s);
    fadd(v0.s, v7.s, v6.s);

    st1(v0.s, ptr(x19));

    add(reg_idx, reg_idx, 4);
    b(l_vec_loop);

    // Tail processing (scalar)
    L(l_tail);
    cmp(reg_idx, reg_bs);
    b(Cond::GE, l_tail_done);

    lsl(x18, reg_idx, 2);
    add(x18, reg_src_ptr, x18);
    lsl(x19, reg_idx, 2);
    add(x19, reg_dst_ptr, x19);
    lsl(x20, reg_idx, 2);
    add(x20, reg_cl, x20);
    lsl(x21, reg_idx, 2);
    add(x21, reg_ch, x21);
    lsl(x22, reg_idx, 2);
    add(x22, reg_isc, x22);
    lsl(x23, reg_idx, 2);
    add(x23, reg_ish, x23);
    lsl(x24, reg_idx, 2);
    add(x24, reg_osc, x24);
    lsl(x25, reg_idx, 2);
    add(x25, reg_osh, x25);

    ldr(s0, ptr(x18));
    ldr(s1, ptr(x20));
    ldr(s2, ptr(x21));
    ldr(s3, ptr(x22));
    ldr(s4, ptr(x23));
    ldr(s5, ptr(x24));
    ldr(s6, ptr(x25));

    fmul(s0, s0, s3);
    fadd(s0, s0, s4);
    frintn(s0, s0);
    fmax(s0, s0, s1);
    fmin(s0, s0, s2);
    fmul(s7, s0, s5);
    fadd(s0, s7, s6);

    str(s0, ptr(x19));

    add(reg_idx, reg_idx, 1);
    b(l_tail);

    L(l_tail_done);
    add(reg_w, reg_w, 1);
    b(l_work_loop);

    L(l_work_done);
    jit_generator::postamble();
}

}  // namespace ov::intel_cpu::node

#endif  // OPENVINO_ARCH_AARCH64 || OPENVINO_ARCH_ARM64
