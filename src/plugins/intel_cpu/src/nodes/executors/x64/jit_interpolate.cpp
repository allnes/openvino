// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_interpolate.hpp"

#if defined(OPENVINO_ARCH_X86_64)

using namespace dnnl::impl;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

namespace ov::intel_cpu::node {

bool isFloatCompatible(ov::element::Type prc) {
    return one_of(prc, ov::element::f32, ov::element::bf16, ov::element::f16, ov::element::f64);
}

template <cpu_isa_t isa>
jit_uni_interpolate_kernel_f32<isa>::jit_uni_interpolate_kernel_f32(jit_interpolate_config_params jcp, const dnnl_primitive_attr& attr)
    : jit_uni_interpolate_kernel(jcp, attr), jit_generator(jit_name()) {}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::create_ker() {
    jit_generator::create_kernel();
    ker_ = (decltype(ker_))jit_ker();
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::generate() {
    // TODO: Move actual JIT implementation from interpolate.cpp
    // This is a placeholder that will be filled with the actual JIT code
    
    // For now, we'll add a basic structure
    this->preamble();
    
    // Basic JIT code structure will be moved here from interpolate.cpp
    // Including all the switch statements and method calls
    
    this->postamble();
}

// Implementation placeholders for the main methods
template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::nn_planar() {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::nn_blk() {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::nn_by_channel() {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::linear_onnx_c_gathered() {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::linear_onnx_planar() {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::linear_onnx_worker_1d() {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::linear_onnx_worker_2d() {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::cubic_planar() {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::cubic_c_gathered() {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::pillow_by_channel() {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::apply_post_ops() {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::prepare_cubic_planar_table() {
    // TODO: Move implementation from interpolate.cpp
}

// Helper methods placeholders
template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::load_scalar(Vmm vmm_dst, const Xbyak::Address& op, ov::element::Type src_prc) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::store_scalar(const Xbyak::Address& op, Vmm vmm_src, ov::element::Type dst_prc) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::load_vector(Vmm vmm_dst, const Xbyak::Address& op, ov::element::Type src_prc, int load_size) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::store_vector(const Xbyak::Address& op, Vmm vmm_src, ov::element::Type dst_prc, int store_size) {
    // TODO: Move implementation from interpolate.cpp
}

// Uni-functions placeholders
template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vpxor(const Vmm& x1, const Vmm& x2, const Vmm& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vmovss(const Vmm& x, const Xbyak::Address& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vmovd(const Vmm& x, const Xbyak::Reg32& reg) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vmovq(const Vmm& x, const Xbyak::Reg64& reg) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vpbroadcastd(const Vmm& x, const Xbyak::Address& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vpbroadcastd(const Vmm& x, const Xbyak::Reg32& reg) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vbroadcastss(const Vmm& x, const Xbyak::Address& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vbroadcastss(const Vmm& x, const Xbyak::Xmm& x2) {
    // TODO: Move implementation from interpolate.cpp
}

// More uni-functions will be added as needed...
template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vfmadd213ps(const Vmm& x1, const Vmm& x2, const Vmm& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vfmadd231ps(const Vmm& x1, const Vmm& x2, const Vmm& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vfmsub213ps(const Vmm& x1, const Vmm& x2, const Vmm& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vfnmadd213ps(const Vmm& x1, const Vmm& x2, const Vmm& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vfnmadd231ps(const Vmm& x1, const Vmm& x2, const Vmm& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vaddps(const Vmm& x1, const Vmm& x2, const Vmm& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vsubps(const Vmm& x1, const Vmm& x2, const Vmm& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vmulps(const Vmm& x1, const Vmm& x2, const Vmm& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vblendvps(const Vmm& x1, const Vmm& x2, const Vmm& op, const Vmm& mask) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vcmpps(const Vmm& x1, const Vmm& x2, const Vmm& op, const unsigned char imm) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vcvtps2dq(const Vmm& x1, const Vmm& x2) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vcvtdq2ps(const Vmm& x1, const Vmm& x2) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vpackssdw(const Vmm& x1, const Vmm& x2, const Vmm& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vpackuswb(const Vmm& x1, const Vmm& x2, const Vmm& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vpmaxsd(const Vmm& x1, const Vmm& x2, const Vmm& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vpminsd(const Vmm& x1, const Vmm& x2, const Vmm& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vpsubd(const Vmm& x1, const Vmm& x2, const Vmm& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vpaddd(const Vmm& x1, const Vmm& x2, const Vmm& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vpmulld(const Vmm& x1, const Vmm& x2, const Vmm& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vpslld(const Vmm& x1, const Vmm& x2, const unsigned char imm) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vpsrld(const Vmm& x1, const Vmm& x2, const unsigned char imm) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vpand(const Vmm& x1, const Vmm& x2, const Vmm& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vpor(const Vmm& x1, const Vmm& x2, const Vmm& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vpxor(const Vmm& x1, const Vmm& x2, const Vmm& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vpshufd(const Vmm& x1, const Vmm& x2, const unsigned char imm) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vmovups(const Vmm& x, const Xbyak::Address& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vmovups(const Xbyak::Address& op, const Vmm& x) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vmovdqu(const Vmm& x, const Xbyak::Address& op) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vmovdqu(const Xbyak::Address& op, const Vmm& x) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vmovq(const Vmm& x, const Vmm& x2) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vmovd(const Vmm& x, const Vmm& x2) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vshufps(const Vmm& x1, const Vmm& x2, const Vmm& op, const unsigned char imm) {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vzeroupper() {
    // TODO: Move implementation from interpolate.cpp
}

template <cpu_isa_t isa>
void jit_uni_interpolate_kernel_f32<isa>::uni_vzeroupper_safe() {
    // TODO: Move implementation from interpolate.cpp
}

// Template instantiations
template class jit_uni_interpolate_kernel_f32<sse41>;
template class jit_uni_interpolate_kernel_f32<avx2>;
template class jit_uni_interpolate_kernel_f32<avx512_core>;

} // namespace ov::intel_cpu::node

#endif // OPENVINO_ARCH_X86_64