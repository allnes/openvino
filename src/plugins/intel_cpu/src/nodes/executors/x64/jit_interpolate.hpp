// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined(OPENVINO_ARCH_X86_64)

#include <memory>
#include <vector>
#include <unordered_map>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <common/primitive_attr.hpp>
#include <cpu/x64/xbyak/xbyak.h>
#include <cpu/x64/jit_generator.hpp>
#include <common/c_types_map.hpp>
#include <common/utils.hpp>

#include "cpu/x64/injectors/jit_uni_depthwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_quantization_injector.hpp"
#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/plugin/x64/jit_load_store_emitters.hpp"

#include "cpu_types.h"
#include "../interpolate.hpp"

namespace ov::intel_cpu::node {

struct jit_interpolate_config_params {
    InterpolateLayoutType layout = InterpolateLayoutType::planar;
    InterpolateMode mode = InterpolateMode::nearest;
    ov::element::Type src_prc;
    ov::element::Type dst_prc;
    int src_data_size = 0;
    int dst_data_size = 0;
    int indices_size = 0;
    int spatial_dim_size = 0;
    int C = 0, ID = 0, IH = 0, IW = 0, OD = 0, OH = 0, OW = 0;
    // for pillow
    int filterLenX = 0;
    int filterLenY = 0;
    int* bound = nullptr;
};

struct jit_interpolate_call_args {
    const void* src_ptr[MAX_INPUT_INTERPOLATE];
    const void* weight_ptr[MAX_INPUT_INTERPOLATE];
    const int* index;
    void* dst;
    size_t work_amount;
    size_t oc_off;
    // ptr to array of post op inputs pointers (flat list)
    const void* post_op_data;
};

struct jit_uni_interpolate_kernel {
    void (*ker_)(const jit_interpolate_call_args*) = nullptr;

    void operator()(const jit_interpolate_call_args* args) const {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_interpolate_kernel(jit_interpolate_config_params jcp, const dnnl_primitive_attr& attr)
        : jcp_(jcp), attr_(attr) {}
    virtual ~jit_uni_interpolate_kernel() = default;

    virtual void create_ker() = 0;

    jit_interpolate_config_params jcp_;
    const dnnl_primitive_attr& attr_;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct jit_uni_interpolate_kernel_f32 : public jit_uni_interpolate_kernel, public dnnl::impl::cpu::x64::jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_interpolate_kernel_f32)

    explicit jit_uni_interpolate_kernel_f32(jit_interpolate_config_params jcp, const dnnl_primitive_attr& attr);

    void create_ker() override;
    void generate() override;

private:
    using Vmm = typename dnnl::impl::cpu::x64::conditional3<
        isa == dnnl::impl::cpu::x64::sse41, 
        Xbyak::Xmm, 
        isa == dnnl::impl::cpu::x64::avx2, 
        Xbyak::Ymm, 
        Xbyak::Zmm
    >::type;

    const int vlen = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen;
    const int vector_step = vlen / sizeof(float);
    const int tail_step = jcp_.C % vector_step;
    const int scalar_step = 1;

    // Registers
    Xbyak::Reg64 reg_src = Xbyak::Reg64(8);
    Xbyak::Reg64 reg_src_aux = Xbyak::Reg64(15);
    Xbyak::Reg64 reg_src_aux1 = Xbyak::Reg64(11);
    Xbyak::Reg64 reg_src_aux2 = Xbyak::Reg64(12);
    Xbyak::Reg64 reg_dst = Xbyak::Reg64(9);
    Xbyak::Reg64 reg_work_amount = Xbyak::Reg64(13);
    Xbyak::Reg64 reg_index = Xbyak::Reg64(14);
    Xbyak::Reg64 reg_params = dnnl::impl::cpu::x64::abi_param1;

    Xbyak::Reg8 reg_tmp_8 = Xbyak::Reg8(10);
    Xbyak::Reg32 reg_tmp_32 = Xbyak::Reg32(10);
    Xbyak::Reg64 reg_tmp_64 = Xbyak::Reg64(10);

    Xbyak::Reg64 reg_oc_off = Xbyak::Reg64(0);
    Xbyak::Reg64 reg_post_ops_data = Xbyak::Reg64(3);
    Xbyak::Reg64 reg_d_weights = reg_tmp_64;
    Xbyak::Reg64 reg_d_bias = Xbyak::Reg64(1);
    Xbyak::Reg32 reg_index_offset = Xbyak::Reg32(2);

    // for cubic planar
    Xbyak::Reg64 reg_tbl_y = Xbyak::Reg64(6);
    Xbyak::Reg64 reg_tbl_x = Xbyak::Reg64(5);
    Xbyak::Reg64 reg_table = Xbyak::Reg64(2);

    // VMM registers
    Vmm vmm_val = Vmm(1);
    Vmm vmm_index = Vmm(0);
    Vmm vmm_zero = Vmm(2);
    Vmm vmm_mask = Vmm(3);
    Vmm vmm_d_weights = Vmm(4);
    Vmm vmm_d_bias = Vmm(5);

    // for linear
    Vmm vmm_weightT = Vmm(15);
    Vmm vmm_weightB = Vmm(14);
    Vmm vmm_weightL = Vmm(13);
    Vmm vmm_weightR = Vmm(12);
    Vmm vmm_weightF = Vmm(6);
    Vmm vmm_weightE = Vmm(7);
    Vmm vmm_valTL = Vmm(11);
    Vmm vmm_valTR = vmm_val;
    Vmm vmm_valBL = Vmm(9);
    Vmm vmm_valBR = Vmm(8);

    // for cubic
    Vmm vmm_src = Vmm(6);
    Xbyak::Xmm xmm_src = Xbyak::Xmm(6);
    Vmm vmm_dstX = Vmm(7);

    // Emitters and injectors
    std::vector<std::shared_ptr<dnnl::impl::cpu::x64::jit_uni_eltwise_injector<isa>>> eltwise_injectors;
    std::vector<std::shared_ptr<dnnl::impl::cpu::x64::jit_uni_depthwise_injector_f32<isa>>> depthwise_injectors;
    std::vector<std::shared_ptr<dnnl::impl::cpu::x64::jit_uni_quantization_injector_f32<isa>>> quantization_injectors;

    std::vector<size_t> load_pool_gpr_idxs;
    std::vector<size_t> store_pool_gpr_idxs;
    std::vector<size_t> store_pool_vec_idxs;

    // Implementation methods
    void nn_planar();
    void nn_blk();
    void nn_by_channel();
    void linear_onnx_c_gathered();
    void linear_onnx_planar();
    void linear_onnx_worker_1d();
    void linear_onnx_worker_2d();
    void cubic_planar();
    void cubic_c_gathered();
    void pillow_by_channel();
    void apply_post_ops();
    void prepare_cubic_planar_table();
    
    // Helper methods
    void load_scalar(Vmm vmm_dst, const Xbyak::Address& op, ov::element::Type src_prc);
    void store_scalar(const Xbyak::Address& op, Vmm vmm_src, ov::element::Type dst_prc);
    void load_vector(Vmm vmm_dst, const Xbyak::Address& op, ov::element::Type src_prc, int load_size);
    void store_vector(const Xbyak::Address& op, Vmm vmm_src, ov::element::Type dst_prc, int store_size);
    void uni_vpxor(const Vmm& x1, const Vmm& x2, const Vmm& op);
    void uni_vmovss(const Vmm& x, const Xbyak::Address& op);
    void uni_vmovd(const Vmm& x, const Xbyak::Reg32& reg);
    void uni_vmovq(const Vmm& x, const Xbyak::Reg64& reg);
    void uni_vpbroadcastd(const Vmm& x, const Xbyak::Address& op);
    void uni_vpbroadcastd(const Vmm& x, const Xbyak::Reg32& reg);
    void uni_vbroadcastss(const Vmm& x, const Xbyak::Address& op);
    void uni_vbroadcastss(const Vmm& x, const Xbyak::Xmm& x2);
    void uni_vfmadd213ps(const Vmm& x1, const Vmm& x2, const Vmm& op);
    void uni_vfmadd231ps(const Vmm& x1, const Vmm& x2, const Vmm& op);
    void uni_vfmsub213ps(const Vmm& x1, const Vmm& x2, const Vmm& op);
    void uni_vfnmadd213ps(const Vmm& x1, const Vmm& x2, const Vmm& op);
    void uni_vfnmadd231ps(const Vmm& x1, const Vmm& x2, const Vmm& op);
    void uni_vaddps(const Vmm& x1, const Vmm& x2, const Vmm& op);
    void uni_vsubps(const Vmm& x1, const Vmm& x2, const Vmm& op);
    void uni_vmulps(const Vmm& x1, const Vmm& x2, const Vmm& op);
    void uni_vblendvps(const Vmm& x1, const Vmm& x2, const Vmm& op, const Vmm& mask);
    void uni_vcmpps(const Vmm& x1, const Vmm& x2, const Vmm& op, const unsigned char imm);
    void uni_vcvtps2dq(const Vmm& x1, const Vmm& x2);
    void uni_vcvtdq2ps(const Vmm& x1, const Vmm& x2);
    void uni_vpackssdw(const Vmm& x1, const Vmm& x2, const Vmm& op);
    void uni_vpackuswb(const Vmm& x1, const Vmm& x2, const Vmm& op);
    void uni_vpmaxsd(const Vmm& x1, const Vmm& x2, const Vmm& op);
    void uni_vpminsd(const Vmm& x1, const Vmm& x2, const Vmm& op);
    void uni_vpsubd(const Vmm& x1, const Vmm& x2, const Vmm& op);
    void uni_vpaddd(const Vmm& x1, const Vmm& x2, const Vmm& op);
    void uni_vpmulld(const Vmm& x1, const Vmm& x2, const Vmm& op);
    void uni_vpslld(const Vmm& x1, const Vmm& x2, const unsigned char imm);
    void uni_vpsrld(const Vmm& x1, const Vmm& x2, const unsigned char imm);
    void uni_vpand(const Vmm& x1, const Vmm& x2, const Vmm& op);
    void uni_vpor(const Vmm& x1, const Vmm& x2, const Vmm& op);
    void uni_vpxor(const Vmm& x1, const Vmm& x2, const Vmm& op);
    void uni_vpshufd(const Vmm& x1, const Vmm& x2, const unsigned char imm);
    void uni_vmovups(const Vmm& x, const Xbyak::Address& op);
    void uni_vmovups(const Xbyak::Address& op, const Vmm& x);
    void uni_vmovdqu(const Vmm& x, const Xbyak::Address& op);
    void uni_vmovdqu(const Xbyak::Address& op, const Vmm& x);
    void uni_vmovq(const Vmm& x, const Vmm& x2);
    void uni_vmovd(const Vmm& x, const Vmm& x2);
    void uni_vshufps(const Vmm& x1, const Vmm& x2, const Vmm& op, const unsigned char imm);
    void uni_vzeroupper();
    void uni_vzeroupper_safe();
};

// Helper function
bool isFloatCompatible(ov::element::Type prc);

// Macro
#define GET_OFF(field) offsetof(jit_interpolate_call_args, field)

} // namespace ov::intel_cpu::node

#endif // OPENVINO_ARCH_X86_64