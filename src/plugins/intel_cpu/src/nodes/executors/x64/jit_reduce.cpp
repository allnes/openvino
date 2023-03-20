// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_reduce.hpp"

// #include <dnnl_extension_utils.h>
// #include "utils/bfloat16.hpp"
// #include "ie_parallel.hpp"
// #include "emitters/x64/jit_load_store_emitters.hpp"
// #include "emitters/x64/jit_bf16_emitters.hpp"

// #include <cpu/x64/jit_generator.hpp>
// #include <cpu/x64/jit_uni_eltwise.hpp>
// #include <cpu/x64/injectors/jit_uni_depthwise_injector.hpp>
// #include <cpu/x64/injectors/jit_uni_quantization_injector.hpp>
// #include <cpu/x64/injectors/jit_uni_eltwise_injector.hpp>

#include <dnnl_extension_utils.h>
#include <onednn/dnnl.h>
#include "utils/bfloat16.hpp"
#include "ie_parallel.hpp"
#include "emitters/x64/jit_bf16_emitters.hpp"
#include <cpu/x64/jit_generator.hpp>
#include <cpu/x64/jit_uni_eltwise.hpp>
#include <cpu/x64/injectors/jit_uni_depthwise_injector.hpp>
#include <cpu/x64/injectors/jit_uni_quantization_injector.hpp>
#include <cpu/x64/injectors/jit_uni_eltwise_injector.hpp>
#include <common/primitive_hashing_utils.hpp>

#define SET_SRC_DIM_VALUE(batch, channel, depth, height, width) IB = batch;   \
                                                                IC = channel; \
                                                                ID = depth;   \
                                                                IH = height;  \
                                                                IW = width;
#define SET_DST_DIM_VALUE(batch, channel, depth, height, width) OB = batch;   \
                                                                OC = channel; \
                                                                OD = depth;   \
                                                                OH = height;  \
                                                                OW = width;

#define GET_OFF(field) offsetof(jit_reduce_call_args, field)
#define GET_OFF_POST(field) offsetof(jit_reduce_post_call_args, field)

#define GET_PTR_N_PLN              const uint8_t    *in_ptr_n      = in_ptr       + src_data_size * ib * IC * ID * IH * IW;               \
                                         uint8_t    *out_ptr_n     = out_ptr      + dst_data_size * ob * OC * OD * OH * OW;
#define GET_PTR_NC_PLN             const uint8_t    *in_ptr_nc     = in_ptr_n     + src_data_size * ic * ID * IH * IW;                    \
                                         uint8_t    *out_ptr_nc    = out_ptr_n    + dst_data_size * oc * OD * OH * OW;
#define GET_PTR_NCD_PLN            const uint8_t    *in_ptr_ncd    = in_ptr_nc    + src_data_size * id * IH * IW;                         \
                                         uint8_t    *out_ptr_ncd   = out_ptr_nc   + dst_data_size * od * OH * OW;
#define GET_PTR_NCDH_PLN           const uint8_t    *in_ptr_ncdh   = in_ptr_ncd   + src_data_size * ih * IW;                              \
                                         uint8_t    *out_ptr_ncdh  = out_ptr_ncd  + dst_data_size * oh * OW;
#define GET_PTR_NCD_BASE_PTR_N_PLN const uint8_t    *in_ptr_ncd    = in_ptr_n     + src_data_size * (ic * ID + id) * IH * IW;             \
                                         uint8_t    *out_ptr_ncd   = out_ptr_n    + dst_data_size * (oc * OD + od) * OH * OW;
#define GET_PTR_N_BLK              const uint8_t    *in_ptr_n      = in_ptr       + src_data_size * ib * ICB * ID * IH * IW * blk_size;   \
                                         uint8_t    *out_ptr_n     = out_ptr      + dst_data_size * ob * OCB * OD * OH * OW * blk_size;
#define GET_PTR_NC_BLK             const uint8_t    *in_ptr_nc     = in_ptr_n     + src_data_size * icb * ID * IH * IW * blk_size;        \
                                         uint8_t    *out_ptr_nc    = out_ptr_n    + dst_data_size * ocb * OD * OH * OW * blk_size;
#define GET_PTR_NCD_BLK            const uint8_t    *in_ptr_ncd    = in_ptr_nc    + src_data_size * id * IH * IW * blk_size;              \
                                         uint8_t    *out_ptr_ncd   = out_ptr_nc   + dst_data_size * od * OH * OW * blk_size;
#define GET_PTR_NCDH_BLK           const uint8_t    *in_ptr_ncdh   = in_ptr_ncd   + src_data_size * ih * IW * blk_size;                   \
                                         uint8_t    *out_ptr_ncdh  = out_ptr_ncd  + dst_data_size * oh * OW * blk_size;
#define GET_PTR_NCDHW_BLK          const uint8_t    *in_ptr_ncdhw  = in_ptr_ncdh  + src_data_size * iw * blk_size;                        \
                                         uint8_t    *out_ptr_ncdhw = out_ptr_ncdh + dst_data_size * ow * blk_size;
#define GET_PTR_NCD_BASE_PTR_N_BLK const uint8_t    *in_ptr_ncd    = in_ptr_n     + src_data_size * (icb * ID + id) * IH * IW * blk_size; \
                                         uint8_t    *out_ptr_ncd   = out_ptr_n    + dst_data_size * (ocb * OD + od) * OH * OW * blk_size;

namespace ov {
namespace intel_cpu {

using namespace dnnl;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;
using namespace InferenceEngine;

struct ReduceKey {
    jit_reduce_config_params jcp;
    dnnl::post_ops postOps;

    size_t hash() const;
    bool operator==(const ReduceKey& rhs) const;
};

size_t ReduceKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    seed = hash_combine(seed, jcp.layout);
    seed = hash_combine(seed, jcp.reduce_mode);
    seed = hash_combine(seed, jcp.src_dt);
    seed = hash_combine(seed, jcp.dst_dt);
    seed = get_post_op_hash(seed, *postOps.get());

    return seed;
}

bool ReduceKey::operator==(const ReduceKey &rhs) const {
    return jcp.layout == rhs.jcp.layout && jcp.reduce_mode == rhs.jcp.reduce_mode &&
           jcp.src_dt == rhs.jcp.src_dt && jcp.dst_dt == rhs.jcp.dst_dt && *postOps.get() == *rhs.postOps.get();
}

// some utility functions
static inline bool isFloatCompatible(memory::data_type type) {
    return memory::data_type::f32 == type || memory::data_type::bf16 == type;
}

template <cpu_isa_t isa>
struct jit_uni_reduce_kernel_f32 : public jit_uni_reduce_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_reduce_kernel_f32)

    explicit jit_uni_reduce_kernel_f32(jit_reduce_config_params jcp)
    : jit_uni_reduce_kernel(jcp), jit_generator(jit_name()) {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        if (jcp_.reduce_mode == Algorithm::ReduceLogSumExp) {
            exp_injector = std::make_shared<jit_uni_eltwise_injector_f32<isa>>(this, alg_kind::eltwise_exp, 0.f, 0.f, 1.f);
        }

        if (mayiuse(avx512_core))
            uni_vcvtneps2bf16 = std::make_shared<jit_uni_vcvtneps2bf16>(this, isa);

        this->preamble();

        planar_layout = jcp_.layout == ReduceLayoutType::reduce_ncsp || jcp_.layout == ReduceLayoutType::reduce_nspc;

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        mov(reg_work_batch, ptr[reg_params + GET_OFF(work_batch)]);
        if (planar_layout)
            mov(reg_reduce_w, ptr[reg_params + GET_OFF(reduce_w)]);

        if (jcp_.reduce_mode == Algorithm::ReduceAnd || jcp_.reduce_mode == Algorithm::ReduceL1 || jcp_.reduce_mode == Algorithm::ReduceMax ||
            jcp_.reduce_mode == Algorithm::ReduceMin || jcp_.reduce_mode == Algorithm::ReduceProd || jcp_.reduce_mode == Algorithm::ReduceOr) {
            mov(reg_table, l_table);
        }

        if (isa == cpu::x64::avx512_core || jcp_.reduce_mode == Algorithm::ReduceAnd || jcp_.reduce_mode == Algorithm::ReduceOr)
            uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        if ((isa == cpu::x64::avx512_core && jcp_.reduce_mode == Algorithm::ReduceAnd) || jcp_.reduce_mode == Algorithm::ReduceOr) {
            uni_vmovups(vmm_aux, table_val(0));
        }

        reduce_main();
        reduce_tail();

        this->postamble();

        if (mayiuse(avx512_core))
            uni_vcvtneps2bf16->emit_data();

        if (jcp_.reduce_mode == Algorithm::ReduceAnd || jcp_.reduce_mode == Algorithm::ReduceL1 || jcp_.reduce_mode == Algorithm::ReduceMax ||
            jcp_.reduce_mode == Algorithm::ReduceMin || jcp_.reduce_mode == Algorithm::ReduceProd || jcp_.reduce_mode == Algorithm::ReduceOr) {
            prepare_aux_table();
        } else if (jcp_.reduce_mode == Algorithm::ReduceLogSumExp) {
            exp_injector->prepare_table();
        }
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;
    size_t vlen = cpu_isa_traits<isa>::vlen;
    bool planar_layout = false;

    Xbyak::Address table_val(int index) { return ptr[reg_table + index * vlen]; }

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 reg_idx = rdx;
    Xbyak::Reg64 reg_work_amount = r10;
    Xbyak::Reg64 reg_reduce_w = r11;
    Xbyak::Reg64 reg_reduce_stride = r12;
    Xbyak::Reg64 reg_work_batch = r13;
    Xbyak::Reg64 reg_table = r14;
    Xbyak::Reg64 reg_params = abi_param1;

    Xbyak::Reg8 reg_tmp_8 = r15b;
    Xbyak::Reg32 reg_tmp_32 = r15d;
    Xbyak::Reg64 reg_tmp_64 = r15;

    Xbyak::Reg64 reg_src_aux = rax;
    Xbyak::Reg64 reg_work_batch_aux = rbx;

    Vmm vmm_aux = Vmm(0);
    Xmm xmm_aux = Xmm(0);
    Vmm vmm_src = Vmm(1);
    Xmm xmm_src = Xmm(1);
    Vmm vmm_dst = Vmm(2);
    Xmm xmm_dst = Xmm(2);
    Vmm vmm_zero = Vmm(3);
    Xmm xmm_zero = Xmm(3);
    Vmm vmm_dst_aux = Vmm(4);
    Xmm xmm_aux1 = Xmm(5);
    Xmm xmm_aux2 = Xmm(6);
    Xmm xmm_aux3 = Xmm(7);
    Vmm vmm_idx = Vmm(8);
    Vmm vmm_mask = Vmm(9);

    const Xbyak::Opmask k_mask = Xbyak::Opmask(1);

    Xbyak::Label l_table;

    std::shared_ptr<jit_uni_vcvtneps2bf16> uni_vcvtneps2bf16;
    std::shared_ptr<jit_uni_eltwise_injector_f32<isa>> exp_injector;

    inline void reduce_main() {
        // ================================================================
        // ***isa: AVX512***
        // ReduceAnd (Logical And)
        // step 1: init dst 0x3f800000 (1.0f)
        //              aux 0x3f800000 (1.0f)
        //             zero 0x00000000 (0.0f)
        // step 2: if src equals 0, set mask bit 0, else set mask bit 1
        // step 3: src = mask bit == 0 ? zero : aux
        // step 4: dst = dst & src
        //                  src    mask_bit    new_src    dst    new_dst
        //         case 1    ~0        1         1.0f     1.0f     1.0f
        //         case 2     0        0         0.0f     1.0f     0.0f
        //         case 3    ~0        1         1.0f     0.0f     0.0f
        //         case 4     0        0         0.0f     0.0f     0.0f
        // step 5: loop: offset src, and do step 2 and step 3
        //
        // ReduceOr (Logical Or)
        // step 1: init dst 0x00000000 (0.0f)
        //              aux 0x3f800000 (1.0f)
        //             zero 0x00000000 (0.0f)
        // step 2: if src equals 0, set mask bit 0, else set mask bit 1
        // step 3: src = mask bit == 0 ? zero : aux
        // step 4: dst = dst | src
        //                  src    mask_bit    new_src    dst    new_dst
        //         case 1     0        0         0.0f     0.0f     0.0f
        //         case 2    ~0        1         1.0f     0.0f     1.0f
        //         case 3     0        0         0.0f     1.0f     1.0f
        //         case 4    ~0        1         1.0f     1.0f     1.0f
        // step 5: loop: offset src, and do step 2 and step 3
        // ================================================================
        // ***isa: OTHER***
        // ReduceAnd (Logical And)
        // step 1: init dst 0x3f800000 (1.0f)
        // step 2: if src equals 0, set it 0x00000000, else set 0xffffffff
        // step 3: dst = dst & src
        //         0x3f800000 = 0x3f800000 & 0xffffffff (result: 1.0f)
        //         0x00000000 = 0x3f800000 & 0x00000000 (result: 0.0f)
        //         0x00000000 = 0x00000000 & 0xffffffff (result: 0.0f)
        //         0x00000000 = 0x00000000 & 0x00000000 (result: 0.0f)
        // step 4: loop: offset src, and do step 2 and step 3
        //
        // ReduceOr (Logical Or)
        // step 1: init dst 0x00000000 (0.0f)
        //              aux 0x3f800000 (1.0f)
        // step 2: dst = dst | src
        //         0x00000000 = 0x00000000 | 0x00000000
        //                  A = 0x00000000 | A
        //                  A =          A | 0x00000000
        //                  C =          A | B
        // (A, B stand for number other than 0x00000000)
        // step 3: loop: offset src, and do step 2
        // step 4: if dst equals 0, set it 0x00000000, else set 0xffffffff
        // step 5: dst = dst & aux
        //         0x00000000 = 0x00000000 & 0x3f800000 (result: 0.0f)
        //         0x3f800000 = 0xffffffff & 0x3f800000 (result: 1.0f)
        // ================================================================
        Xbyak::Label reduce_to_vector_label;
        Xbyak::Label reduce_to_scalar_label;
        Xbyak::Label reduce_to_gather_label;
        Xbyak::Label reduce_main_end_label;
        if (planar_layout) {
            cmp(reg_work_batch, 0);
            je(reduce_to_gather_label, T_NEAR);

            cmp(reg_reduce_w, 1); // planar layout reducing W
            je(reduce_to_scalar_label, T_NEAR);
        }

        // store vmm_dst directly into memory after reducing
        // cases: [planar layout reducing other dimensions but W] [blocked layout]
        L(reduce_to_vector_label);
        {
            int step = vlen / sizeof(float) < 8 ? 8 : vlen / sizeof(float);
            cmp(reg_work_amount, step);
            jl(reduce_main_end_label, T_NEAR); //avoid illegal loading and storing

            if (jcp_.reduce_mode == Algorithm::ReduceL1) {
                uni_vmovups(vmm_aux, table_val(1));
            }

            // load
            load_dst_vector();

            // reduce
            reduce_kernel();

            // store
            store_dst_vector();

            jmp(reduce_main_end_label, T_NEAR);
        }

        // reduce vector in vmm_dst to be a scalar before store into memory
        // cases: [planar layout reducing W]
        L(reduce_to_scalar_label);
        {
            // init dst, dst loading is embedded in horiz_reduce_store
            switch (jcp_.reduce_mode) {
                case Algorithm::ReduceAnd:
                case Algorithm::ReduceProd:
                    uni_vmovups(vmm_dst, table_val(0));
                    break;
                case Algorithm::ReduceL1:
                    uni_vmovups(vmm_aux, table_val(1));
                    uni_vpxor(vmm_dst, vmm_dst, vmm_dst);
                    break;
                case Algorithm::ReduceL2:
                case Algorithm::ReduceLogSum:
                case Algorithm::ReduceLogSumExp:
                case Algorithm::ReduceMean:
                case Algorithm::ReduceOr:
                case Algorithm::ReduceSum:
                case Algorithm::ReduceSumSquare:
                    uni_vpxor(vmm_dst, vmm_dst, vmm_dst);
                    break;
                case Algorithm::ReduceMax:
                    if (isFloatCompatible(jcp_.dst_dt))
                        uni_vmovups(vmm_dst, table_val(2));
                    else
                        uni_vmovups(vmm_dst, table_val(4));
                    break;
                case Algorithm::ReduceMin:
                    if (isFloatCompatible(jcp_.dst_dt))
                        uni_vmovups(vmm_dst, table_val(3));
                    else
                        uni_vmovups(vmm_dst, table_val(5));
                    break;
                default:
                    assert(!"unsupported reduce mode");
            }
            // reduce
            reduce_main_loop();
            if (jcp_.reduce_mode == Algorithm::ReduceOr && isa != cpu::x64::avx512_core) {
                uni_cmpneqps(vmm_dst, vmm_dst, vmm_zero);
                uni_vandps(vmm_dst, vmm_dst, vmm_aux);
            }
            // store
            // store after horizontal calculation and calculation with loaded original ptr[reg_dst]
            horiz_reduce_store(vmm_dst, jcp_.dst_dt, true);

            jmp(reduce_main_end_label, T_NEAR);
        }

        // load vmm_src with gather, then store vmm_dst directly into memory after reducing
        // cases: [planar layout reducing small W]
        L(reduce_to_gather_label);
        {
            int step = 1;
            cmp(reg_work_amount, step);
            jl(reduce_main_end_label, T_NEAR); //avoid illegal loading and storing

            mov(reg_idx, ptr[reg_params + GET_OFF(idx)]);
            uni_vmovdqu(vmm_idx, ptr[reg_idx]);

            if (jcp_.reduce_mode == Algorithm::ReduceL1) {
                uni_vmovups(vmm_aux, table_val(1));
            }

            // load
            load_dst_vector();

            // reduce
            Xbyak::Label reduce_loop_label;
            Xbyak::Label reduce_loop_end_label;
            L(reduce_loop_label);
            {
                cmp(reg_work_amount, step);
                jl(reduce_loop_end_label, T_NEAR);

                reduce_gather(vmm_dst, 0);
                if (isa == cpu::x64::sse41) {
                    reduce_gather(vmm_dst_aux, 4 * jcp_.src_data_size);
                }

                add(reg_src, step * jcp_.src_data_size);
                sub(reg_work_amount, step);
                jmp(reduce_loop_label, T_NEAR);
            }
            L(reduce_loop_end_label);

            // store
            store_dst_vector();

            jmp(reduce_main_end_label, T_NEAR);
        }

        L(reduce_main_end_label);
    }

    inline void reduce_tail() {
        if (jcp_.reduce_mode == Algorithm::ReduceL1) {
            uni_vmovups(xmm_aux, table_val(1));
        }

        Xbyak::Label tail_dst_shifted_label;
        Xbyak::Label tail_dst_fixed_label;
        Xbyak::Label reduce_tail_end_label;
        if (planar_layout) {
            cmp(reg_reduce_w, 1);  // planar layout reducing W
            je(tail_dst_fixed_label, T_NEAR);
        }

        // each src scalar reduce to each dst scalar (X1, X2, X3, ...) -> (Y1, Y2, Y3, ...)
        // cases: [planar layout reducing other dimensions but W] [blocked layout concern padding]
        L(tail_dst_shifted_label);
        {
            reduce_kernel_tail();

            jmp(reduce_tail_end_label, T_NEAR);
        }

        // each src scalar reduce to the same dst scalar (X1, X2, X3, ...) -> (Y1)
        // cases: [planar layout reducing W]
        L(tail_dst_fixed_label);
        {
            // load
            load_scalar(xmm_dst, ptr[reg_dst], jcp_.dst_dt);

            Xbyak::Label reduce_loop_label;
            Xbyak::Label reduce_loop_end_label;

            // reduce
            int step = 1;
            L(reduce_loop_label);
            {
                cmp(reg_work_amount, step);
                jl(reduce_loop_end_label, T_NEAR);

                load_scalar(xmm_src, ptr[reg_src], jcp_.src_dt);

                reduce_kernel_scalar(xmm_src, xmm_dst);
                if (jcp_.reduce_mode == Algorithm::ReduceOr) {
                    uni_cmpneqps(xmm_dst, xmm_dst, xmm_zero);
                    uni_vandps(xmm_dst, xmm_dst, xmm_aux);
                }

                add(reg_src, step * jcp_.src_data_size);
                sub(reg_work_amount, step);

                jmp(reduce_loop_label, T_NEAR);
            }
            L(reduce_loop_end_label);

            // store
            store_scalar(ptr[reg_dst], xmm_dst, jcp_.dst_dt);
        }

        L(reduce_tail_end_label);
    }

    inline void init_reg_reduce_stride() {
        mov(reg_reduce_stride, ptr[reg_params + GET_OFF(reduce_stride)]);
        mul_by_const(reg_reduce_stride, reg_tmp_64, jcp_.src_data_size);
    }

    inline void reduce_kernel() {
        Xbyak::Label reduce_label;
        Xbyak::Label reduce_end_label;
        Xbyak::Label reduce_batch_label;
        Xbyak::Label reduce_batch_end_label;

        int step = vlen / sizeof(float) < 8 ? 8 : vlen / sizeof(float);
        cmp(reg_work_batch, 1);
        je(reduce_label, T_NEAR);

        init_reg_reduce_stride();

        L(reduce_batch_label);
        {
            cmp(reg_work_amount, step);
            jl(reduce_end_label, T_NEAR);

            reduce_batch();

            add(reg_src, step * jcp_.src_data_size);
            sub(reg_work_amount, step);
            jmp(reduce_batch_label, T_NEAR);
        }
        L(reduce_batch_end_label);

        L(reduce_label);
        {
            cmp(reg_work_amount, step);
            jl(reduce_end_label, T_NEAR);

            reduce_once();

            add(reg_src, step * jcp_.src_data_size);
            sub(reg_work_amount, step);
            jmp(reduce_label, T_NEAR);
        }
        L(reduce_end_label);
    }

    inline void reduce_once() {
        load_vector(vmm_src, ptr[reg_src], jcp_.src_dt);
        reduce_kernel(vmm_src, vmm_dst);

        if (isa == cpu::x64::sse41) {
            load_vector(vmm_src, ptr[reg_src + 4 * jcp_.src_data_size], jcp_.src_dt);
            reduce_kernel(vmm_src, vmm_dst_aux);
        }
    }

    inline void reduce_batch() {
        mov(reg_src_aux, reg_src);
        mov(reg_work_batch_aux, reg_work_batch);

        Xbyak::Label reduce_batch_loop_label;
        Xbyak::Label reduce_batch_loop_end_label;
        L(reduce_batch_loop_label);
        {
            cmp(reg_work_batch_aux, 1);
            jl(reduce_batch_loop_end_label, T_NEAR);

            load_vector(vmm_src, ptr[reg_src_aux], jcp_.src_dt);
            reduce_kernel(vmm_src, vmm_dst);
            if (isa == cpu::x64::sse41) {
                load_vector(vmm_src, ptr[reg_src_aux + 4 * jcp_.src_data_size], jcp_.src_dt);
                reduce_kernel(vmm_src, vmm_dst_aux);
            }

            add(reg_src_aux, reg_reduce_stride);
            sub(reg_work_batch_aux, 1);
            jmp(reduce_batch_loop_label, T_NEAR);
        }
        L(reduce_batch_loop_end_label);
    }

    inline void reduce_gather(Vmm vmm_dst, int offset) {
        switch (jcp_.src_dt) {
            case memory::data_type::f32:
            case memory::data_type::s32:
                if (isa == cpu::x64::avx512_core) {
                    kxnord(k_mask, k_mask, k_mask);
                    if (jcp_.src_dt == memory::data_type::f32) {
                        vgatherdps(vmm_src | k_mask, ptr[reg_src + offset + vmm_idx]);
                    } else {
                        vpgatherdd(vmm_src | k_mask, ptr[reg_src + offset + vmm_idx]);
                        uni_vcvtdq2ps(vmm_src, vmm_src);
                    }
                } else if (isa == cpu::x64::avx2) {
                    uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
                    if (jcp_.src_dt == memory::data_type::f32) {
                        vgatherdps(vmm_src, ptr[reg_src + offset + vmm_idx], vmm_mask);
                    } else {
                        vpgatherdd(vmm_src, ptr[reg_src + offset + vmm_idx], vmm_mask);
                        uni_vcvtdq2ps(vmm_src, vmm_src);
                    }
                } else {
                    pack_gathered_vector(vmm_src, vmm_idx, offset, jcp_.src_dt);
                }
                break;
            case memory::data_type::bf16:
            case memory::data_type::s8:
            case memory::data_type::u8:
                pack_gathered_vector(vmm_src, vmm_idx, offset, jcp_.src_dt);
                break;
            default:
                assert(!"unknown src_dt");
        }
        reduce_kernel(vmm_src, vmm_dst);
    }

    inline void pack_gathered_vector(Vmm vmm_val, Vmm vmm_index, int offset, memory::data_type src_dt) {
        sub(rsp, vlen);
        uni_vmovdqu(ptr[rsp], vmm_index);
        int repeats = vlen / sizeof(float);
        for (size_t i = 0; i < repeats; i++) {
            mov(reg_tmp_64.cvt32(), ptr[rsp + i * sizeof(int)]);
            Xbyak::Address table_idx = ptr[reg_src + offset + reg_tmp_64];
            switch (src_dt) {
                case memory::data_type::f32:
                case memory::data_type::s32:
                    mov(reg_tmp_64.cvt32(), table_idx);
                    mov(ptr[rsp + i * sizeof(int)], reg_tmp_64.cvt32());
                    break;
                case memory::data_type::bf16:
                    mov(reg_tmp_64.cvt16(), table_idx);
                    mov(ptr[rsp + i * sizeof(ov::intel_cpu::bfloat16_t)], reg_tmp_64.cvt16());
                    break;
                case memory::data_type::s8:
                case memory::data_type::u8:
                    mov(reg_tmp_64.cvt8(), table_idx);
                    mov(ptr[rsp + i * sizeof(char)], reg_tmp_64.cvt8());
                    break;
                default:
                    assert(!"unknown src_dt");
            }
        }

        switch (src_dt) {
            case memory::data_type::f32:
            case memory::data_type::s32:
                uni_vmovups(vmm_val, ptr[rsp]);
                break;
            case memory::data_type::bf16:
                uni_vpmovzxwd(vmm_val, ptr[rsp]);
                uni_vpslld(vmm_val, vmm_val, 16);
            break;
            case memory::data_type::s8:
                uni_vpmovsxbd(vmm_val, ptr[rsp]);
                break;
            case memory::data_type::u8:
                uni_vpmovzxbd(vmm_val, ptr[rsp]);
                break;
            default:
                assert(!"unknown src_dt");
        }

        if (!isFloatCompatible(src_dt))
            uni_vcvtdq2ps(vmm_val, vmm_val);
        add(rsp, vlen);
    }

    inline void reduce_kernel_tail() {
        Xbyak::Label reduce_label;
        Xbyak::Label reduce_end_label;
        Xbyak::Label reduce_batch_label;
        Xbyak::Label reduce_batch_end_label;

        int step = 1;
        cmp(reg_work_batch, 1);
        je(reduce_label, T_NEAR);

        init_reg_reduce_stride();

        L(reduce_batch_label);
        {
            cmp(reg_work_amount, step);
            jl(reduce_end_label, T_NEAR);

            // load
            load_scalar(xmm_dst, ptr[reg_dst], jcp_.dst_dt);

            // reduce
            reduce_batch_tail();

            // store
            store_scalar(ptr[reg_dst], xmm_dst, jcp_.dst_dt);

            add(reg_dst, step * jcp_.dst_data_size);
            add(reg_src, step * jcp_.src_data_size);
            sub(reg_work_amount, step);

            jmp(reduce_batch_label, T_NEAR);
        }
        L(reduce_batch_end_label);

        L(reduce_label);
        {
            cmp(reg_work_amount, step);
            jl(reduce_end_label, T_NEAR);

            // load
            load_scalar(xmm_dst, ptr[reg_dst], jcp_.dst_dt);

            // reduce
            reduce_batch_tail();

            // store
            store_scalar(ptr[reg_dst], xmm_dst, jcp_.dst_dt);

            add(reg_dst, step * jcp_.dst_data_size);
            add(reg_src, step * jcp_.src_data_size);
            sub(reg_work_amount, step);

            jmp(reduce_label, T_NEAR);
        }
        L(reduce_end_label);
    }

    inline void reduce_once_tail() {
        load_scalar(xmm_src, ptr[reg_src], jcp_.src_dt);
        reduce_kernel_scalar(xmm_src, xmm_dst);
        if (jcp_.reduce_mode == Algorithm::ReduceOr) {
            uni_cmpneqps(xmm_dst, xmm_dst, xmm_zero);
            uni_vandps(xmm_dst, xmm_dst, xmm_aux);
        }
    }

    inline void reduce_batch_tail() {
        mov(reg_src_aux, reg_src);
        mov(reg_work_batch_aux, reg_work_batch);

        Xbyak::Label reduce_batch_loop_label;
        Xbyak::Label reduce_batch_loop_end_label;
        L(reduce_batch_loop_label);
        {
            cmp(reg_work_batch_aux, 1);
            jl(reduce_batch_loop_end_label, T_NEAR);

            load_scalar(xmm_src, ptr[reg_src_aux], jcp_.src_dt);
            reduce_kernel_scalar(xmm_src, xmm_dst);
            if (jcp_.reduce_mode == Algorithm::ReduceOr) {
                uni_cmpneqps(xmm_dst, xmm_dst, xmm_zero);
                uni_vandps(xmm_dst, xmm_dst, xmm_aux);
            }

            add(reg_src_aux, reg_reduce_stride);
            sub(reg_work_batch_aux, 1);
            jmp(reduce_batch_loop_label, T_NEAR);
        }
        L(reduce_batch_loop_end_label);
    }

    inline void reduce_main_loop() {
        Xbyak::Label reduce_loop_label;
        Xbyak::Label reduce_loop_end_label;

        int step = vlen / sizeof(float) < 8 ? 8 : vlen / sizeof(float);
        L(reduce_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(reduce_loop_end_label, T_NEAR);

            load_vector(vmm_src, ptr[reg_src], jcp_.src_dt);
            reduce_kernel(vmm_src, vmm_dst);

            if (isa == cpu::x64::sse41) {
                load_vector(vmm_src, ptr[reg_src + 4 * jcp_.src_data_size], jcp_.src_dt);
                reduce_kernel(vmm_src, vmm_dst);
            }

            add(reg_src, step * jcp_.src_data_size);
            sub(reg_work_amount, step);

            jmp(reduce_loop_label, T_NEAR);
        }
        L(reduce_loop_end_label);
    }

    inline void reduce_kernel(Vmm vmm_src, Vmm vmm_dst) {
        switch (jcp_.reduce_mode) {
            case Algorithm::ReduceAnd:
                if (isa == cpu::x64::avx512_core) {
                    vcmpps(k_mask, vmm_src, vmm_zero, _cmp_neq_uq);
                    vblendmps(vmm_src | k_mask, vmm_zero, vmm_aux);
                } else {
                    uni_cmpneqps(vmm_src, vmm_src, vmm_zero);
                }
                uni_vandps(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceL1:
                uni_vandps(vmm_src, vmm_src, vmm_aux);
                uni_vaddps(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceLogSum:
            case Algorithm::ReduceMean:
            case Algorithm::ReduceSum:
                uni_vaddps(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceMax:
                uni_vmaxps(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceMin:
                uni_vminps(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceL2:
            case Algorithm::ReduceSumSquare:
                uni_vmulps(vmm_src, vmm_src, vmm_src);
                uni_vaddps(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceLogSumExp:
                exp_injector->compute_vector_range(vmm_src.getIdx(), vmm_src.getIdx() + 1);
                uni_vaddps(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceOr:
                if (isa == cpu::x64::avx512_core) {
                    vcmpps(k_mask, vmm_src, vmm_zero, _cmp_neq_uq);
                    vblendmps(vmm_src | k_mask, vmm_zero, vmm_aux);
                }
                uni_vorps(vmm_dst, vmm_dst, vmm_src);
                break;
            case Algorithm::ReduceProd:
                uni_vmulps(vmm_dst, vmm_dst, vmm_src);
                break;
            default:
                assert(!"unsupported reduce mode");
        }
    }

    inline void reduce_kernel_scalar(Xmm xmm_src, Xmm xmm_dst) {
        switch (jcp_.reduce_mode) {
            case Algorithm::ReduceAnd:
                uni_cmpneqps(xmm_src, xmm_src, xmm_zero);
                uni_vandps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceL1:
                uni_vandps(xmm_src, xmm_src, xmm_aux);
                uni_vaddps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceLogSum:
            case Algorithm::ReduceMean:
            case Algorithm::ReduceSum:
                uni_vaddps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceMax:
                uni_vmaxps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceMin:
                uni_vminps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceL2:
            case Algorithm::ReduceSumSquare:
                uni_vmulps(xmm_src, xmm_src, xmm_src);
                uni_vaddps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceLogSumExp:
                exp_injector->compute_vector_range(xmm_src.getIdx(), xmm_src.getIdx() + 1);
                uni_vaddps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceOr:
                uni_vorps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Algorithm::ReduceProd:
                uni_vmulps(xmm_dst, xmm_dst, xmm_src);
                break;
            default:
                assert(!"unsupported reduce mode");
        }
    }

    inline void load_dst_vector() {
        load_vector(vmm_dst, ptr[reg_dst], jcp_.dst_dt);
        if (isa == cpu::x64::sse41)
            load_vector(vmm_dst_aux, ptr[reg_dst + 4 * jcp_.dst_data_size], jcp_.dst_dt);
    }

    inline void store_dst_vector() {
        if (jcp_.reduce_mode == Algorithm::ReduceOr && isa != cpu::x64::avx512_core) {
            uni_cmpneqps(vmm_dst, vmm_dst, vmm_zero);
            uni_vandps(vmm_dst, vmm_dst, vmm_aux);

            if (isa == cpu::x64::sse41) {
                uni_cmpneqps(vmm_dst_aux, vmm_dst_aux, vmm_zero);
                uni_vandps(vmm_dst_aux, vmm_dst_aux, vmm_aux);
            }
        }
        store_vector(ptr[reg_dst], vmm_dst, jcp_.dst_dt);
        if (isa == cpu::x64::sse41)
            store_vector(ptr[reg_dst + 4 * jcp_.dst_data_size], vmm_dst_aux, jcp_.dst_dt);
    }

    inline void load_vector(Vmm vmm_src, const Xbyak::Address &op, memory::data_type src_dt) {
        switch (src_dt) {
            case memory::data_type::f32:
            case memory::data_type::s32:
                uni_vmovups(vmm_src, op);
                break;
            case memory::data_type::bf16:
                uni_vpmovzxwd(vmm_src, op);
                uni_vpslld(vmm_src, vmm_src, 16);
                break;
            case memory::data_type::s8:
                uni_vpmovsxbd(vmm_src, op);
                break;
            case memory::data_type::u8:
                uni_vpmovzxbd(vmm_src, op);
                break;
            default:
                assert(!"unknown src_dt");
        }

        if (!isFloatCompatible(src_dt))
            uni_vcvtdq2ps(vmm_src, vmm_src);
    }

    inline void load_scalar(Xmm xmm_src, const Xbyak::Address &op, memory::data_type src_dt) {
        switch (src_dt) {
            case memory::data_type::f32:
            case memory::data_type::s32:
                uni_vmovss(xmm_src, op);
                break;
            case memory::data_type::bf16:
                uni_vpinsrw(xmm_src, xmm_src, op, 0x0);
                uni_vpslld(xmm_src, xmm_src, 16);
                break;
            case memory::data_type::s8:
                movsx(reg_tmp_32, op);
                uni_vmovq(xmm_src, reg_tmp_64);
                break;
            case memory::data_type::u8:
                movzx(reg_tmp_32, op);
                uni_vmovq(xmm_src, reg_tmp_64);
                break;
            default:
                assert(!"unknown src_dt");
        }

        if (!isFloatCompatible(src_dt)) {
            uni_vcvtdq2ps(xmm_src, xmm_src);
        }
    }

    inline void store_vector(const Xbyak::Address &op, Vmm vmm_dst, memory::data_type dst_dt) {
        Xmm xmm_dst = Xmm(vmm_dst.getIdx());
        Ymm ymm_dst = Ymm(vmm_dst.getIdx());

        if (!isFloatCompatible(dst_dt)) {
            uni_vcvtps2dq(vmm_dst, vmm_dst);
        }

        switch (dst_dt) {
            case memory::data_type::f32:
            case memory::data_type::s32:
                uni_vmovups(op, vmm_dst);
                break;
            case memory::data_type::bf16:
                uni_vcvtneps2bf16->emit_code({static_cast<size_t>(vmm_dst.getIdx())}, {static_cast<size_t>(ymm_dst.getIdx())});
                vmovdqu16(op, ymm_dst);
                break;
            case memory::data_type::s8:
                if (isa == cpu::x64::avx512_core) {
                    vpmovsdb(op, vmm_dst);
                } else {
                    uni_vpackssdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != cpu::x64::sse41)
                        vpermq(ymm_dst, ymm_dst, 0x08);
                    uni_vpacksswb(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != cpu::x64::sse41)
                        vmovq(op, xmm_dst);
                    else
                        uni_vmovd(op, xmm_dst);
                }
                break;
            case memory::data_type::u8:
                if (isa == cpu::x64::avx512_core) {
                    vpmaxsd(vmm_dst, vmm_zero, vmm_dst);
                    vpmovusdb(op, vmm_dst);
                } else {
                    uni_vpackusdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != cpu::x64::sse41)
                        vpermq(ymm_dst, ymm_dst, 0x08);
                    uni_vpackuswb(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != cpu::x64::sse41)
                        vmovq(op, xmm_dst);
                    else
                        uni_vmovd(op, xmm_dst);
                }
                break;
            default:
                assert(!"unknown dst_dt");
        }
    }

    inline void store_scalar(const Xbyak::Address &op, Xmm xmm_dst, memory::data_type dst_dt) {
        if (!isFloatCompatible(dst_dt)) {
            uni_vcvtps2dq(xmm_dst, xmm_dst);
        }

        switch (dst_dt) {
            case memory::data_type::f32:
            case memory::data_type::s32:
                uni_vmovss(op, xmm_dst);
                break;
            case memory::data_type::bf16:
                uni_vpsrld(xmm_dst, xmm_dst, 16);
                uni_vpextrw(op, xmm_dst, 0x0);
                break;
            case memory::data_type::s8:
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
                uni_vmovq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            case memory::data_type::u8:
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
                uni_vmovq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            default:
                assert(!"unknown dst_dt");
        }
    }

    inline void horiz_reduce_store(Vmm vmm_dst, memory::data_type dst_dt, bool load_embedded = false) {
        if (isa == cpu::x64::sse41) {
            horiz_store(vmm_dst, dst_dt, load_embedded);
        } else if (isa == cpu::x64::avx2) {
            Xbyak::Ymm ymm_dst = Xbyak::Ymm(vmm_dst.getIdx());
            vextractf128(xmm_aux1, ymm_dst, 0);
            vextractf128(xmm_aux2, ymm_dst, 1);
            horiz_ps(xmm_aux1, xmm_aux2);
            horiz_store(xmm_aux1, dst_dt, load_embedded);
        } else {
            Xbyak::Zmm zmm_dst = Xbyak::Zmm(vmm_dst.getIdx());
            vextractf32x4(xmm_aux1, zmm_dst, 0);
            vextractf32x4(xmm_aux2, zmm_dst, 1);
            horiz_ps(xmm_aux1, xmm_aux2);
            vextractf32x4(xmm_aux2, zmm_dst, 2);
            vextractf32x4(xmm_aux3, zmm_dst, 3);
            horiz_ps(xmm_aux2, xmm_aux3);
            horiz_ps(xmm_aux1, xmm_aux2);
            horiz_store(xmm_aux1, dst_dt, load_embedded);
        }
    }

    inline void horiz_store(Xbyak::Xmm xmm_dst, memory::data_type dst_dt, bool load_embedded) {
        uni_vmovshdup(xmm_aux3, xmm_dst);          // dst:1,2,3,4; aux3:2,2,4,4
        horiz_ps(xmm_dst, xmm_aux3);               // dst:f(1,2),f(2,2),f(3,4),f(4,4)
        uni_vmovhlps(xmm_aux3, xmm_aux3, xmm_dst); // aux3:f(3,4),f(4,4),4,4
        horiz_ps(xmm_dst, xmm_aux3);               // dst:f(1,2,3,4),...
        if (load_embedded) {
            load_scalar(xmm_aux3, ptr[reg_dst], dst_dt);
            horiz_ps(xmm_dst, xmm_aux3);
        }
        store_scalar(ptr[reg_dst], xmm_dst, dst_dt);
    }

    inline void horiz_ps(const Xmm& xmm, const Operand& op) {
        switch (jcp_.reduce_mode) {
            case Algorithm::ReduceAnd:
                uni_vandps(xmm, xmm, op);
                break;
            case Algorithm::ReduceL1:
            case Algorithm::ReduceL2:
            case Algorithm::ReduceLogSum:
            case Algorithm::ReduceMean:
            case Algorithm::ReduceSum:
            case Algorithm::ReduceSumSquare:
            case Algorithm::ReduceLogSumExp:
                uni_vaddps(xmm, xmm, op);
                break;
            case Algorithm::ReduceMax:
                uni_vmaxps(xmm, xmm, op);
                break;
            case Algorithm::ReduceMin:
                uni_vminps(xmm, xmm, op);
                break;
            case Algorithm::ReduceOr:
                uni_vorps(xmm, xmm, op);
                break;
            case Algorithm::ReduceProd:
                uni_vmulps(xmm, xmm, op);
                break;
            default:
                assert(!"unsupported reduce mode");
        }
    }

    void prepare_aux_table() {
        auto broadcast_int = [&](int val) {
            for (size_t d = 0; d < vlen / sizeof(float); ++d) {
                dd(val);
            }
        };

        align(64);
        L(l_table);

        broadcast_int(aux_vals.float_one);
        broadcast_int(aux_vals.float_abs);
        broadcast_int(aux_vals.float_min);
        broadcast_int(aux_vals.float_max);
        broadcast_int(aux_vals.int32_min);
        broadcast_int(aux_vals.int32_max);
    }

    const struct aux_vals_type {
        int float_one = 0x3f800000; // 1.0f
        int float_abs = 0x7fffffff; // mask to make positive
        int float_min = 0xff7fffff; // float minimum
        int float_max = 0x7f7fffff; // float maximum
        int int32_min = 0xcf000000; // -2^31 presented in float
        int int32_max = 0x4effffff; // 2^31-1 presented in float
    } aux_vals;
};

template <cpu_isa_t isa>
struct jit_uni_reduce_post_kernel_f32 : public jit_uni_reduce_post_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_reduce_post_kernel_f32)

    explicit jit_uni_reduce_post_kernel_f32(jit_reduce_config_params jcp, const dnnl_primitive_attr &attr)
    : jit_uni_reduce_post_kernel(jcp, attr), jit_generator(jit_name()) {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        const auto &p = attr_.post_ops_;
        for (int i = 0; i < p.len(); i++) {
            auto &post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors.push_back(std::make_shared<jit_uni_eltwise_injector_f32<isa>>(
                        this, post_op.eltwise.alg, post_op.eltwise.alpha, post_op.eltwise.beta, post_op.eltwise.scale));
            } else if (post_op.is_depthwise()) {
                depthwise_injectors.push_back(std::make_shared<jit_uni_depthwise_injector_f32<isa>>(
                        this, post_op));
            } else if (post_op.is_quantization()) {
                quantization_injectors.push_back(std::make_shared<jit_uni_quantization_injector_f32<isa>>(
                        this, post_op, vmm_d_weights, vmm_d_bias, reg_d_weights, reg_d_bias));
            }
        }

        if (jcp_.reduce_mode == Algorithm::ReduceLogSum || jcp_.reduce_mode == Algorithm::ReduceLogSumExp) {
            log_injector = std::make_shared<jit_uni_eltwise_injector_f32<isa>>(this, alg_kind::eltwise_log, 0.f, 0.f, 1.f);
        }

        if (mayiuse(avx512_core))
            uni_vcvtneps2bf16 = std::make_shared<jit_uni_vcvtneps2bf16>(this, isa);

        this->preamble();

        planar_layout = jcp_.layout == ReduceLayoutType::reduce_ncsp || jcp_.layout == ReduceLayoutType::reduce_nspc;

        mov(reg_dst, ptr[reg_params + GET_OFF_POST(dst)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF_POST(work_amount)]);
        mov(reg_channel_size, ptr[reg_params + GET_OFF_POST(channel_size)]);
        mov(reg_divisor, ptr[reg_params + GET_OFF_POST(divisor)]);
        if (!planar_layout)
            mov(reg_reduce_c, ptr[reg_params + GET_OFF_POST(reduce_c)]);
        if (attr_.post_ops_.len() != 0) {
            mov(reg_post_ops_data, ptr[reg_params + GET_OFF_POST(post_op_data)]);
            mov(reg_oc_off, ptr[reg_params + GET_OFF_POST(oc_off)]);
        }

        if (isa == cpu::x64::avx512_core)
            uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        if (jcp_.layout == ReduceLayoutType::reduce_blocked) {
            reduce_post_main();
        } else if (jcp_.layout == ReduceLayoutType::reduce_nspc && attr_.post_ops_.len() != 0) {
            // the tail of channel dimension should always be concerned during post ops fusing for nspc layout
            Xbyak::Label reduce_nspc_loop_label;
            Xbyak::Label reduce_nspc_loop_end_label;
            mov(reg_total_work_amount, reg_work_amount);
            L(reduce_nspc_loop_label);
            {
                cmp(reg_total_work_amount, 0);
                jle(reduce_nspc_loop_end_label, T_NEAR);

                mov(reg_oc_off, 0);
                mov(reg_work_amount, reg_channel_size);
                reduce_post_main();
                reduce_post_tail();

                sub(reg_total_work_amount, reg_channel_size);
                jmp(reduce_nspc_loop_label, T_NEAR);
            }
            L(reduce_nspc_loop_end_label);
        } else {
            reduce_post_main();
            reduce_post_tail();
        }

        this->postamble();

        if (mayiuse(avx512_core))
            uni_vcvtneps2bf16->emit_data();

        if (jcp_.reduce_mode == Algorithm::ReduceLogSum || jcp_.reduce_mode == Algorithm::ReduceLogSumExp) {
            log_injector->prepare_table();
        }

        for (auto& inj : eltwise_injectors)
            inj->prepare_table();
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;
    size_t vlen = cpu_isa_traits<isa>::vlen;
    bool planar_layout = false;

    Xbyak::Reg64 reg_dst = r8;
    Xbyak::Reg64 reg_work_amount = r9;
    Xbyak::Reg64 reg_total_work_amount = r10;
    Xbyak::Reg64 reg_channel_size = r11;
    Xbyak::Reg64 reg_divisor = r12;
    Xbyak::Reg64 reg_reduce_c = r13;
    Xbyak::Reg64 reg_params = abi_param1;

    Xbyak::Reg8 reg_tmp_8 = r14b;
    Xbyak::Reg32 reg_tmp_32 = r14d;
    Xbyak::Reg64 reg_tmp_64 = r14;

    Xbyak::Reg64 reg_oc_off = rax;
    Xbyak::Reg64 reg_d_weights = rbx;
    Xbyak::Reg64 reg_d_bias = rdx;
    Xbyak::Reg64 reg_post_ops_data = r15;

    Vmm vmm_aux = Vmm(0);
    Xmm xmm_aux = Xmm(0);
    Vmm vmm_dst = Vmm(1);
    Xmm xmm_dst = Xmm(1);
    Vmm vmm_zero = Vmm(2);
    Vmm vmm_dst_aux = Vmm(3);
    Xbyak::Xmm xmm_aux1 = Xbyak::Xmm(4);
    Xbyak::Xmm xmm_aux2 = Xbyak::Xmm(5);
    Xbyak::Xmm xmm_aux3 = Xbyak::Xmm(6);

    Vmm vmm_d_weights = Vmm(7);
    Vmm vmm_d_bias = Vmm(8);

    std::shared_ptr<jit_uni_vcvtneps2bf16> uni_vcvtneps2bf16;
    std::shared_ptr<jit_uni_eltwise_injector_f32<isa>> log_injector;

    std::vector<std::shared_ptr<jit_uni_eltwise_injector_f32<isa>>> eltwise_injectors;
    std::vector<std::shared_ptr<jit_uni_depthwise_injector_f32<isa>>> depthwise_injectors;
    std::vector<std::shared_ptr<jit_uni_quantization_injector_f32<isa>>> quantization_injectors;

    inline void reduce_post_main() {
        Xbyak::Label reduce_channel_label;
        Xbyak::Label reduce_map_label;
        if (planar_layout) {
            jmp(reduce_map_label, T_NEAR);
        } else {
            cmp(reg_reduce_c, 1);
            jne(reduce_map_label, T_NEAR);
        }

        // further reduce channel block since reduce channel batch has already been reduced
        // (X1, X2, X3, X4, X5, X6, X7, X8) -> (Y1, N/A, N/A, N/A, N/A, N/A, N/A, N/A)
        // cases: [blocked layout reducing channel dimensions]
        L(reduce_channel_label);
        {
            Xbyak::Label reduce_loop_label;
            Xbyak::Label reduce_loop_end_label;

            int step = vlen / sizeof(float) < 8 ? 8 : vlen / sizeof(float);
            L(reduce_loop_label);
            {
                cmp(reg_work_amount, step);
                jl(reduce_loop_end_label, T_NEAR);

                // load
                load_vector(vmm_dst, ptr[reg_dst], jcp_.dst_dt);
                if (isa == cpu::x64::sse41)
                    load_vector(vmm_dst_aux, ptr[reg_dst + 4 * jcp_.dst_data_size], jcp_.dst_dt);

                // reduce and store
                horiz_reduce_store(vmm_dst, jcp_.dst_dt);
                if (isa == cpu::x64::sse41)
                    horiz_reduce_store(vmm_dst_aux, jcp_.dst_dt, true);

                add(reg_dst, step * jcp_.dst_data_size);
                sub(reg_work_amount, step);

                jmp(reduce_loop_label, T_NEAR);
            }
            L(reduce_loop_end_label);

            mov(reg_dst, ptr[reg_params + GET_OFF_POST(dst)]);
            mov(reg_work_amount, ptr[reg_params + GET_OFF_POST(work_amount)]);
        }

        // reduce map for value in dst memory
        // cases: [ReduceL2] [ReduceLogSum] [ReduceLogSumExp] [ReduceMean]
        L(reduce_map_label);
        {
            if (jcp_.reduce_mode == Algorithm::ReduceL2 || jcp_.reduce_mode == Algorithm::ReduceMean ||
                jcp_.reduce_mode == Algorithm::ReduceLogSum || jcp_.reduce_mode == Algorithm::ReduceLogSumExp) {
                if (jcp_.reduce_mode == Algorithm::ReduceMean)
                    uni_vbroadcastss(vmm_aux, ptr[reg_divisor]);

                Xbyak::Label reduce_loop_label;
                Xbyak::Label reduce_loop_end_label;

                int step = vlen / sizeof(float) < 8 ? 8 : vlen / sizeof(float);
                L(reduce_loop_label);
                {
                    cmp(reg_work_amount, step);
                    jl(reduce_loop_end_label, T_NEAR);

                    load_vector(vmm_dst, ptr[reg_dst], jcp_.dst_dt);
                    reduce_map_kernel(vmm_dst);
                    if (attr_.post_ops_.len() != 0)
                        apply_post_ops(jcp_.dst_dt, jcp_.layout == ReduceLayoutType::reduce_ncsp);
                    store_vector(ptr[reg_dst], vmm_dst, jcp_.dst_dt);

                    if (isa == cpu::x64::sse41) {
                        load_vector(vmm_dst, ptr[reg_dst + 4 * jcp_.dst_data_size], jcp_.dst_dt);
                        reduce_map_kernel(vmm_dst);
                        if (attr_.post_ops_.len() != 0) {
                            if (jcp_.layout != ReduceLayoutType::reduce_ncsp)
                                add(reg_oc_off, 4 * sizeof(float));
                            apply_post_ops(jcp_.dst_dt, jcp_.layout == ReduceLayoutType::reduce_ncsp);
                            if (jcp_.layout != ReduceLayoutType::reduce_ncsp)
                                sub(reg_oc_off, 4 * sizeof(float));
                        }
                        store_vector(ptr[reg_dst + 4 * jcp_.dst_data_size], vmm_dst, jcp_.dst_dt);
                    }

                    add(reg_dst, step * jcp_.dst_data_size);
                    if (jcp_.layout == ReduceLayoutType::reduce_nspc && attr_.post_ops_.len() != 0)
                        add(reg_oc_off, step * sizeof(float));
                    sub(reg_work_amount, step);

                    jmp(reduce_loop_label, T_NEAR);
                }
                L(reduce_loop_end_label);
            } else {
                if (attr_.post_ops_.len() != 0) {
                    Xbyak::Label reduce_loop_label;
                    Xbyak::Label reduce_loop_end_label;

                    int step = vlen / sizeof(float) < 8 ? 8 : vlen / sizeof(float);
                    L(reduce_loop_label);
                    {
                        cmp(reg_work_amount, step);
                        jl(reduce_loop_end_label, T_NEAR);

                        load_vector(vmm_dst, ptr[reg_dst], jcp_.dst_dt);
                        apply_post_ops(jcp_.dst_dt, jcp_.layout == ReduceLayoutType::reduce_ncsp);
                        store_vector(ptr[reg_dst], vmm_dst, jcp_.dst_dt);

                        if (isa == cpu::x64::sse41) {
                            load_vector(vmm_dst, ptr[reg_dst + 4 * jcp_.dst_data_size], jcp_.dst_dt);
                            if (jcp_.layout != ReduceLayoutType::reduce_ncsp)
                                add(reg_oc_off, 4 * sizeof(float));
                            apply_post_ops(jcp_.dst_dt, jcp_.layout == ReduceLayoutType::reduce_ncsp);
                            if (jcp_.layout != ReduceLayoutType::reduce_ncsp)
                                sub(reg_oc_off, 4 * sizeof(float));
                            store_vector(ptr[reg_dst + 4 * jcp_.dst_data_size], vmm_dst, jcp_.dst_dt);
                        }

                        add(reg_dst, step * jcp_.dst_data_size);
                        if (jcp_.layout == ReduceLayoutType::reduce_nspc && attr_.post_ops_.len() != 0)
                            add(reg_oc_off, step * sizeof(float));
                        sub(reg_work_amount, step);

                        jmp(reduce_loop_label, T_NEAR);
                    }
                    L(reduce_loop_end_label);
                }
            }
        }
    }

    inline void reduce_post_tail() {
        // reduce map for tail in dst memory
        // cases: [ReduceL2] [ReduceLogSum] [ReduceLogSumExp] [ReduceMean] in planar layout
        if (jcp_.reduce_mode == Algorithm::ReduceL2 || jcp_.reduce_mode == Algorithm::ReduceMean ||
                jcp_.reduce_mode == Algorithm::ReduceLogSum || jcp_.reduce_mode == Algorithm::ReduceLogSumExp) {
            if (jcp_.reduce_mode == Algorithm::ReduceMean)
                uni_vbroadcastss(xmm_aux, ptr[reg_divisor]);

            Xbyak::Label reduce_loop_label;
            Xbyak::Label reduce_loop_end_label;

            int step = 1;
            L(reduce_loop_label);
            {
                cmp(reg_work_amount, step);
                jl(reduce_loop_end_label, T_NEAR);

                // load
                load_scalar(xmm_dst, ptr[reg_dst], jcp_.dst_dt);

                // reduce
                reduce_map_kernel_scalar(xmm_dst);

                // store
                if (attr_.post_ops_.len() != 0)
                    apply_post_ops(jcp_.dst_dt, jcp_.layout == ReduceLayoutType::reduce_ncsp);
                store_scalar(ptr[reg_dst], xmm_dst, jcp_.dst_dt);

                add(reg_dst, step * jcp_.dst_data_size);
                if (jcp_.layout == ReduceLayoutType::reduce_nspc && attr_.post_ops_.len() != 0)
                    add(reg_oc_off, step * sizeof(float));
                sub(reg_work_amount, step);

                jmp(reduce_loop_label, T_NEAR);
            }
            L(reduce_loop_end_label);
        } else {
            if (attr_.post_ops_.len() != 0) {
                Xbyak::Label reduce_loop_label;
                Xbyak::Label reduce_loop_end_label;

                int step = 1;
                L(reduce_loop_label);
                {
                    cmp(reg_work_amount, step);
                    jl(reduce_loop_end_label, T_NEAR);

                    // load
                    load_scalar(xmm_dst, ptr[reg_dst], jcp_.dst_dt);

                    // store
                    apply_post_ops(jcp_.dst_dt, jcp_.layout == ReduceLayoutType::reduce_ncsp);
                    store_scalar(ptr[reg_dst], xmm_dst, jcp_.dst_dt);

                    add(reg_dst, step * jcp_.dst_data_size);
                    if (jcp_.layout == ReduceLayoutType::reduce_nspc && attr_.post_ops_.len() != 0)
                        add(reg_oc_off, step * sizeof(float));
                    sub(reg_work_amount, step);

                    jmp(reduce_loop_label, T_NEAR);
                }
                L(reduce_loop_end_label);
            }
        }
    }

    void apply_post_ops(memory::data_type dst_dt, bool is_broadcast) {
        const auto &p = attr_.post_ops_;
        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        int quantization_inj_idx = 0;
        int post_ops_data_offset = 0;
        for (int i = 0; i < p.len(); i++) {
            auto& post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors[eltwise_inj_idx]->compute_vector_range(vmm_dst.getIdx(), vmm_dst.getIdx() + 1);
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                mov(reg_d_weights, ptr[reg_post_ops_data + post_ops_data_offset]);
                add(reg_d_weights, reg_oc_off);

                depthwise_injectors[depthwise_inj_idx]->compute_vector_range(
                        vmm_dst.getIdx(), vmm_dst.getIdx() + 1, reg_d_weights, reg_d_weights, is_broadcast);

                post_ops_data_offset += depthwise_injectors[depthwise_inj_idx]->memoryStep();
                depthwise_inj_idx++;
            } else if (post_op.is_quantization()) {
                bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
                bool do_rounding = do_dequantization || isFloatCompatible(dst_dt) || i != p.len() - 1;

                int s_idx = vmm_dst.getIdx();

                quantization_injectors[quantization_inj_idx]->init_crop_ptrs(reg_post_ops_data + post_ops_data_offset, reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_crop(s_idx, s_idx + 1, 0, 0, is_broadcast);

                quantization_injectors[quantization_inj_idx]->init_input_scale_shift_ptrs(reg_post_ops_data + post_ops_data_offset, reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_input_scale_shift(s_idx, s_idx + 1, 0, do_rounding, 0, is_broadcast);

                if (do_dequantization) {
                    quantization_injectors[quantization_inj_idx]->init_output_scale_shift_ptrs(reg_post_ops_data + post_ops_data_offset, reg_oc_off);
                    quantization_injectors[quantization_inj_idx]->compute_output_scale_shift(s_idx, s_idx + 1, 0, 0, is_broadcast);
                }

                post_ops_data_offset += quantization_injectors[quantization_inj_idx]->memoryStep();
                quantization_inj_idx++;
            }
        }
    }

    inline void reduce_map_kernel(Vmm vmm_dst) {
        if (jcp_.reduce_mode == Algorithm::ReduceMean)
            uni_vdivps(vmm_dst, vmm_dst, vmm_aux);
        else if (jcp_.reduce_mode == Algorithm::ReduceL2)
            uni_vsqrtps(vmm_dst, vmm_dst);
        else if (jcp_.reduce_mode == Algorithm::ReduceLogSum || jcp_.reduce_mode == Algorithm::ReduceLogSumExp)
            log_injector->compute_vector_range(vmm_dst.getIdx(), vmm_dst.getIdx() + 1);
    }

    inline void reduce_map_kernel_scalar(Xmm xmm_dst) {
        if (jcp_.reduce_mode == Algorithm::ReduceMean)
            uni_vdivps(xmm_dst, xmm_dst, xmm_aux);
        else if (jcp_.reduce_mode == Algorithm::ReduceL2)
            uni_vsqrtps(xmm_dst, xmm_dst);
        else if (jcp_.reduce_mode == Algorithm::ReduceLogSum || jcp_.reduce_mode == Algorithm::ReduceLogSumExp)
            log_injector->compute_vector_range(xmm_dst.getIdx(), xmm_dst.getIdx() + 1);
    }

    inline void load_vector(Vmm vmm_src, const Xbyak::Address &op, memory::data_type src_dt) {
        switch (src_dt) {
            case memory::data_type::f32:
            case memory::data_type::s32:
                uni_vmovups(vmm_src, op);
                break;
            case memory::data_type::bf16:
                uni_vpmovzxwd(vmm_src, op);
                uni_vpslld(vmm_src, vmm_src, 16);
                break;
            case memory::data_type::s8:
                uni_vpmovsxbd(vmm_src, op);
                break;
            case memory::data_type::u8:
                uni_vpmovzxbd(vmm_src, op);
                break;
            default:
                assert(!"unknown src_dt");
        }

        if (!isFloatCompatible(src_dt))
            uni_vcvtdq2ps(vmm_src, vmm_src);
    }

    inline void load_scalar(Xmm xmm_src, const Xbyak::Address &op, memory::data_type src_dt) {
        switch (src_dt) {
            case memory::data_type::f32:
            case memory::data_type::s32:
                uni_vmovss(xmm_src, op);
                break;
            case memory::data_type::bf16:
                uni_vpinsrw(xmm_src, xmm_src, op, 0x0);
                uni_vpslld(xmm_src, xmm_src, 16);
                break;
            case memory::data_type::s8:
                movsx(reg_tmp_32, op);
                uni_vmovq(xmm_src, reg_tmp_64);
                break;
            case memory::data_type::u8:
                movzx(reg_tmp_32, op);
                uni_vmovq(xmm_src, reg_tmp_64);
                break;
            default:
                assert(!"unknown src_dt");
        }

        if (!isFloatCompatible(src_dt)) {
            uni_vcvtdq2ps(xmm_src, xmm_src);
        }
    }

    inline void store_vector(const Xbyak::Address &op, Vmm vmm_dst, memory::data_type dst_dt) {
        Xmm xmm_dst = Xmm(vmm_dst.getIdx());
        Ymm ymm_dst = Ymm(vmm_dst.getIdx());

        if (!isFloatCompatible(dst_dt)) {
            uni_vcvtps2dq(vmm_dst, vmm_dst);
        }

        switch (dst_dt) {
            case memory::data_type::f32:
            case memory::data_type::s32:
                uni_vmovups(op, vmm_dst);
                break;
            case memory::data_type::bf16:
                uni_vcvtneps2bf16->emit_code({static_cast<size_t>(vmm_dst.getIdx())}, {static_cast<size_t>(ymm_dst.getIdx())});
                vmovdqu16(op, ymm_dst);
                break;
            case memory::data_type::s8:
                if (isa == cpu::x64::avx512_core) {
                    vpmovsdb(op, vmm_dst);
                } else {
                    uni_vpackssdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != cpu::x64::sse41)
                        vpermq(ymm_dst, ymm_dst, 0x08);
                    uni_vpacksswb(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != cpu::x64::sse41)
                        vmovq(op, xmm_dst);
                    else
                        uni_vmovd(op, xmm_dst);
                }
                break;
            case memory::data_type::u8:
                if (isa == cpu::x64::avx512_core) {
                    vpmaxsd(vmm_dst, vmm_zero, vmm_dst);
                    vpmovusdb(op, vmm_dst);
                } else {
                    uni_vpackusdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != cpu::x64::sse41)
                        vpermq(ymm_dst, ymm_dst, 0x08);
                    uni_vpackuswb(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != cpu::x64::sse41)
                        vmovq(op, xmm_dst);
                    else
                        uni_vmovd(op, xmm_dst);
                }
                break;
            default:
                assert(!"unknown dst_dt");
        }
    }

    inline void store_scalar(const Xbyak::Address &op, Xmm xmm_dst, memory::data_type dst_dt) {
        if (!isFloatCompatible(dst_dt)) {
            uni_vcvtps2dq(xmm_dst, xmm_dst);
        }

        switch (dst_dt) {
            case memory::data_type::f32:
            case memory::data_type::s32:
                uni_vmovss(op, xmm_dst);
                break;
            case memory::data_type::bf16:
                uni_vpsrld(xmm_dst, xmm_dst, 16);
                uni_vpextrw(op, xmm_dst, 0x0);
                break;
            case memory::data_type::s8:
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
                uni_vmovq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            case memory::data_type::u8:
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
                uni_vmovq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            default:
                assert(!"unknown dst_dt");
        }
    }

    inline void horiz_reduce_store(Vmm vmm_dst, memory::data_type dst_dt, bool load_embedded = false) {
        if (isa == cpu::x64::sse41) {
            horiz_store(vmm_dst, dst_dt, load_embedded);
        } else if (isa == cpu::x64::avx2) {
            Xbyak::Ymm ymm_dst = Xbyak::Ymm(vmm_dst.getIdx());
            vextractf128(xmm_aux1, ymm_dst, 0);
            vextractf128(xmm_aux2, ymm_dst, 1);
            horiz_ps(xmm_aux1, xmm_aux2);
            horiz_store(xmm_aux1, dst_dt, load_embedded);
        } else {
            Xbyak::Zmm zmm_dst = Xbyak::Zmm(vmm_dst.getIdx());
            vextractf32x4(xmm_aux1, zmm_dst, 0);
            vextractf32x4(xmm_aux2, zmm_dst, 1);
            horiz_ps(xmm_aux1, xmm_aux2);
            vextractf32x4(xmm_aux2, zmm_dst, 2);
            vextractf32x4(xmm_aux3, zmm_dst, 3);
            horiz_ps(xmm_aux2, xmm_aux3);
            horiz_ps(xmm_aux1, xmm_aux2);
            horiz_store(xmm_aux1, dst_dt, load_embedded);
        }
    }

    inline void horiz_store(Xbyak::Xmm xmm_dst, memory::data_type dst_dt, bool load_embedded) {
        uni_vmovshdup(xmm_aux3, xmm_dst);          // dst:1,2,3,4; aux3:2,2,4,4
        horiz_ps(xmm_dst, xmm_aux3);               // dst:f(1,2),f(2,2),f(3,4),f(4,4)
        uni_vmovhlps(xmm_aux3, xmm_aux3, xmm_dst); // aux3:f(3,4),f(4,4),4,4
        horiz_ps(xmm_dst, xmm_aux3);               // dst:f(1,2,3,4),...
        if (load_embedded) {
            load_scalar(xmm_aux3, ptr[reg_dst], dst_dt);
            horiz_ps(xmm_dst, xmm_aux3);
        }
        store_scalar(ptr[reg_dst], xmm_dst, dst_dt);
    }

    inline void horiz_ps(const Xmm& xmm, const Operand& op) {
        switch (jcp_.reduce_mode) {
            case Algorithm::ReduceAnd:
                uni_vandps(xmm, xmm, op);
                break;
            case Algorithm::ReduceL1:
            case Algorithm::ReduceL2:
            case Algorithm::ReduceLogSum:
            case Algorithm::ReduceMean:
            case Algorithm::ReduceSum:
            case Algorithm::ReduceSumSquare:
            case Algorithm::ReduceLogSumExp:
                uni_vaddps(xmm, xmm, op);
                break;
            case Algorithm::ReduceMax:
                uni_vmaxps(xmm, xmm, op);
                break;
            case Algorithm::ReduceMin:
                uni_vminps(xmm, xmm, op);
                break;
            case Algorithm::ReduceOr:
                uni_vorps(xmm, xmm, op);
                break;
            case Algorithm::ReduceProd:
                uni_vmulps(xmm, xmm, op);
                break;
            default:
                assert(!"unsupported reduce mode");
        }
    }
};


void JitReduceExecutor::reduce_type(const uint8_t *in_ptr, uint8_t *out_ptr, size_t dst_size, const void *post_ops_data_) {
    auto proc_ptr = is_hybrid_layout ? reinterpret_cast<uint8_t *>(prc_mem->get_data_handle())
                                     : out_ptr;

    init_dst_data(proc_ptr, dst_size);
    reduce_stride = IW;

    if (layout == ReduceLayoutType::reduce_ncsp || layout == ReduceLayoutType::reduce_nspc) {
        reduce_PLN(in_ptr, proc_ptr, post_ops_data_);
    } else {
        if (ReduceC && (IC % blk_size)) {
            reduce_BLK_concern_padding(in_ptr, proc_ptr, post_ops_data_);
        } else {
            reduce_BLK(in_ptr, proc_ptr, post_ops_data_);
        }
    }

    if (is_hybrid_layout) {
        if (layout == ReduceLayoutType::reduce_nspc) {
            nspc2ncsp(proc_ptr, out_ptr);
        } else {
            blocked2ncsp(proc_ptr, out_ptr);
        }
    }
}

void JitReduceExecutor::reduce_PLN(const uint8_t *in_ptr, uint8_t *out_ptr, const void *post_ops_data_) {
    if (ReduceN && !ReduceC && !ReduceD && !ReduceH && !ReduceW) {
        size_t IA = IC * ID * IH * IW;
        reduce_stride = IA;
        parallel_for(IA / blk_size, [&](size_t iba){
            size_t oba = iba;
            reduce_kernel_process(in_ptr + iba * blk_size * src_data_size, out_ptr + oba * blk_size * dst_data_size,
                                  blk_size, 0, IB);
        });

        size_t tail_start = IA / blk_size * blk_size;
        reduce_kernel_process(in_ptr + tail_start * src_data_size, out_ptr + tail_start * dst_data_size,
                              IA - tail_start, 0, IB);
    } else {
        for (size_t ib = 0; ib < IB; ib++) {
            size_t ob = ReduceN ? 0 : ib; GET_PTR_N_PLN;
            if (!ReduceC && !ReduceD && ReduceW) {
                size_t work_amount = ReduceH ? IH * IW : IW;
                if (work_amount < blk_size && mayiuse(cpu::x64::avx2)) {
                    size_t outer_size = ReduceH ? IC * ID : IC * ID * IH;
                    size_t inner_size = ReduceH ? IH * IW : IW;
                    size_t output_inner_size = ReduceH ? OH * OW : OW;
                    size_t IK = outer_size / blk_size;
                    std::vector<int> index_buf(blk_size);
                    for (size_t i = 0; i < blk_size; i++) {
                        index_buf[i] = i * work_amount * src_data_size;
                    }
                    parallel_for(IK, [&](size_t ik) {
                        size_t ok = ik;
                        reduce_kernel_process(in_ptr_n + ik * blk_size * inner_size * src_data_size,
                                              out_ptr_n + ok * blk_size * output_inner_size * dst_data_size,
                                              work_amount, 1, 0, static_cast<int *>(&index_buf[0]));
                    });
                    size_t tail_start = IK * blk_size;
                    size_t IT = outer_size - tail_start;
                    parallel_for(IT, [&](size_t it) {
                        size_t ot = it;
                        reduce_kernel_process(in_ptr_n + (tail_start + it) * inner_size * src_data_size,
                                              out_ptr_n + (tail_start + ot) * output_inner_size * dst_data_size, work_amount, 1);
                    });
                } else {
                    if (ReduceH) {
                        parallel_for2d(IC, ID, [&](size_t ic, size_t id) {
                            size_t oc = ic, od = id; GET_PTR_NCD_BASE_PTR_N_PLN;
                            reduce_kernel_process(in_ptr_ncd, out_ptr_ncd, work_amount, 1);
                        });
                    } else {
                        parallel_for3d(IC, ID, IH, [&](size_t ic, size_t id, size_t ih) {
                            size_t oc = ic, od = id; GET_PTR_NCD_BASE_PTR_N_PLN;
                            size_t oh = ih; GET_PTR_NCDH_PLN;
                            reduce_kernel_process(in_ptr_ncdh, out_ptr_ncdh, work_amount, 1);
                        });
                    }
                }
            } else if (ReduceH && ReduceW) {
                for (size_t ic = 0; ic < IC; ic++) {
                    size_t oc = ReduceC ? 0 : ic; GET_PTR_NC_PLN;
                    for (size_t id = 0; id < ID; id++) {
                        size_t od = ReduceD ? 0 : id; GET_PTR_NCD_PLN;
                        reduce_kernel_process(in_ptr_ncd, out_ptr_ncd, IH * IW, 1);
                    }
                }
            } else if (!ReduceH && ReduceW) {
                for (size_t ic = 0; ic < IC; ic++) {
                    size_t oc = ReduceC ? 0 : ic; GET_PTR_NC_PLN;
                    for (size_t id = 0; id < ID; id++) {
                        size_t od = ReduceD ? 0 : id; GET_PTR_NCD_PLN;
                        parallel_for(IH, [&](size_t ih){
                            size_t oh = ih; GET_PTR_NCDH_PLN;
                            reduce_kernel_process(in_ptr_ncdh, out_ptr_ncdh, IW, 1);
                        });
                    }
                }
            } else if (ReduceW) {
                for (size_t ic = 0; ic < IC; ic++) {
                    size_t oc = ReduceC ? 0 : ic; GET_PTR_NC_PLN;
                    for (size_t id = 0; id < ID; id++) {
                        size_t od = ReduceD ? 0 : id; GET_PTR_NCD_PLN;
                        for (size_t ih = 0; ih < IH; ih++) {
                            size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_PLN;
                            reduce_kernel_process(in_ptr_ncdh, out_ptr_ncdh, IW, 1);
                        }
                    }
                }
            } else if (!ReduceC && !ReduceD && ReduceH && !ReduceW) {
                parallel_for2d(IC, ID, [&](size_t ic, size_t id) {
                    size_t oc = ic, od = id; GET_PTR_NCD_BASE_PTR_N_PLN;
                    parallel_for(IW / blk_size, [&](size_t ibw){
                        size_t obw = ibw;
                        reduce_kernel_process(in_ptr_ncd + ibw * blk_size * src_data_size, out_ptr_ncd + obw * blk_size * dst_data_size,
                                              blk_size, 0, IH);
                    });
                    size_t tail_start = IW / blk_size * blk_size;
                    reduce_kernel_process(in_ptr_ncd + tail_start * src_data_size, out_ptr_ncd + tail_start * dst_data_size,
                                          IW - tail_start, 0, IH);
                });
            } else if (!ReduceC && ReduceD && ReduceH && !ReduceW) {
                size_t IWB = IW / blk_size;
                if (ReduceDH_opt) {
                    // reduce parallelly in D dimension
                    // step1: !ReduceD && ReduceH && !ReduceW
                    uint8_t *prc_ptr_n = &vec_reduceDH_prc[0];
                    init_dst_data(prc_ptr_n, prc_size);
                    parallel_for2d(ID, IWB, [&](size_t id, size_t iwb){
                        size_t pd = id, pwb = iwb;
                        reduce_kernel_process(in_ptr_n + (id * IH * IW + iwb * blk_size) * src_data_size,
                                              prc_ptr_n + (pd * PW + pwb * blk_size) * prc_data_size, blk_size, 0, IH);
                    });
                    // step2: ReduceD
                    reduce_stride = PW;
                    parallel_for(IWB, [&](size_t iwb){
                        size_t pwb = iwb, owb = iwb;
                        reduce_kernel_process(prc_ptr_n + pwb * blk_size * prc_data_size,
                                              out_ptr_n + owb * blk_size * dst_data_size, blk_size, 0, ID);
                    });
                    // reduce tail
                    reduce_stride = IW;
                    size_t tail_start = IWB * blk_size;
                    parallel_for(IW - tail_start, [&](size_t i_tail) {
                        reduce_kernel_process(in_ptr_n + (tail_start + i_tail) * src_data_size, out_ptr_n + (tail_start + i_tail) * dst_data_size,
                                            1, 0, ID * IH);
                    });
                } else {
                    parallel_for(IC, [&](size_t ic) {
                        size_t oc = ic; GET_PTR_NC_PLN;
                        parallel_for(IWB, [&](size_t iwb){
                            size_t owb = iwb;
                            reduce_kernel_process(in_ptr_nc + iwb * blk_size * src_data_size, out_ptr_nc + owb * blk_size * dst_data_size,
                                                blk_size, 0, ID * IH);
                        });
                        size_t tail_start = IWB * blk_size;
                        parallel_for(IW - tail_start, [&](size_t i_tail) {
                            reduce_kernel_process(in_ptr_nc + (tail_start + i_tail) * src_data_size, out_ptr_nc + (tail_start + i_tail) * dst_data_size,
                                                1, 0, ID * IH);
                        });
                    });
                }
            } else if (ReduceC && ReduceD && ReduceH && !ReduceW) {
                parallel_for(IW / blk_size, [&](size_t ibw){
                    size_t obw = ibw;
                    reduce_kernel_process(in_ptr_n + ibw * blk_size * src_data_size, out_ptr_n + obw * blk_size * dst_data_size,
                                          blk_size, 0, IC * ID * IH);
                });

                size_t tail_start = IW / blk_size * blk_size;
                reduce_kernel_process(in_ptr_n + tail_start * src_data_size, out_ptr_n + tail_start * dst_data_size,
                                      IW - tail_start, 0, IC * ID * IH);
            } else if (ReduceC && !ReduceD && !ReduceH && !ReduceW) {
                size_t IS = ID * IH * IW;
                reduce_stride = IS;
                parallel_for(IS / blk_size, [&](size_t ibs){
                    size_t obs = ibs;
                    reduce_kernel_process(in_ptr_n + ibs * blk_size * src_data_size, out_ptr_n + obs * blk_size * dst_data_size,
                                          blk_size, 0, IC);
                });

                size_t tail_start = IS / blk_size * blk_size;
                reduce_kernel_process(in_ptr_n + tail_start * src_data_size, out_ptr_n + tail_start * dst_data_size,
                                      IS - tail_start, 0, IC);
            } else {
                for (size_t ic = 0; ic < IC; ic++) {
                    size_t oc = ReduceC ? 0 : ic; GET_PTR_NC_PLN;
                    for (size_t id = 0; id < ID; id++) {
                        size_t od = ReduceD ? 0 : id; GET_PTR_NCD_PLN;
                        for (size_t ih = 0; ih < IH; ih++) {
                            size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_PLN;
                            for (size_t ibw = 0; ibw < IW / blk_size; ibw++) {
                                size_t obw = ibw;
                                reduce_kernel_process(in_ptr_ncdh + ibw * blk_size * src_data_size,
                                                      out_ptr_ncdh + obw * blk_size * dst_data_size, blk_size, 0);
                            }
                            size_t tail_start = IW / blk_size * blk_size;
                            reduce_kernel_process(in_ptr_ncdh + tail_start * src_data_size, out_ptr_ncdh + tail_start * dst_data_size, IW - tail_start, 0);
                        }
                    }
                }
            }
        }
    }

    reduce_kernel_post_process(out_ptr, post_ops_data_);
}

void JitReduceExecutor::reduce_BLK(const uint8_t *in_ptr, uint8_t *out_ptr, const void *post_ops_data_) {
    size_t ICB = div_up(IC, blk_size);
    size_t OCB = div_up(OC, blk_size);

    for (size_t ib = 0; ib < IB; ib++) {
        size_t ob = ReduceN ? 0 : ib; GET_PTR_N_BLK;
        if (!ReduceC && !ReduceD && ReduceH && ReduceW) {
            parallel_for2d(ICB, ID, [&](size_t icb, size_t id) {
                size_t ocb = icb, od = id; GET_PTR_NCD_BASE_PTR_N_BLK;
                reduce_kernel_process(in_ptr_ncd, out_ptr_ncd, IH * IW * blk_size);
            });
        } else if (ReduceC && ReduceD && ReduceH && ReduceW) {
            if (!support_split) {
                reduce_kernel_process(in_ptr_n, out_ptr_n, ICB * ID * IH * IW * blk_size);
            } else {
                // reduce parallelly
                // step1: !ReduceC && ReduceD && ReduceH && ReduceW
                size_t prc_size = ICB * blk_size * dst_data_size;
                std::vector<uint8_t> vec_prc(prc_size);
                init_dst_data(vec_prc.data(), prc_size);
                uint8_t *out_ptr_n_cp = out_ptr_n;
                out_ptr_n = vec_prc.data();
                parallel_for(ICB, [&](size_t icb) {
                    size_t ocb = icb; GET_PTR_NC_BLK;
                    reduce_kernel_process(in_ptr_nc, out_ptr_nc, ID * IH * IW * blk_size);
                });
                // step2: ReduceC
                reduce_kernel_process(out_ptr_n, out_ptr_n_cp, ICB * blk_size);
            }
        } else if (ReduceW) {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = ReduceC ? 0 : icb; GET_PTR_NC_BLK;
                for (size_t id = 0; id < ID; id++) {
                    size_t od = ReduceD ? 0 : id; GET_PTR_NCD_BLK;
                    for (size_t ih = 0; ih < IH; ih++) {
                        size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_BLK;
                        reduce_kernel_process(in_ptr_ncdh, out_ptr_ncdh, IW * blk_size);
                    }
                }
            }
        } else if (ReduceC && !ReduceD && !ReduceH && !ReduceW) {
            reduce_stride = ID * IH * IW * blk_size;
            parallel_for3d(ID, IH, IW, [&](size_t id, size_t ih, size_t iw) {
                size_t icb = 0, ocb = 0; GET_PTR_NC_BLK;
                size_t od = id; GET_PTR_NCD_BLK;
                size_t oh = ih; GET_PTR_NCDH_BLK;
                size_t ow = iw; GET_PTR_NCDHW_BLK;
                reduce_kernel_process(in_ptr_ncdhw, out_ptr_ncdhw, blk_size, 0, ICB);
            });
        } else {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = ReduceC ? 0 : icb; GET_PTR_NC_BLK;
                for (size_t id = 0; id < ID; id++) {
                    size_t od = ReduceD ? 0 : id; GET_PTR_NCD_BLK;
                    for (size_t ih = 0; ih < IH; ih++) {
                        size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_BLK;
                        parallel_for(IW, [&](size_t iw) {
                            size_t ow = iw; GET_PTR_NCDHW_BLK;
                            reduce_kernel_process(in_ptr_ncdhw, out_ptr_ncdhw, blk_size);
                        });
                    }
                }
            }
        }
    }

    reduce_kernel_post_process(out_ptr, post_ops_data_);
}

void JitReduceExecutor::reduce_BLK_concern_padding(const uint8_t *in_ptr, uint8_t *out_ptr, const void *post_ops_data_) {
    size_t ICB = div_up(IC, blk_size);
    size_t OCB = div_up(OC, blk_size);

    auto reduceSkipPadding = [&](const uint8_t *in_ptr_ncd, uint8_t *out_ptr_ncd, size_t ic) {
        size_t blk_valid_size = IC - ic;
        for (size_t ih = 0; ih < IH; ih++) {
            size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_BLK;
            for (size_t iw = 0; iw < IW; iw++) {
                size_t ow = ReduceW ? 0 : iw; GET_PTR_NCDHW_BLK;
                reduce_kernel_process(in_ptr_ncdhw, out_ptr_ncdhw, blk_valid_size);
            }
        }
    };

    for (size_t ib = 0; ib < IB; ib++) {
        size_t ob = ReduceN ? 0 : ib; GET_PTR_N_BLK;
        if (!ReduceD && ReduceH && ReduceW) {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = 0;;
                size_t ic = icb * blk_size;
                parallel_for(ID, [&](size_t id) {
                    size_t od = id; GET_PTR_NCD_BASE_PTR_N_BLK;
                    if (ic + blk_size <= IC) {
                        reduce_kernel_process(in_ptr_ncd, out_ptr_ncd, IH * IW * blk_size);
                    } else {
                        reduceSkipPadding(in_ptr_ncd, out_ptr_ncd, ic);
                    }
                });
            }
        } else if (ReduceD && ReduceH && ReduceW) {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = 0; GET_PTR_NC_BLK;
                size_t ic = icb * blk_size;
                if (ic + blk_size <= IC) {
                    reduce_kernel_process(in_ptr_nc, out_ptr_nc, ID * IH * IW * blk_size);
                } else {
                    for (size_t id = 0; id < ID; id++) {
                        size_t od = 0; GET_PTR_NCD_BLK;
                        reduceSkipPadding(in_ptr_ncd, out_ptr_ncd, ic);
                    }
                }
            }
        } else if (ReduceW) {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = 0; GET_PTR_NC_BLK;
                size_t ic = icb * blk_size;
                for (size_t id = 0; id < ID; id++) {
                    size_t od = ReduceD ? 0 : id; GET_PTR_NCD_BLK;
                    if (ic + blk_size <= IC) {
                        for (size_t ih = 0; ih < IH; ih++) {
                            size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_BLK;
                            reduce_kernel_process(in_ptr_ncdh, out_ptr_ncdh, IW * blk_size);
                        }
                    } else {
                        reduceSkipPadding(in_ptr_ncd, out_ptr_ncd, ic);
                    }
                }
            }
        } else {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = 0; GET_PTR_NC_BLK;
                size_t ic = icb * blk_size;
                for (size_t id = 0; id < ID; id++) {
                    size_t od = ReduceD ? 0 : id; GET_PTR_NCD_BLK;
                    if (ic + blk_size <= IC) {
                        for (size_t ih = 0; ih < IH; ih++) {
                            size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_BLK;
                            parallel_for(IW, [&](size_t iw) {
                                size_t ow = iw; GET_PTR_NCDHW_BLK;
                                reduce_kernel_process(in_ptr_ncdhw, out_ptr_ncdhw, blk_size);
                            });
                        }
                    } else {
                        reduceSkipPadding(in_ptr_ncd, out_ptr_ncd, ic);
                    }
                }
            }
        }
    }

    reduce_kernel_post_process(out_ptr, post_ops_data_);
}

inline void JitReduceExecutor::reduce_kernel_process(const uint8_t *in_p, uint8_t *out_p, size_t work_amount,
                                                    size_t reduce_w, size_t work_batch, const int *tab_idx) {
    auto arg = jit_reduce_call_args();
    arg.src = static_cast<const void *>(in_p);
    arg.idx = tab_idx;
    arg.dst = static_cast<void *>(out_p);
    arg.work_amount = work_amount;
    arg.work_batch = work_batch;
    arg.reduce_w = reduce_w;
    arg.reduce_stride = reduce_stride;

    (*reduce_kernel)(&arg);
}

inline void JitReduceExecutor::reduce_kernel_post_process(uint8_t *out_ptr, const void *post_ops_data_) {
    const size_t integerDivisor = IB * IC * ID * IH * IW / (OB * OC * OD * OH * OW);
    const float divisor = static_cast<float>(integerDivisor);
    if (layout == ReduceLayoutType::reduce_ncsp || layout == ReduceLayoutType::reduce_nspc) {
        parallel_for2d(OB, OC, [&](size_t ob, size_t oc) {
            uint8_t *out_p = out_ptr + (ob * OC + oc) * OD * OH * OW * dst_data_size;
            auto arg = jit_reduce_post_call_args();
            arg.dst = static_cast<void *>(out_p);
            arg.oc_off = layout == ReduceLayoutType::reduce_nspc ? 0 : oc * sizeof(float);
            arg.channel_size = layout == ReduceLayoutType::reduce_nspc ? OW : OC; // OW is related to nspc-ncsp dimension reinterpret
            arg.work_amount = OD * OH * OW;
            arg.divisor = &divisor;
            arg.post_op_data = post_ops_data_;
            (*reduce_post_kernel)(&arg);
        });
    } else {
        size_t OCB = div_up(OC, blk_size);
        parallel_for2d(OB, OCB, [&](size_t ob, size_t ocb) {
            uint8_t *out_p = out_ptr + (ob * OCB + ocb) * OD * OH * OW * blk_size * dst_data_size;
            auto arg = jit_reduce_post_call_args();
            arg.dst = static_cast<void *>(out_p);
            arg.reduce_c = ReduceC ? 1 : 0;
            arg.oc_off = ocb * blk_size * sizeof(float);
            arg.work_amount = OD * OH * OW * blk_size;
            arg.divisor = &divisor;
            arg.post_op_data = post_ops_data_;
            (*reduce_post_kernel)(&arg);
        });
    }
}

void JitReduceExecutor::nspc2ncsp(uint8_t *proc_ptr, uint8_t *out_ptr) {
    // dimension reinterpret after nspc reusing routine reduce_PLN
    // demote -- nspc -- ncsp
    //  DIM0  --   B  --  B
    //  DIM1  --   C  --  W
    //  DIM2  --   D  --  C
    //  DIM3  --   H  --  D
    //  DIM4  --   W  --  H
    const size_t DIM0 = OB;
    const size_t DIM1 = OW;
    const size_t DIM2 = OC;
    const size_t DIM3 = OD;
    const size_t DIM4 = OH;
    const size_t stride1 = DIM2 * DIM3 * DIM4;
    const size_t stride0 = stride1 * DIM1;

    if (dst_data_size == 4) {
        auto src_data = reinterpret_cast<const float *>(proc_ptr);
        auto dst_data = reinterpret_cast<float *>(out_ptr);
        parallel_for2d(DIM0, stride1, [&](size_t b, size_t j) {
            auto src_off = b * stride0 + j * DIM1;
            auto dst_off = b * stride0 + j;
            for (size_t dim1 = 0; dim1 < DIM1; dim1++) {
                dst_data[dst_off] = src_data[src_off];
                src_off++;
                dst_off += stride1;
            }
        });
    } else if (dst_data_size == 2) {
        auto src_data = reinterpret_cast<const uint16_t *>(proc_ptr);
        auto dst_data = reinterpret_cast<uint16_t *>(out_ptr);
        parallel_for2d(DIM0, stride1, [&](size_t b, size_t j) {
            auto src_off = b * stride0 + j * DIM1;
            auto dst_off = b * stride0 + j;
            for (size_t dim1 = 0; dim1 < DIM1; dim1++) {
                dst_data[dst_off] = src_data[src_off];
                src_off++;
                dst_off += stride1;
            }
        });
    } else {
        auto src_data = reinterpret_cast<const uint8_t *>(proc_ptr);
        auto dst_data = reinterpret_cast<uint8_t *>(out_ptr);
        parallel_for2d(DIM0, stride1, [&](size_t b, size_t j) {
            auto src_off = b * stride0 + j * DIM1;
            auto dst_off = b * stride0 + j;
            for (size_t dim1 = 0; dim1 < DIM1; dim1++) {
                dst_data[dst_off] = src_data[src_off];
                src_off++;
                dst_off += stride1;
            }
        });
    }
}

void JitReduceExecutor::blocked2ncsp(uint8_t *proc_ptr, uint8_t *out_ptr) {
    const size_t DIM0 = OB;
    const size_t DIM1 = OC;
    const size_t DIM2 = OD;
    const size_t DIM3 = OH;
    const size_t DIM4 = OW;
    const size_t stride1 = DIM2 * DIM3 * DIM4;
    const size_t src_stride0 = stride1 * div_up(OC, blk_size) * blk_size;
    const size_t dst_stride0 = stride1 * DIM1;

    if (dst_data_size == 4) {
        auto src_data = reinterpret_cast<const float *>(proc_ptr);
        auto dst_data = reinterpret_cast<float *>(out_ptr);
        parallel_for2d(DIM0, stride1, [&](size_t b, size_t j) {
            auto src_off = b * src_stride0 + j * blk_size;
            auto dst_off = b * dst_stride0 + j;
            for (size_t dim1 = 0; dim1 + blk_size <= DIM1; dim1 += blk_size) {
                for (size_t k = 0; k < blk_size; k++) {
                    dst_data[dst_off] = src_data[src_off];
                    src_off++;
                    dst_off += stride1;
                }
                src_off += (stride1 - 1) * blk_size;
            }
            size_t tail = DIM1 % blk_size;
            for (size_t k = 0; k < tail; k++) {
                dst_data[dst_off] = src_data[src_off];
                src_off++;
                dst_off += stride1;
            }
        });
    } else if (dst_data_size == 2) {
        auto src_data = reinterpret_cast<const uint16_t *>(proc_ptr);
        auto dst_data = reinterpret_cast<uint16_t *>(out_ptr);
        parallel_for2d(DIM0, stride1, [&](size_t b, size_t j) {
            auto src_off = b * src_stride0 + j * blk_size;
            auto dst_off = b * dst_stride0 + j;
            for (size_t dim1 = 0; dim1 + blk_size <= DIM1; dim1 += blk_size) {
                for (size_t k = 0; k < blk_size; k++) {
                    dst_data[dst_off] = src_data[src_off];
                    src_off++;
                    dst_off += stride1;
                }
                src_off += (stride1 - 1) * blk_size;
            }
            size_t tail = DIM1 % blk_size;
            for (size_t k = 0; k < tail; k++) {
                dst_data[dst_off] = src_data[src_off];
                src_off++;
                dst_off += stride1;
            }
        });
    } else {
        auto src_data = reinterpret_cast<const uint8_t *>(proc_ptr);
        auto dst_data = reinterpret_cast<uint8_t *>(out_ptr);
        parallel_for2d(DIM0, stride1, [&](size_t b, size_t j) {
            auto src_off = b * src_stride0 + j * blk_size;
            auto dst_off = b * dst_stride0 + j;
            for (size_t dim1 = 0; dim1 + blk_size <= DIM1; dim1 += blk_size) {
                for (size_t k = 0; k < blk_size; k++) {
                    dst_data[dst_off] = src_data[src_off];
                    src_off++;
                    dst_off += stride1;
                }
                src_off += (stride1 - 1) * blk_size;
            }
            size_t tail = DIM1 % blk_size;
            for (size_t k = 0; k < tail; k++) {
                dst_data[dst_off] = src_data[src_off];
                src_off++;
                dst_off += stride1;
            }
        });
    }
}

inline void JitReduceExecutor::init_dst_data(uint8_t *out_ptr, size_t dst_size) {
    switch (reduceAttrs.operation) {
        case Algorithm::ReduceL1:
        case Algorithm::ReduceL2:
        case Algorithm::ReduceLogSum:
        case Algorithm::ReduceLogSumExp:
        case Algorithm::ReduceMean:
        case Algorithm::ReduceOr:
        case Algorithm::ReduceSum:
        case Algorithm::ReduceSumSquare:
            memset(out_ptr, 0, dst_size);
            break;
        case Algorithm::ReduceAnd:
        case Algorithm::ReduceProd:
            if (output_prec == Precision::FP32) {
                auto out_p = reinterpret_cast<float *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = static_cast<float>(1); });
            } else if (output_prec == Precision::I32) {
                auto out_p = reinterpret_cast<int32_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = static_cast<int32_t>(1); });
            } else if (output_prec == Precision::BF16) {
                auto out_p = reinterpret_cast<bfloat16_t*>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = static_cast<bfloat16_t>(1); });
            } else if (output_prec == Precision::U8) {
                auto out_p = reinterpret_cast<uint8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = static_cast<uint8_t>(1); });
            } else if (output_prec == Precision::I8) {
                auto out_p = reinterpret_cast<int8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = static_cast<int8_t>(1); });
            }
            break;
        case Algorithm::ReduceMax:
            if (output_prec == Precision::FP32) {
                auto out_p = reinterpret_cast<float *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<float>::lowest(); });
            } else if (output_prec == Precision::I32) {
                auto out_p = reinterpret_cast<int32_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<int32_t>::min(); });
            } else if (output_prec == Precision::BF16) {
                auto out_p = reinterpret_cast<bfloat16_t*>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<bfloat16_t>::lowest(); });
            } else if (output_prec == Precision::U8) {
                auto out_p = reinterpret_cast<uint8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<uint8_t>::min(); });
            } else if (output_prec == Precision::I8) {
                auto out_p = reinterpret_cast<int8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<int8_t>::min(); });
            }
            break;
        case Algorithm::ReduceMin:
            if (output_prec == Precision::FP32) {
                auto out_p = reinterpret_cast<float *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<float>::max(); });
            } else if (output_prec == Precision::I32) {
                auto out_p = reinterpret_cast<int32_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<int32_t>::max(); });
            } else if (output_prec == Precision::BF16) {
                auto out_p = reinterpret_cast<bfloat16_t*>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<bfloat16_t>::max(); });
            } else if (output_prec == Precision::U8) {
                auto out_p = reinterpret_cast<uint8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<uint8_t>::max(); });
            } else if (output_prec == Precision::I8) {
                auto out_p = reinterpret_cast<int8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<int8_t>::max(); });
            }
            break;
        default:
            IE_THROW() << " gets unsupported reduce mode.";
    }
}

inline void JitReduceExecutor::create_working_memory(size_t rank) {
    memory::format_tag format = (layout == ReduceLayoutType::reduce_nspc) ? (rank == 4 ? memory::format_tag::nhwc : memory::format_tag::ndhwc)
                                        : (rank == 4 ? (mayiuse(cpu::x64::avx512_core) ? memory::format_tag::nChw16c : memory::format_tag::nChw8c)
                                                     : (mayiuse(cpu::x64::avx512_core) ? memory::format_tag::nCdhw16c : memory::format_tag::nCdhw8c));
    auto prc_dims = rank == 4 ? std::vector<size_t>{OB, OC, OH, OW} : std::vector<size_t>{OB, OC, OD, OH, OW};
    auto desc = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(prc_dims), DnnlExtensionUtils::IEPrecisionToDataType(output_prec), format);
    prc_mem = std::make_shared<dnnl::memory>(desc, context->getEngine());
    dst_size = desc.get_size();
}

inline void JitReduceExecutor::create_DH_working_memory() {
    ReduceDH_opt = layout == ReduceLayoutType::reduce_nspc && !isDynamic && support_split &&
                   !ReduceC && ReduceD && ReduceH && !ReduceW && IC == 1 && ID > 1;
    if (ReduceDH_opt) {
        PD = ID;
        PW = IW / blk_size * blk_size;
        prc_data_size = src_data_size;
        prc_size = PD * PW * src_data_size;
        if (prc_size > vec_reduceDH_prc.size()) {
            vec_reduceDH_prc.resize(prc_size);
        }
    }
}

inline void JitReduceExecutor::calc_process_dst_dims(std::vector<int> &reduce_axes, const SizeVector &dst_dims) {
    std::set<size_t> axes;
    SizeVector out_dims;
    process_dst_dims.clear();
    axes_for_reduction.clear();
    for (auto &axis : reduce_axes) {
        if (axis < 0)
            axis += src_dims.size();
        if (static_cast<size_t>(axis) > src_dims.size())
            IE_THROW() << "exceeds data tensor dimension on index to reduce";
        axes.insert(static_cast<size_t>(axis));
    }
    for (size_t i = 0; i < src_dims.size(); i++) {
        bool found = false;
        for (auto axis : axes) {
            if (i == axis) {
                found = true;
                break;
            }
        }
        if (found) {
            if (reduceAttrs.keepDims)
                out_dims.push_back(1);
            process_dst_dims.push_back(1);
            axes_for_reduction.push_back(i);
        } else {
            out_dims.push_back(src_dims[i]);
            process_dst_dims.push_back(src_dims[i]);
        }
    }

    if (jit_beyond_5D) {
        if (std::accumulate(out_dims.begin(), out_dims.end(), size_t(1), std::multiplies<size_t>()) !=
            std::accumulate(dst_dims.begin(), dst_dims.end(), size_t(1), std::multiplies<size_t>()))
            IE_THROW() << "gets incorrect number of output dimensions!";
    } else {
        for (size_t i = 0; i < std::min(out_dims.size(), dst_dims.size()); i++) {
            if (out_dims[i] != dst_dims[i])
                IE_THROW() << "gets incorrect number of output dimensions!";
        }
    }
}

inline void JitReduceExecutor::set_reduce_dim_flags(size_t rank) {
    size_t dims_size = src_dims.size();
    if (dims_size == 5) {
        SET_SRC_DIM_VALUE(src_dims[0], src_dims[1], src_dims[2], src_dims[3], src_dims[4]);
        SET_DST_DIM_VALUE(process_dst_dims[0], process_dst_dims[1], process_dst_dims[2], process_dst_dims[3], process_dst_dims[4]);
    } else if (dims_size == 4) {
        SET_SRC_DIM_VALUE(src_dims[0], src_dims[1], 1, src_dims[2], src_dims[3]);
        SET_DST_DIM_VALUE(process_dst_dims[0], process_dst_dims[1], 1, process_dst_dims[2], process_dst_dims[3]);
    } else if (dims_size == 3) {
        SET_SRC_DIM_VALUE(1, src_dims[0], 1, src_dims[1], src_dims[2]);
        SET_DST_DIM_VALUE(1, process_dst_dims[0], 1, process_dst_dims[1], process_dst_dims[2]);
    } else if (dims_size == 2) {
        SET_SRC_DIM_VALUE(1, 1, 1, src_dims[0], src_dims[1]);
        SET_DST_DIM_VALUE(1, 1, 1, process_dst_dims[0], process_dst_dims[1]);
    } else {
        SET_SRC_DIM_VALUE(1, src_dims[0], 1, 1, 1);
        SET_DST_DIM_VALUE(1, process_dst_dims[0], 1, 1, 1);
    }

    // must be done before the following dimension change
    if (is_hybrid_layout) {
        create_working_memory(rank);
    }

    // Reducing a dimesion in nspc layout can be treated as reducing another dimension in ncsp layout,
    // eg. reducing C in nspc can be treated as reducing W in ncsp layout, so that the routine reduce_PLN can be reused.
    // nspc -- ncsp
    //    D -- C
    //    H -- D
    //    W -- H
    //    C -- W
    if (layout == ReduceLayoutType::reduce_nspc) {
        size_t ITmp = IC; IC = ID; ID = IH; IH = IW; IW = ITmp;
        size_t OTmp = OC; OC = OD; OD = OH; OH = OW; OW = OTmp;
    }

    ReduceN = IB != OB && OB == 1;
    ReduceC = IC != OC && OC == 1;
    ReduceD = ID != OD && OD == 1;
    ReduceH = IH != OH && OH == 1;
    ReduceW = IW != OW && OW == 1;

    // must be done before the above dimension change
    create_DH_working_memory();

    // suit for parallel
    if (ReduceH && IW == 1) {
        ReduceW = true;
    }
    if (ReduceC && ReduceH && ID == 1) {
        ReduceD = true;
    }
}

void JitReduceExecutor::setJITBeyond5D(size_t rank) {
    jit_beyond_5D = false;
    if (rank > 5) {
        if (reduceAttrs.axes.size() <= 1) {
            jit_beyond_5D = true;
        } else {
            for (size_t i = 1; i < reduceAttrs.axes.size(); i++) {
                if (reduceAttrs.axes[i] != reduceAttrs.axes[i - 1] + 1) {
                    jit_beyond_5D = false;
                    break;
                }
                jit_beyond_5D = true;
            }
        }
    }
}

std::vector<int> JitReduceExecutor::update_src_dims() {
    std::vector<int> reduce_axes = reduceAttrs.axes;

    if (reduce_axes.size() < 1)
        return reduce_axes;

    size_t axis_dim = 1;
    size_t outer_dim = 1;
    size_t inner_dim = 1;
    int outer_end = reduce_axes[0];
    int inner_start = reduce_axes[reduce_axes.size() - 1];
    for (size_t i = 0; i < src_dims.size(); i++) {
        if (i < outer_end) {
            outer_dim *= src_dims[i];
        } else if (i > inner_start) {
            inner_dim *= src_dims[i];
        } else {
            axis_dim *= src_dims[i];
        }
    }

    reduce_axes.clear();
    reduce_axes.push_back(1);

    src_dims.clear();
    src_dims.push_back(outer_dim);
    src_dims.push_back(axis_dim);
    src_dims.push_back(inner_dim);

    return reduce_axes;
}

JitReduceExecutor::JitReduceExecutor(const ExecutorContext::CPtr context) : ReduceExecutor(context) {}

bool JitReduceExecutor::init(const ReduceAttrs& reduceAttrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs,
                             const dnnl::primitive_attr &attr) {
    this->reduceAttrs = reduceAttrs;

    isDynamic = srcDescs[0]->getShape().isDynamic() || dstDescs[0]->getShape().isDynamic();

    input_prec = srcDescs[0]->getPrecision();
    output_prec = dstDescs[0]->getPrecision();

    setJITBeyond5D(srcDescs[0]->getShape().getRank());

    static const Precision supportedPrecisions[] = {
            Precision::FP32,
            Precision::BF16,
            Precision::I32,
            Precision::I8,
            Precision::U8
    };

    if (!((mayiuse(cpu::x64::sse41)) && (srcDescs[0]->getShape().getRank() <= 5 || jit_beyond_5D) &&
           std::find(std::begin(supportedPrecisions), std::end(supportedPrecisions), input_prec) != std::end(supportedPrecisions) &&
           std::find(std::begin(supportedPrecisions), std::end(supportedPrecisions), output_prec) != std::end(supportedPrecisions))) {
        return false;
    }

    support_split = reduceAttrs.operation != Algorithm::ReduceL2 && reduceAttrs.operation != Algorithm::ReduceLogSumExp &&
                    reduceAttrs.operation != Algorithm::ReduceSumSquare && input_prec == output_prec;

    src_data_size = input_prec.size();
    dst_data_size = output_prec.size();

    if (srcDescs[0]->hasLayoutType(LayoutType::ncsp)) {
        layout = ReduceLayoutType::reduce_ncsp;
    } else if (srcDescs[0]->hasLayoutType(LayoutType::nspc)) {
        layout = ReduceLayoutType::reduce_nspc;
    } else {
        layout = ReduceLayoutType::reduce_blocked;
    }

    // hybrid layout: nspc/blocked layout for input and ncsp for output
    // !reduceAttrs.keepDims is needed to avoid hybrid layout for cases eg. (A, B, C, D) reduce to (A, 1, 1, 1)
    if (!reduceAttrs.keepDims && (layout == ReduceLayoutType::reduce_nspc || layout == ReduceLayoutType::reduce_blocked)) {
        is_hybrid_layout = dstDescs[0]->hasLayoutType(LayoutType::ncsp);
    }

    jcp = jit_reduce_config_params();
    jcp.src_dt = DnnlExtensionUtils::IEPrecisionToDataType(input_prec);
    jcp.dst_dt = DnnlExtensionUtils::IEPrecisionToDataType(output_prec);
    jcp.src_data_size = src_data_size;
    jcp.dst_data_size = dst_data_size;
    jcp.layout = layout;
    jcp.reduce_mode = reduceAttrs.operation;

    compile_post_kernel = true;

    if (mayiuse(cpu::x64::avx512_core)) {
        blk_size = 16;
    } else {
        blk_size = 8;
    }

    if (mayiuse(cpu::x64::avx512_core)) {
        reduce_kernel.reset(new jit_uni_reduce_kernel_f32<cpu::x64::avx512_core>(jcp));
        implType = impl_desc_type::jit_avx512;
    } else if (mayiuse(cpu::x64::avx2)) {
        reduce_kernel.reset(new jit_uni_reduce_kernel_f32<cpu::x64::avx2>(jcp));
        implType = impl_desc_type::jit_avx2;
    } else if (mayiuse(cpu::x64::sse41)) {
        reduce_kernel.reset(new jit_uni_reduce_kernel_f32<cpu::x64::sse41>(jcp));
        implType = impl_desc_type::jit_sse42;
    }

    if (reduce_kernel)
        reduce_kernel->create_ker();

    // TODO: part above should be done on compilation stage, not per each inference

    src_dims = srcDescs[0]->getShape().getDims();
    std::vector<int> reduce_axes;
    if (jit_beyond_5D) {
        reduce_axes = update_src_dims();
    } else {
        reduce_axes = reduceAttrs.axes;
    }

    const SizeVector &dst_dims = dstDescs[0]->getShape().getDims();
    dst_size = dstDescs[0]->getCurrentMemSize();
    calc_process_dst_dims(reduce_axes, dst_dims);
    set_reduce_dim_flags(srcDescs[0]->getShape().getRank());

    auto builder = [&](const ReduceKey& key) -> std::shared_ptr<jit_uni_reduce_post_kernel> {
        std::shared_ptr<jit_uni_reduce_post_kernel> post_kernel;

        if (mayiuse(cpu::x64::avx512_core)) {
            post_kernel.reset(new jit_uni_reduce_post_kernel_f32<cpu::x64::avx512_core>(key.jcp, *attr.get()));
        } else if (mayiuse(cpu::x64::avx2)) {
            post_kernel.reset(new jit_uni_reduce_post_kernel_f32<cpu::x64::avx2>(key.jcp, *attr.get()));
        } else if (mayiuse(cpu::x64::sse41)) {
            post_kernel.reset(new jit_uni_reduce_post_kernel_f32<cpu::x64::sse41>(key.jcp, *attr.get()));
        }
        if (post_kernel)
            post_kernel->create_ker();

        return post_kernel;
    };

    if (compile_post_kernel) {
        // setPostOps(attr, dst_dims, true);

        ReduceKey key = {jcp, attr.get_post_ops()};
        auto cache = context->getRuntimeCache();
        auto result = cache.lock()->getOrCreate(key, builder);
        if (!result.first) {
            IE_THROW() << "has not found jit_uni_reduce_post_kernel_f32.";
        }

        reduce_post_kernel = result.first;

        if (!isDynamic) {
            compile_post_kernel = false;
        }
    }

    return true;
}

void JitReduceExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) {
    const uint8_t *src_data = reinterpret_cast<uint8_t*>(src[0]->GetPtr());
    uint8_t *dst_data = reinterpret_cast<uint8_t*>(dst[0]->GetPtr());

    reduce_type(src_data, dst_data, dst_size, post_ops_data_);
}

// JitReduceExecutor::Key::Key(const ReduceAttrs& reduceAttrs,
//                     const std::vector<MemoryDescCPtr>& srcDescs,
//                     const std::vector<MemoryDescCPtr>& dstDescs,
//                     const dnnl::primitive_attr &attr) {
//     auto blockedDesc = srcDescs[0]->as<BlockedMemoryDesc>();
//     this->reduceAttrs = reduceAttrs;
//     this->srcDims = blockedDesc->getShape().getStaticDims();
//     this->srcOrder = blockedDesc->getOrder();
//     this->srcPrc = srcDescs[0]->getPrecision();
//     this->dstPrc = dstDescs[0]->getPrecision();
// }

// size_t JitReduceExecutor::Key::hash() const {
//     using namespace dnnl::impl;
//     using namespace dnnl::impl::primitive_hashing;

//     size_t seed = 0;

//     seed = hash_combine(seed, reduceAttrs.initAcrossChannels_);
//     seed = hash_combine(seed, reduceAttrs.normalizeVariance_);
//     seed = hash_combine(seed, reduceAttrs.epsValue_);
//     seed = hash_combine(seed, reduceAttrs.epsMode_);
//     seed = get_vector_hash(seed, srcDims);
//     seed = get_vector_hash(seed, srcOrder);
//     seed = hash_combine(seed, srcPrc.getPrecVal());
//     seed = hash_combine(seed, dstPrc.getPrecVal());
//     seed = hash_combine(seed, get_attr_hash(*attr.get()));
//     return seed;
// }

// bool JitReduceExecutor::Key::operator==(const Key& rhs) const {
//     bool retVal = true;
//     retVal = retVal &&
//              reduceAttrs.initAcrossChannels_ == rhs.reduceAttrs.initAcrossChannels_ &&
//              reduceAttrs.normalizeVariance_ == rhs.reduceAttrs.normalizeVariance_ &&
//              reduceAttrs.epsValue_ == rhs.reduceAttrs.epsValue_ &&
//              reduceAttrs.epsMode_ == rhs.reduceAttrs.epsMode_ &&
//              srcDims == rhs.srcDims &&
//              srcOrder == rhs.srcOrder &&
//              srcPrc == rhs.srcPrc &&
//              dstPrc == rhs.dstPrc;
//     retVal = retVal && *attr.get() == *rhs.attr.get();
//     return retVal;
// }

}   // namespace intel_cpu
}   // namespace ov