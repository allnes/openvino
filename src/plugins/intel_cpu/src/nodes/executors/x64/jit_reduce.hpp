// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/reduce.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"

namespace ov {
namespace intel_cpu {

enum ReduceLayoutType {
    reduce_ncsp,
    reduce_nspc,
    reduce_blocked
};

struct jit_reduce_config_params {
    ReduceLayoutType layout;
    Algorithm reduce_mode;
    dnnl::memory::data_type src_dt;
    dnnl::memory::data_type dst_dt;
    int src_data_size;
    int dst_data_size;
};

struct jit_reduce_call_args {
    const void *src;
    const int *idx;
    void *dst;
    size_t work_amount;
    size_t work_batch;
    size_t reduce_w = 2;    // only used in planar layout  [1: reduce width dimension]   [0: reduce other dimension] [other value: N/A]
    size_t reduce_stride;   // only used in planar layout while reducing dimensions except for width
};

struct jit_reduce_post_call_args {
    const void *src;
    void *dst;
    size_t work_amount;
    size_t reduce_c = 2;    // only used in blocked layout [1: reduce channel dimension] [0: reduce other dimension] [other value: N/A]
    size_t oc_off;          // offset in byte along channel on output tensor
    size_t channel_size;    // only for post ops fusion of nspc layout
    const float *divisor;   // mean = sum / divisor
    const void* post_op_data;
};

struct jit_uni_reduce_kernel {
    void (*ker_)(const jit_reduce_call_args *);

    void operator()(const jit_reduce_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_reduce_kernel(jit_reduce_config_params jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_reduce_kernel() {}

    virtual void create_ker() = 0;

    jit_reduce_config_params jcp_;
};

struct jit_uni_reduce_post_kernel {
    void (*ker_)(const jit_reduce_post_call_args *);

    void operator()(const jit_reduce_post_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_reduce_post_kernel(jit_reduce_config_params jcp, const dnnl_primitive_attr &attr) : ker_(nullptr), jcp_(jcp), attr_(attr) {}
    virtual ~jit_uni_reduce_post_kernel() {}

    virtual void create_ker() = 0;

    jit_reduce_config_params jcp_;
    const dnnl_primitive_attr &attr_;
};

class JitReduceExecutor : public ReduceExecutor {
public:
    JitReduceExecutor(const ExecutorContext::CPtr context);

    bool init(const ReduceAttrs& reduceAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) override;

    impl_desc_type getImplType() const override {
        return implType;
    }

    // struct Key {
    //     ReduceAttrs reduceAttrs;
    //     VectorDims srcDims;
    //     VectorDims srcOrder;
    //     InferenceEngine::Precision srcPrc;
    //     InferenceEngine::Precision dstPrc;
    //     dnnl::primitive_attr attr;

    //     Key(const ReduceAttrs& reduceAttrs,
    //         const std::vector<MemoryDescCPtr>& srcDescs,
    //         const std::vector<MemoryDescCPtr>& dstDescs,
    //         const dnnl::primitive_attr &attr);
    //     size_t hash() const;
    //     bool operator==(const Key& rhs) const;
    // };

private:
    void reduce_type(const uint8_t *in_ptr, uint8_t *out_ptr, size_t dst_size, const void *post_ops_data_);
    void reduce_PLN(const uint8_t *in_ptr, uint8_t *out_ptr, const void *post_ops_data_);
    void reduce_BLK(const uint8_t *in_ptr, uint8_t *out_ptr, const void *post_ops_data_);
    void reduce_BLK_concern_padding(const uint8_t *in_ptr, uint8_t *out_ptr, const void *post_ops_data_);
    inline void reduce_kernel_process(const uint8_t *in_p, uint8_t *out_p, size_t work_amount,
                                      size_t reduce_w = 2, size_t work_batch = 1, const int *tab_idx = NULL);
    inline void reduce_kernel_post_process(uint8_t *out_ptr, const void *post_ops_data_);
    inline void init_dst_data(uint8_t *out_ptr, size_t dst_size);
    inline void create_working_memory(size_t rank);
    inline void create_DH_working_memory();
    inline void calc_process_dst_dims(std::vector<int> &reduce_axes, const InferenceEngine::SizeVector &dst_dim);
    inline void set_reduce_dim_flags(size_t rank);
    void nspc2ncsp(uint8_t *proc_ptr, uint8_t *out_ptr);
    void blocked2ncsp(uint8_t *proc_ptr, uint8_t *out_ptr);
    void setJITBeyond5D(size_t rank);
    std::vector<int> update_src_dims();

    ReduceAttrs reduceAttrs;
    InferenceEngine::Precision input_prec;
    InferenceEngine::Precision output_prec;
    size_t blk_size;
    size_t dst_size;
    size_t prc_size;
    bool jit_beyond_5D = false;
    bool is_hybrid_layout = false;
    bool compile_post_kernel = true;
    bool support_split = false;
    bool ReduceDH_opt = false;
    bool ReduceN, ReduceC, ReduceD, ReduceH, ReduceW;
    size_t IB, IC, ID, IH, IW;
    size_t OB, OC, OD, OH, OW;
    size_t PD, PW;
    size_t src_data_size, dst_data_size, prc_data_size;
    size_t reduce_stride;
    ReduceLayoutType layout;
    InferenceEngine::SizeVector src_dims;
    InferenceEngine::SizeVector process_dst_dims;
    InferenceEngine::SizeVector axes_for_reduction;
    bool isDynamic = false;

    jit_reduce_config_params jcp;

    dnnl::memory prc_mem;
    std::vector<uint8_t> vec_reduceDH_prc = {};

    std::shared_ptr<jit_uni_reduce_kernel> reduce_kernel;
    std::shared_ptr<jit_uni_reduce_post_kernel> reduce_post_kernel;

    impl_desc_type implType = impl_desc_type::jit_uni;
};

class JitReduceExecutorBuilder : public ReduceExecutorBuilder {
public:
    bool isSupported(const ReduceAttrs& reduceAttrs, const std::vector<MemoryDescPtr>& srcDescs, const std::vector<MemoryDescPtr>& dstDescs) const override {
        return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41);
    }

    ReduceExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<JitReduceExecutor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov