// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_fullyconnected.hpp"
#include "acl_utils.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/debug_capabilities.h"
#include "nodes/executors/debug_messages.hpp"
#include "nodes/executors/implementation_utils.hpp"

namespace ov {
namespace intel_cpu {

void reorder_to_weight_format(ACLInfo &info,
                              dnnl::impl::memory_desc_t &md,
                              MemoryPtr memoryPtr,
                              arm_compute::WeightFormat wf,
                              dnnl::impl::dim_t I_dim,
                              dnnl::impl::dim_t O_dim,
                              std::vector<dnnl::impl::dim_t> spatial_dims,
                              std::vector<dnnl::impl::dim_t> batch_dims) {
    md.format_kind = dnnl::impl::format_kind::blocked;
    md.format_desc.blocking = dnnl::impl::blocking_desc_t {};
    const int interleaved_by = arm_compute::interleave_by(wf);
    const int block_by = arm_compute::block_by(wf);

    // I dimension becomes densest (apart from blocking)
    md.format_desc.blocking.strides[I_dim] = interleaved_by * block_by;
    md.padded_dims[I_dim] = dnnl::impl::utils::rnd_up(md.dims[I_dim], block_by);

    // Then any spatial dimensions (e.g. HW)
    dnnl::impl::dim_t ldb = interleaved_by * md.padded_dims[I_dim];
    for (dnnl::impl::dim_t sd : spatial_dims) {
        md.format_desc.blocking.strides[sd] = ldb;
        ldb *= md.padded_dims[sd];
    }

    // O dim (which was the innermost) becomes the outermost (apart from batching)
    md.format_desc.blocking.strides[O_dim] = ldb;
    md.padded_dims[O_dim] = dnnl::impl::utils::rnd_up(md.dims[O_dim], interleaved_by);

    // Update the batch dimensions, starting with stride of the innermost batch
    const dnnl::impl::dim_t innermost_batch_stride
            = md.padded_dims[I_dim] * md.padded_dims[O_dim];
    dnnl::impl::dim_t batch_stride = innermost_batch_stride;
    for (dnnl::impl::dim_t bd : batch_dims) {
        md.format_desc.blocking.strides[bd] = batch_stride;
        batch_stride *= md.padded_dims[bd];
    }

    // Weights can only be blocked if they are also interleaved
    if (interleaved_by > 1) {
        md.format_desc.blocking.inner_nblks = 1 + (block_by > 1);

        md.format_desc.blocking.inner_idxs[0] = O_dim;
        md.format_desc.blocking.inner_blks[0] = interleaved_by;
        if (block_by > 1) {
            md.format_desc.blocking.inner_idxs[1] = I_dim;
            md.format_desc.blocking.inner_blks[1] = block_by;
        }
    }

//    if (arm_compute::is_fixed_format_fast_math(wf)) {
//        md.data_type = dnnl_bf16;
//        info->set_data_type(arm_compute::DataType::BFLOAT16);
//    }

    // The data layout is now determined by the manually set strides
//    info->set_data_layout(arm_compute::DataLayout::UNKNOWN);

    // x is ignored in fixed format kernels
    // y is the leading dimension of b (ldb) in the GEMM d = a*b + c
    //   This is the stride of O_dim in the md
    // z is the batch dimension (not strictly needed if there's only 1 batch)
    //   i.e. how much do I need to stride to get to the next matmul (ignoring
    //   the interleaving). Note that we use the innermost_batch_stride
    //   because all the batched dimensions are collapsed (as required by ACL).
    arm_compute::Strides new_strides_in_bytes = info->strides_in_bytes();
    new_strides_in_bytes.set(1, ldb * info->element_size());
    new_strides_in_bytes.set(2, innermost_batch_stride * info->element_size());

    info->init(info->tensor_shape(), info->num_channels(), info->data_type(),
              new_strides_in_bytes, info->offset_first_element_in_bytes(),
              memoryPtr->getDescPtr()->getCurrentMemSize());
}

static void updateFCShapes(ACLMemoryShapes& aclMemoryShapes, bool isTranspose) {
    if (aclMemoryShapes[ACLArgs::ACL_WEI].num_dimensions() == 3U) {
        aclMemoryShapes[ACLArgs::ACL_WEI] = arm_compute::TensorShape(
                {aclMemoryShapes[ACLArgs::ACL_WEI][0] * aclMemoryShapes[ACLArgs::ACL_WEI][1],
                 aclMemoryShapes[ACLArgs::ACL_WEI][2]});
    }

    if (one_of(aclMemoryShapes[ACLArgs::ACL_SRC_0].num_dimensions(), 3U, 4U)) {
        aclMemoryShapes[ACLArgs::ACL_SRC_0] = arm_compute::TensorShape({
            aclMemoryShapes[ACLArgs::ACL_WEI][0],
            aclMemoryShapes[ACLArgs::ACL_SRC_0].total_size() / aclMemoryShapes[ACLArgs::ACL_WEI][0]});
    }

    if (one_of(aclMemoryShapes[ACLArgs::ACL_DST].num_dimensions(), 3U, 4U)) {
        aclMemoryShapes[ACLArgs::ACL_DST] = arm_compute::TensorShape({
            aclMemoryShapes[ACLArgs::ACL_WEI][1],
            aclMemoryShapes[ACLArgs::ACL_SRC_0][1]});
    }

    if (!isTranspose) {
        std::swap(aclMemoryShapes[ACLArgs::ACL_WEI][0], aclMemoryShapes[ACLArgs::ACL_WEI][1]);
    }
}

static MemoryPtr prepareWeightMemory(const MemoryArgs& memory,
                                     const ACLTensorAttrs& aclTensorAttrs,
                                     const ExecutorContext::CPtr context,
                                     const arm_compute::FullyConnectedLayerInfo& fullyConnectedLayerInfo,
                                     arm_compute::WeightsInfo& weightsInfo,
                                     ACLInfo& aclInfo) {
    DEBUG_LOG("ACLFullyConnectedExecutor: prepack weights");
    // Initialize ACL tensors params
    ACLMemoryShapes  aclMemoryShapes;
    ACLMemoryTypes   aclDataType{};
    ACLMemoryLayouts aclDataLayout{};
    for (auto& cpu_mem_ptr : memory) {
        const ACLArgs index = argConvert.at(cpu_mem_ptr.first);
        ACLCommonExecutor::initACLTensorParams(cpu_mem_ptr.second, aclTensorAttrs,
                                               aclMemoryShapes[index],
                                               aclDataType[index],
                                               aclDataLayout[index]);
    }

    // Update ACL tensors shapes
    updateFCShapes(aclMemoryShapes, fullyConnectedLayerInfo.transpose_weights);

    ACLMemoryInfo aclMemoryInfos;
    for (int i = 0; i < ACLArgs::COUNT_OF_ARGS; i++) {
        aclMemoryInfos[i] = ACLCommonExecutor::initTensorInfo(aclMemoryShapes[i], aclDataType[i], aclDataLayout[i]);
    }

    weightsInfo = arm_compute::WeightsInfo(false, 1, 1, aclMemoryShapes[ACLArgs::ACL_SRC_0][0], false, arm_compute::WeightFormat::ANY);
    arm_compute::WeightFormat expectedWeightFormat;
    auto status = arm_compute::NEFullyConnectedLayer::has_opt_impl(
            expectedWeightFormat,
            aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
            aclMemoryInfos[ACLArgs::ACL_WEI].get(),
            aclMemoryInfos[ACLArgs::ACL_BIAS].get(),
            aclMemoryInfos[ACLArgs::ACL_DST].get(),
            fullyConnectedLayerInfo,
            weightsInfo);

    aclInfo = std::make_shared<arm_compute::TensorInfo>(*aclMemoryInfos[ACLArgs::ACL_WEI].get());
    dnnl::impl::memory_desc_t memoryDesc;
    memoryDesc.ndims = 2;
    memoryDesc.dims[0] = aclMemoryShapes[ACLArgs::ACL_WEI][0];
    memoryDesc.dims[1] = aclMemoryShapes[ACLArgs::ACL_WEI][1];
    memoryDesc.data_type = dnnl_data_type_t::dnnl_f32;
    reorder_to_weight_format(aclInfo,
                             memoryDesc,
                             memory.at(ARG_WEI),
                             expectedWeightFormat,
                             0, 1, {}, {});

    if (!status) {
        DEBUG_LOG("ACL operator validation was failed: ", status.error_description());
        return {};
    }
    arm_compute::NEReorderLayer neReorderLayer;
    arm_compute::NEReorderLayer::validate(
            aclMemoryInfos[ACLArgs::ACL_WEI].get(),
            aclInfo.get(),
            arm_compute::WeightFormat::OHWI,
            expectedWeightFormat);

    auto create = [&]() {
        auto inWeightsTensor = ACLCommonExecutor::initTensor(aclMemoryInfos[ACLArgs::ACL_WEI]);
        auto outWeightsTensor = ACLCommonExecutor::initTensor(aclInfo);
        neReorderLayer.configure(
                inWeightsTensor.get(),
                outWeightsTensor.get(),
                arm_compute::WeightFormat::OHWI,
                expectedWeightFormat);

        inWeightsTensor->allocator()->import_memory(memory.at(ARG_WEI)->getDataAs<float>());
        MemoryPtr _ptr = std::make_shared<Memory>(context->getEngine(),
                                                  intel_cpu::CpuBlockedMemoryDesc(
                                                          ov::element::f32,
                                                          intel_cpu::Shape{aclMemoryShapes[ACLArgs::ACL_WEI][0],
                                                                           aclMemoryShapes[ACLArgs::ACL_WEI][1]}));
        outWeightsTensor->allocator()->import_memory(_ptr->getDataAs<float>());
        neReorderLayer.run();
        DEBUG_LOG("ACLFullyConnectedExecutor: cache miss, perform packing");
        return _ptr;
    };

    weightsInfo.set_weight_format(expectedWeightFormat);

    auto weightCache = context->getWeightsCache();
    if (weightCache != nullptr) {
        std::string format = "fc_acl_" +
                std::to_string(aclMemoryShapes[ACLArgs::ACL_WEI][0]) + "_" +
                std::to_string(aclMemoryShapes[ACLArgs::ACL_WEI][1]);
        const std::string string_hash = format + "_" + std::to_string(memory.at(ARG_WEI)->getSize()) + "_" +
                                        std::to_string(reinterpret_cast<uint64_t>(memory.at(ARG_WEI)->getData()));
        DEBUG_LOG("ACLFullyConnectedExecutor: findOrCreate, string_hash: ", string_hash);
        return *weightCache->findOrCreate(string_hash, create);
    }

    DEBUG_LOG("ACLFullyConnectedExecutor: Weights cache is not available");
    return create();
}

ACLFullyConnectedExecutor::ACLFullyConnectedExecutor(const FCAttrs &attrs, const PostOps &postOps,
                                                     const MemoryArgs &memory,
                                                     const ExecutorContext::CPtr context) {
    aclTensorAttrs.hasLayoutTypeNHWC = memory.at(ARG_SRC)->getDescPtr()->hasLayoutType(LayoutType::nspc);
    fullyConnectedLayerInfo.weights_trained_layout = getAclDataLayoutByMemoryDesc(memory.at(ARG_WEI)->getDescPtr());
    fullyConnectedLayerInfo.transpose_weights = !attrs.weightsNonTransposed;

    // Add postops
    if (!postOps.empty() && postOps.size() == 1) {
        if (const auto activation = std::dynamic_pointer_cast<ActivationPostOp>(postOps[0])) {
            fullyConnectedLayerInfo.activation_info = getActivationLayerInfo(convertToEltwiseAlgorithm(activation->type()),
                                                                             activation->alpha(),
                                                                             activation->beta(),
                                                                             activation->gamma());
        }
    }
    packedWeights = prepareWeightMemory(memory,
                                        aclTensorAttrs,
                                        context,
                                        fullyConnectedLayerInfo,
                                        weightsInfo,
                                        newWeightsInfo);
}

bool ACLFullyConnectedExecutor::supports(const FCConfig &config) {
    VERIFY(one_of(srcType(config), ov::element::f16, ov::element::f32), UNSUPPORTED_SRC_PRECISIONS);
    VERIFY(postOpsNumbers(config) < 2,          UNSUPPORTED_NUMBER_OF_POSTOPS);
    VERIFY(one_of(srcRank(config), 2U, 3U, 4U), UNSUPPORTED_SRC_RANK);
    VERIFY(one_of(weiRank(config), 2U, 3U),     UNSUPPORTED_WEI_RANK);
    return true;
}

void ACLFullyConnectedExecutor::updateTensorsShapes(ACLMemoryShapes& aclMemoryShapes) {
    updateFCShapes(aclMemoryShapes, fullyConnectedLayerInfo.transpose_weights);
}

arm_compute::Status ACLFullyConnectedExecutor::validateTensorsInfo(const ACLMemoryInfo & aclMemoryInfos) {
    *aclMemoryInfos[ACLArgs::ACL_WEI] = *newWeightsInfo;
    return arm_compute::NEFullyConnectedLayer::validate(
            aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
            aclMemoryInfos[ACLArgs::ACL_WEI].get(),
            aclMemoryInfos[ACLArgs::ACL_BIAS].get(),
            aclMemoryInfos[ACLArgs::ACL_DST].get(),
            fullyConnectedLayerInfo,
            weightsInfo);
}

ACLFunction ACLFullyConnectedExecutor::configureFunction(const ACLMemoryTensors & aclMemoryTensors) {
    auto neFC = std::make_unique<arm_compute::NEFullyConnectedLayer>();
    neFC->configure(
            aclMemoryTensors[ACLArgs::ACL_SRC_0].get(),
            aclMemoryTensors[ACLArgs::ACL_WEI].get(),
            aclMemoryTensors[ACLArgs::ACL_BIAS].get(),
            aclMemoryTensors[ACLArgs::ACL_DST].get(),
            fullyConnectedLayerInfo,
            weightsInfo);
    return neFC;
}

}   // namespace intel_cpu
}   // namespace ov
