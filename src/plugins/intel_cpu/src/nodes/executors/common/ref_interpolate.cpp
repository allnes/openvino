// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ref_interpolate.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <iostream>

#include "openvino/core/parallel.hpp"
#include "utils/general_utils.h"
#include "nodes/common/cpu_memcpy.h"

namespace ov {
namespace intel_cpu {

namespace {
// Helper functions
static inline bool isFloatCompatible(ov::element::Type prc) {
    return prc == ov::element::f32 || prc == ov::element::bf16 || 
           prc == ov::element::f16 || prc == ov::element::f64;
}

static VectorDims getPaddedInputShape(const VectorDims& srcDims,
                                     const std::vector<int>& padBegin,
                                     const std::vector<int>& padEnd) {
    VectorDims paddedShape;
    for (size_t i = 0; i < srcDims.size(); i++) {
        paddedShape.push_back(srcDims[i] + padBegin[i] + padEnd[i]);
    }
    return paddedShape;
}

template <typename T>
inline T rnd_up(const T a, const T b) {
    return (a + b - 1) / b * b;
}

static size_t getSpatialDimsNum(size_t rank) {
    switch (rank) {
        case 1:
        case 3:
            return 1;
        case 2:
        case 4:
            return 2;
        case 5:
            return 3;
        default:
            OPENVINO_THROW("Unsupported tensor rank: ", rank);
    }
}

static VectorDims to5Dim(const VectorDims& dims) {
    VectorDims dims5(5, 1);
    size_t rank = dims.size();
    
    dims5[4] = dims[rank - 1];  // width
    if (rank > 1) dims5[3] = dims[rank - 2];  // height
    if (rank > 2) dims5[0] = dims[0];  // batch
    if (rank > 3) dims5[1] = dims[1];  // channels
    if (rank > 4) dims5[2] = dims[2];  // depth
    
    if (rank == 3) {  // nhw -> ncw
        dims5[1] = dims5[3];
        dims5[3] = 1;
    }
    
    return dims5;
}
}  // namespace

RefInterpolateExecutor::RefInterpolateExecutor(const InterpolateAttrs& attrs,
                                             const PostOpsPtr& postOps,
                                             const MemoryArgs& memory,
                                             const ExecutorContext::CPtr context)
    : m_attrs(attrs), m_postOps(postOps) {
    // Initialize based on attrs and memory descriptors
    const auto& srcDesc = memory.at(ARG_SRC)->getDescPtr();
    const auto& dstDesc = memory.at(ARG_DST)->getDescPtr();
    
    const auto& srcDims = srcDesc->getShape().getStaticDims();
    const auto& dstDims = dstDesc->getShape().getStaticDims();
    
    dataRank = srcDims.size();
    srcDataSize = srcDesc->getPrecision().size();
    
    // Store input/output precision
    m_attrs.inPrc = srcDesc->getPrecision();
    m_attrs.outPrc = dstDesc->getPrecision();
    
    // std::cout << "RefInterpolateExecutor: srcDataSize=" << srcDataSize 
    //           << " inPrc=" << m_attrs.inPrc << " outPrc=" << m_attrs.outPrc 
    //           << " NCHWAsNHWC=" << m_attrs.NCHWAsNHWC << std::endl;
    
    // Set default layout if not specified
    if (m_attrs.layout != InterpolateLayoutType::planar && 
        m_attrs.layout != InterpolateLayoutType::block && 
        m_attrs.layout != InterpolateLayoutType::by_channel) {
        m_attrs.layout = InterpolateLayoutType::planar;
    }
    
    // Convert to 5D for unified processing
    srcDimPad5d = to5Dim(getPaddedInputShape(srcDims, m_attrs.padBegin, m_attrs.padEnd));
    dstDim5d = to5Dim(dstDims);
    
    spatialDimSize = getSpatialDimsNum(dataRank);
    
    // Build interpolation tables based on mode
    std::vector<float> dataScales = m_attrs.dataScales;
    if (dataScales.empty()) {
        // Compute scales from dimensions - need one scale per dimension in the original tensor
        dataScales.resize(dataRank, 1.0f);
        
        // For 5D internal representation (B,C,D,H,W), map back to original tensor dimensions
        if (dataRank == 5) {
            dataScales[0] = 1.0f; // Batch - not scaled
            dataScales[1] = 1.0f; // Channel - not scaled
            dataScales[2] = (srcDimPad5d[2] > 0) ? static_cast<float>(dstDim5d[2]) / static_cast<float>(srcDimPad5d[2]) : 1.0f; // Depth
            dataScales[3] = (srcDimPad5d[3] > 0) ? static_cast<float>(dstDim5d[3]) / static_cast<float>(srcDimPad5d[3]) : 1.0f; // Height
            dataScales[4] = (srcDimPad5d[4] > 0) ? static_cast<float>(dstDim5d[4]) / static_cast<float>(srcDimPad5d[4]) : 1.0f; // Width
        } else if (dataRank == 4) {
            dataScales[0] = 1.0f; // Batch - not scaled
            dataScales[1] = (srcDimPad5d[1] > 0) ? static_cast<float>(dstDim5d[1]) / static_cast<float>(srcDimPad5d[1]) : 1.0f; // Channel
            dataScales[2] = (srcDimPad5d[3] > 0) ? static_cast<float>(dstDim5d[3]) / static_cast<float>(srcDimPad5d[3]) : 1.0f; // Height  
            dataScales[3] = (srcDimPad5d[4] > 0) ? static_cast<float>(dstDim5d[4]) / static_cast<float>(srcDimPad5d[4]) : 1.0f; // Width
            // std::cout << "Calculated scales: [0]=" << dataScales[0] << " [1]=" << dataScales[1] 
            //           << " [2]=" << dataScales[2] << " [3]=" << dataScales[3] << std::endl;
        } else if (dataRank == 3) {
            dataScales[0] = 1.0f; // Batch - not scaled
            dataScales[1] = (srcDimPad5d[3] > 0) ? static_cast<float>(dstDim5d[3]) / static_cast<float>(srcDimPad5d[3]) : 1.0f; // Height
            dataScales[2] = (srcDimPad5d[4] > 0) ? static_cast<float>(dstDim5d[4]) / static_cast<float>(srcDimPad5d[4]) : 1.0f; // Width
        } else if (dataRank == 2) {
            dataScales[0] = (srcDimPad5d[3] > 0) ? static_cast<float>(dstDim5d[3]) / static_cast<float>(srcDimPad5d[3]) : 1.0f; // Height
            dataScales[1] = (srcDimPad5d[4] > 0) ? static_cast<float>(dstDim5d[4]) / static_cast<float>(srcDimPad5d[4]) : 1.0f; // Width
        } else if (dataRank == 1) {
            dataScales[0] = (srcDimPad5d[4] > 0) ? static_cast<float>(dstDim5d[4]) / static_cast<float>(srcDimPad5d[4]) : 1.0f; // Width
        }
    }
    
    if (m_attrs.mode == InterpolateMode::nearest) {
        buildTblNN(srcDimPad5d, dstDim5d, dataScales);
    } else if (m_attrs.mode == InterpolateMode::linear_onnx) {
        buildTblLinearOnnx(srcDimPad5d, dstDim5d, dataScales);
    } else if (m_attrs.mode == InterpolateMode::linear) {
        // For now, use linear_onnx tables for linear mode
        // TODO: Implement proper buildTblLinear with triangular kernel
        buildTblLinearOnnx(srcDimPad5d, dstDim5d, dataScales);
    } else if (m_attrs.mode == InterpolateMode::cubic) {
        buildTblCubic(srcDimPad5d, dstDim5d, dataScales);
    } else if (m_attrs.mode == InterpolateMode::bilinear_pillow || 
               m_attrs.mode == InterpolateMode::bicubic_pillow) {
        buildTblPillow(srcDimPad5d, dstDim5d, dataScales);
    }
}

bool RefInterpolateExecutor::update(const MemoryArgs& memory) {
    // Update logic if needed when memory changes
    return true;
}

void RefInterpolateExecutor::execute(const MemoryArgs& memory) {
    const auto& srcMem = memory.at(ARG_SRC);
    const auto& dstMem = memory.at(ARG_DST);
    
    const uint8_t* src_data = srcMem->getDataAs<const uint8_t>();
    uint8_t* dst_data = dstMem->getDataAs<uint8_t>();
    
    // Handle padding if needed
    const uint8_t* src_data_ptr = src_data;
    if (m_attrs.hasPad) {
        src_data_ptr = padPreprocess(srcMem, dstMem);
    }
    
    // Execute interpolation based on mode
    if (m_attrs.mode == InterpolateMode::nearest) {
        size_t N = srcDimPad5d[0];
        size_t C = srcDimPad5d[1];
        NNRef(src_data_ptr, dst_data, N, C, srcDimPad5d[2], srcDimPad5d[3], srcDimPad5d[4],
              dstDim5d[2], dstDim5d[3], dstDim5d[4]);
    } else if (m_attrs.mode == InterpolateMode::linear_onnx) {
        size_t N = srcDimPad5d[0];
        size_t C = srcDimPad5d[1];
        linearOnnxRef(src_data_ptr, dst_data, N, C, srcDimPad5d[2], srcDimPad5d[3], srcDimPad5d[4],
                     dstDim5d[2], dstDim5d[3], dstDim5d[4]);
    } else if (m_attrs.mode == InterpolateMode::linear) {
        // For now, use linear_onnx implementation for linear mode
        // TODO: Implement proper linear mode with triangular kernel and antialiasing support
        size_t N = srcDimPad5d[0];
        size_t C = srcDimPad5d[1];
        linearOnnxRef(src_data_ptr, dst_data, N, C, srcDimPad5d[2], srcDimPad5d[3], srcDimPad5d[4],
                     dstDim5d[2], dstDim5d[3], dstDim5d[4]);
    } else if (m_attrs.mode == InterpolateMode::cubic) {
        size_t N = srcDimPad5d[0];
        size_t C = srcDimPad5d[1];
        cubicRef(src_data_ptr, dst_data, N, C, srcDimPad5d[3], srcDimPad5d[4],
                dstDim5d[3], dstDim5d[4]);
    } else if (m_attrs.mode == InterpolateMode::bilinear_pillow || 
               m_attrs.mode == InterpolateMode::bicubic_pillow) {
        size_t N = srcDimPad5d[0];
        size_t C = srcDimPad5d[1];
        
        if (m_attrs.NCHWAsNHWC && dataRank == 4) {
            // For NCHWAsNHWC with 4D tensor and axes=(1,2):
            // Physical layout is NCHW but we're scaling the C and H dimensions (axes 1,2)
            // The pillowRefNCHWAsNHWC function should handle scaling of physical C and H dimensions
            // Parameters: (B, logicalC, logicalIH, logicalIW, logicalOH, logicalOW)
            // For axes=(1,2), we scale C: src[1] -> dst[1] and H: src[3] -> dst[3]
            int logicalC = srcDimPad5d[4];  // physical W (constant) -> 3
            int logicalIH = srcDimPad5d[1]; // physical C (to be scaled) -> 4
            int logicalIW = srcDimPad5d[3]; // physical H (to be scaled) -> 4  
            int logicalOH = dstDim5d[1];    // physical C (scaled) -> 8
            int logicalOW = dstDim5d[3];    // physical H (scaled) -> 16
            
            // std::cout << "NCHWAsNHWC pillow: physical src(" << N << "," << srcDimPad5d[1] << ","
            //           << srcDimPad5d[3] << "," << srcDimPad5d[4] << ") -> dst(" << N << "," 
            //           << dstDim5d[1] << "," << dstDim5d[3] << "," << dstDim5d[4] << ")" << std::endl;
            // std::cout << "Logical mapping: C=" << logicalC << " IH=" << logicalIH 
            //           << " IW=" << logicalIW << " OH=" << logicalOH << " OW=" << logicalOW << std::endl;
            
            pillowRefNCHWAsNHWC(src_data_ptr, dst_data, N, 
                               logicalC, logicalIH, logicalIW,
                               logicalOH, logicalOW);
        } else {
            pillowRef(src_data_ptr, dst_data, N, C, srcDimPad5d[3], srcDimPad5d[4],
                     dstDim5d[3], dstDim5d[4]);
        }
    } else {
        // For unsupported modes, fallback to old implementation
        OPENVINO_THROW("RefInterpolateExecutor: unsupported mode: " + std::to_string(static_cast<int>(m_attrs.mode)));
    }
}

// Helper functions implementation
void RefInterpolateExecutor::buildTblNN(const VectorDims& srcDimPad5d, const VectorDims& dstDim5d,
                                       const std::vector<float>& dataScales) {
    const int dimSize = dataRank;
    // The coordTransToInput function expects the inverse scale (srcDim/dstDim), not (dstDim/srcDim)
    float fz = (dimSize == 5 && dataScales.size() > 0) ? (1.0f / dataScales[0]) : 1.f;
    float fy = (dataScales.size() > dimSize - 2) ? (1.0f / dataScales[dimSize - 2]) : 1.f;
    float fx = (dataScales.size() > dimSize - 1) ? (1.0f / dataScales[dimSize - 1]) : 1.f;
    
    size_t ID = srcDimPad5d[2];
    size_t IH = srcDimPad5d[3];
    size_t IW = srcDimPad5d[4];
    size_t OD = dstDim5d[2];
    size_t OH = dstDim5d[3];
    size_t OW = dstDim5d[4];
    
    auxTable.resize(OD + OH + OW);
    
    for (size_t oz = 0; oz < OD; oz++) {
        float iz = coordTransToInput(oz, fz, ID, OD);
        // fz is inverted scale (srcDim/dstDim), so downsample when fz > 1
        int idx = nearestRound(iz, fz > 1, m_attrs.nearestMode);
        auxTable[oz] = std::max(0, std::min(idx, static_cast<int>(ID) - 1));
    }
    
    for (size_t oy = 0; oy < OH; oy++) {
        float iy = coordTransToInput(oy, fy, IH, OH);
        // fy is inverted scale (srcDim/dstDim), so downsample when fy > 1
        int idx = nearestRound(iy, fy > 1, m_attrs.nearestMode);
        auxTable[OD + oy] = std::max(0, std::min(idx, static_cast<int>(IH) - 1));
    }
    
    for (size_t ox = 0; ox < OW; ox++) {
        float ix = coordTransToInput(ox, fx, IW, OW);
        // fx is inverted scale (srcDim/dstDim), so downsample when fx > 1
        int idx = nearestRound(ix, fx > 1, m_attrs.nearestMode);
        auxTable[OD + OH + ox] = std::max(0, std::min(idx, static_cast<int>(IW) - 1));
    }
    
    // Debug output - commented out for performance
    // std::cout << "buildTblNN: ID=" << ID << " IH=" << IH << " IW=" << IW << " OD=" << OD << " OH=" << OH << " OW=" << OW << std::endl;
    // std::cout << "auxTable first 10 elements: ";
    // for (size_t i = 0; i < std::min(auxTable.size(), size_t(10)); i++) {
    //     std::cout << auxTable[i] << " ";
    // }
    // std::cout << std::endl;
}

void RefInterpolateExecutor::buildTblLinearOnnx(const VectorDims& srcDimPad5d, const VectorDims& dstDim5d,
                                               const std::vector<float>& dataScales) {
    int dimSize = dataRank;
    // The coordTransToInput function expects the inverse scale (srcDim/dstDim), not (dstDim/srcDim)
    float fz = (spatialDimSize > 2) ? (1.0f / dataScales[dimSize - 3]) : 1.F;
    float fy = (spatialDimSize > 1) ? (1.0f / dataScales[dimSize - 2]) : 1.F;
    float fx = 1.0f / dataScales[dimSize - 1];
    int ID = srcDimPad5d[2];
    int IH = srcDimPad5d[3];
    int IW = srcDimPad5d[4];
    int OD = dstDim5d[2];
    int OH = dstDim5d[3];
    int OW = dstDim5d[4];

    // FrontTopLeft:0, FrontTopRight:1, FrontBottomLeft:2, FrontBottomRight:3,
    // EndTopLeft:4,   EndTopRight:5,   EndBottomLeft:6,   EndBottomRight:7
    // weight: Left:0, right:1, top:2, bottom:3, front:4, end:5
    int eltInGrid = [&]() {
        if (spatialDimSize > 2) {
            return MAX_INPUT_INTERPOLATE;
        }
        if (spatialDimSize > 1) {
            return 4;
        }
        return 2;
    }();
    int idxType = 2;
    int scratchLen = rnd_up(eltInGrid * OW * OH * OD, 16);
    auxTable.resize(idxType * scratchLen);

    // For planar layout, we need to calculate full indices
    // On x86 with SSE41, indices are multiplied by srcDataSize for byte addressing
    // On other platforms (like ARM), indices are element-based
    const int scale = 1; // For ref executor, use element-based indexing
    
    parallel_for3d(OD, OH, OW, [&](int oz, int oy, int ox) {
        int idxOz = oz * OH * OW;
        int idxOzOy = idxOz + oy * OW;
        int idxOzOyOx = idxOzOy + ox;
        
        int ixL = 0, ixR = 0;
        float weightL = 0.f, weightR = 0.f;
        linearOnnxCF(ox, fx, IW, OW, ixL, ixR, weightL, weightR);
        
        int iyT = 0, iyB = 0;
        float weightT = 0.f, weightB = 0.f;
        if (spatialDimSize > 1) {
            linearOnnxCF(oy, fy, IH, OH, iyT, iyB, weightT, weightB);
        }
        
        int izF = 0, izE = 0;
        float weightF = 0.f, weightE = 0.f;
        if (spatialDimSize > 2) {
            linearOnnxCF(oz, fz, ID, OD, izF, izE, weightF, weightE);
        }
        
        // Store indices as full offsets multiplied by scale
        auxTable[idxOzOyOx] = (izF * IH * IW + iyT * IW + ixL) * scale;
        auxTable[OW * OH * OD + idxOzOyOx] = (izF * IH * IW + iyT * IW + ixR) * scale;
        reinterpret_cast<float&>(auxTable[scratchLen + idxOzOyOx]) = weightL;
        reinterpret_cast<float&>(auxTable[scratchLen + OW * OH * OD + idxOzOyOx]) = weightR;
        
        if (spatialDimSize > 1) {
            auxTable[2 * OW * OH * OD + idxOzOyOx] = (izF * IH * IW + iyB * IW + ixL) * scale;
            auxTable[3 * OW * OH * OD + idxOzOyOx] = (izF * IH * IW + iyB * IW + ixR) * scale;
            reinterpret_cast<float&>(auxTable[scratchLen + 2 * OW * OH * OD + idxOzOyOx]) = weightT;
            reinterpret_cast<float&>(auxTable[scratchLen + 3 * OW * OH * OD + idxOzOyOx]) = weightB;
        }
        
        if (spatialDimSize > 2) {
            auxTable[4 * OW * OH * OD + idxOzOyOx] = (izE * IH * IW + iyT * IW + ixL) * scale;
            auxTable[5 * OW * OH * OD + idxOzOyOx] = (izE * IH * IW + iyT * IW + ixR) * scale;
            auxTable[6 * OW * OH * OD + idxOzOyOx] = (izE * IH * IW + iyB * IW + ixL) * scale;
            auxTable[7 * OW * OH * OD + idxOzOyOx] = (izE * IH * IW + iyB * IW + ixR) * scale;
            reinterpret_cast<float&>(auxTable[scratchLen + 4 * OW * OH * OD + idxOzOyOx]) = weightF;
            reinterpret_cast<float&>(auxTable[scratchLen + 5 * OW * OH * OD + idxOzOyOx]) = weightE;
        }
    });
}

void RefInterpolateExecutor::NNRef(const uint8_t* in_ptr_, uint8_t* out_ptr_,
                                  int B, int C, int ID, int IH, int IW, int OD, int OH, int OW) {
    const float* in_ptr_f32 = reinterpret_cast<const float*>(in_ptr_);
    float* out_ptr_f32 = reinterpret_cast<float*>(out_ptr_);
    
    const int* index_d = &auxTable[0];
    const int* index_h = &auxTable[OD];
    const int* index_w = &auxTable[OD + OH];
    
    parallel_for3d(B, C, OD, [&](size_t b, size_t c, size_t od) {
        int input_d_idx = index_d[od];
        const float* in_ptr = in_ptr_f32 + (b * C * ID * IH * IW + c * ID * IH * IW + input_d_idx * IH * IW);
        float* out_ptr = out_ptr_f32 + (b * C * OD * OH * OW + c * OD * OH * OW + od * OH * OW);
        
        for (int oh = 0; oh < OH; oh++) {
            int input_h_idx = index_h[oh];
            const float* in_ptr_h = in_ptr + (input_h_idx * IW);
            float* out_ptr_h = out_ptr + (oh * OW);
            for (int ow = 0; ow < OW; ow++) {
                int input_w_idx = index_w[ow];
                out_ptr_h[ow] = in_ptr_h[input_w_idx];
            }
        }
    });
}

void RefInterpolateExecutor::linearOnnxRef(const uint8_t* in_ptr_, uint8_t* out_ptr_,
                                          int B, int C, int ID, int IH, int IW, int OD, int OH, int OW) {
    std::vector<int*> indexPtr(MAX_INPUT_INTERPOLATE, nullptr);
    std::vector<float*> weightPtr(MAX_INPUT_INTERPOLATE, nullptr);
    // FrontTopLeft:0, FrontTopRight:1, FrontBottomLeft:2, FrontBottomRight:3,
    // EndTopLeft:4,   EndTopRight:5,   EndBottomLeft:6,   EndBottomRight:7
    // weight: Left:0, right:1, top:2, bottom:3, front:4, end:5

    int eltInGrid = [&]() {
        if (spatialDimSize > 2) {
            return MAX_INPUT_INTERPOLATE;
        }
        if (spatialDimSize > 1) {
            return 4;
        }
        return 2;
    }();
    int scratchLen = rnd_up(eltInGrid * OW * OH * OD, 16);

    indexPtr[0] = auxTable.data();
    indexPtr[1] = &auxTable[OW * OH * OD];
    weightPtr[0] = reinterpret_cast<float*>(&auxTable[scratchLen]);
    weightPtr[1] = reinterpret_cast<float*>(&auxTable[scratchLen + OW * OH * OD]);
    if (spatialDimSize > 1) {
        indexPtr[2] = &auxTable[2 * OW * OH * OD];
        indexPtr[3] = &auxTable[3 * OW * OH * OD];
        weightPtr[2] = reinterpret_cast<float*>(&auxTable[scratchLen + 2 * OW * OH * OD]);
        weightPtr[3] = reinterpret_cast<float*>(&auxTable[scratchLen + 3 * OW * OH * OD]);
    }
    if (spatialDimSize > 2) {
        indexPtr[4] = &auxTable[4 * OW * OH * OD];
        indexPtr[5] = &auxTable[5 * OW * OH * OD];
        indexPtr[6] = &auxTable[6 * OW * OH * OD];
        indexPtr[7] = &auxTable[7 * OW * OH * OD];
        weightPtr[4] = reinterpret_cast<float*>(&auxTable[scratchLen + 4 * OW * OH * OD]);
        weightPtr[5] = reinterpret_cast<float*>(&auxTable[scratchLen + 5 * OW * OH * OD]);
    }

    const auto* in_ptr_f32 = reinterpret_cast<const float*>(in_ptr_);
    auto* out_ptr_f32 = reinterpret_cast<float*>(out_ptr_);
    
    // Debug output
    // std::cout << "linearOnnxRef: B=" << B << " C=" << C << " ID=" << ID << " IH=" << IH 
    //           << " IW=" << IW << " OD=" << OD << " OH=" << OH << " OW=" << OW 
    //           << " spatialDimSize=" << spatialDimSize << std::endl;

    // Check for planar layout (NCHW)
    if (m_attrs.layout == InterpolateLayoutType::planar) {
        parallel_for2d(B, C, [&](size_t b, size_t c) {
            const float* in_ptr_nc = in_ptr_f32 + (b * C * ID * IH * IW + c * ID * IH * IW);
            float* out_ptr_nc = out_ptr_f32 + (b * C * OD * OH * OW + c * OD * OH * OW);
            
            // Use the same algorithm as original code
            switch (spatialDimSize) {
            case 1:
                for (int i = 0; i < OW; i++) {
                    float src0 = in_ptr_nc[indexPtr[0][i]];
                    float src1 = in_ptr_nc[indexPtr[1][i]];
                    out_ptr_nc[i] = src0 * weightPtr[0][i] + src1 * weightPtr[1][i];
                }
                break;
            case 2:
                for (int i = 0; i < OH * OW; i++) {
                    float src00 = in_ptr_nc[indexPtr[0][i]];
                    float src01 = in_ptr_nc[indexPtr[1][i]];
                    float src10 = in_ptr_nc[indexPtr[2][i]];
                    float src11 = in_ptr_nc[indexPtr[3][i]];
                    
                    out_ptr_nc[i] = src00 * weightPtr[2][i] * weightPtr[0][i] + 
                                    src01 * weightPtr[2][i] * weightPtr[1][i] +
                                    src10 * weightPtr[3][i] * weightPtr[0][i] + 
                                    src11 * weightPtr[3][i] * weightPtr[1][i];
                }
                break;
            case 3:
                for (int i = 0; i < OD * OH * OW; i++) {
                    float src000 = in_ptr_nc[indexPtr[0][i]];
                    float src001 = in_ptr_nc[indexPtr[1][i]];
                    float src010 = in_ptr_nc[indexPtr[2][i]];
                    float src011 = in_ptr_nc[indexPtr[3][i]];
                    float src100 = in_ptr_nc[indexPtr[4][i]];
                    float src101 = in_ptr_nc[indexPtr[5][i]];
                    float src110 = in_ptr_nc[indexPtr[6][i]];
                    float src111 = in_ptr_nc[indexPtr[7][i]];
                    
                    out_ptr_nc[i] = 
                        weightPtr[4][i] * (weightPtr[2][i] * (weightPtr[0][i] * src000 + weightPtr[1][i] * src001) +
                                           weightPtr[3][i] * (weightPtr[0][i] * src010 + weightPtr[1][i] * src011)) +
                        weightPtr[5][i] * (weightPtr[2][i] * (weightPtr[0][i] * src100 + weightPtr[1][i] * src101) +
                                           weightPtr[3][i] * (weightPtr[0][i] * src110 + weightPtr[1][i] * src111));
                }
                break;
            }
        });
    } else {
        // For non-planar layouts, assume it's similar to by_channel
        // This needs to be implemented based on the specific layout
        OPENVINO_THROW("Non-planar layouts not yet implemented in RefInterpolateExecutor");
    }
}

float RefInterpolateExecutor::coordTransToInput(int outCoord, float scale, int inShape, int outShape) const {
    if (scale == 1.0f || (inShape == outShape)) {
        return outCoord;
    }
    
    switch (m_attrs.coordTransMode) {
        case InterpolateCoordTransMode::half_pixel:
            return (outCoord + 0.5f) * scale - 0.5f;
        case InterpolateCoordTransMode::pytorch_half_pixel:
            if (outShape > 1)
                return (outCoord + 0.5f) * scale - 0.5f;
            else
                return 0;
        case InterpolateCoordTransMode::asymmetric:
            return static_cast<float>(outCoord) * scale;
        case InterpolateCoordTransMode::tf_half_pixel_for_nn:
            return (outCoord + 0.5f) * scale;
        case InterpolateCoordTransMode::align_corners:
            if (outShape > 1)
                return outCoord * (static_cast<float>(inShape - 1) / (outShape - 1));
            else
                return 0;
        default:
            OPENVINO_THROW("RefInterpolateExecutor: unsupported coordinate transformation mode");
    }
}

int RefInterpolateExecutor::nearestRound(float origin, bool isDownsample, InterpolateNearestMode nearestMode) {
    switch (nearestMode) {
        case InterpolateNearestMode::round_prefer_floor:
            if (origin == static_cast<int>(origin) + 0.5f)
                return static_cast<int>(std::floor(origin));
            else
                return static_cast<int>(std::round(origin));
        case InterpolateNearestMode::round_prefer_ceil:
            if (origin == static_cast<int>(origin) + 0.5f)
                return static_cast<int>(std::ceil(origin));
            else
                return static_cast<int>(std::round(origin));
        case InterpolateNearestMode::floor:
            return static_cast<int>(std::floor(origin));
        case InterpolateNearestMode::ceil:
            return static_cast<int>(std::ceil(origin));
        case InterpolateNearestMode::simple:
            if (isDownsample)
                return static_cast<int>(std::ceil(origin));
            else
                return static_cast<int>(origin);
        default:
            OPENVINO_THROW("RefInterpolateExecutor: unsupported nearest mode");
    }
}

void RefInterpolateExecutor::linearOnnxCF(int outCoord,
                                         float scale,
                                         int inShape,
                                         int outShape,
                                         int& index0,
                                         int& index1,
                                         float& weight0,
                                         float& weight1) {
    float inCoord = coordTransToInput(outCoord, scale, inShape, outShape);
    inCoord = std::max(0.0f, std::min(inCoord, static_cast<float>(inShape - 1)));
    index0 = std::min(static_cast<int>(inCoord), inShape - 1);
    index1 = std::min(index0 + 1, inShape - 1);

    weight1 = std::fabs(inCoord - index0);
    weight0 = std::fabs(inCoord - index1);
    if (index0 == index1) {
        weight0 = 0.5f;
        weight1 = 0.5f;
    }
}

const uint8_t* RefInterpolateExecutor::padPreprocess(const MemoryCPtr& src, const MemoryPtr& dst) {
    const auto& srcDims = src->getStaticDims();
    const auto& srcPrec = src->getPrecision();
    const uint8_t* src_data = src->getDataAs<const uint8_t>();
    
    VectorDims inDimsPad = getPaddedInputShape(srcDims, m_attrs.padBegin, m_attrs.padEnd);
    
    size_t padded_size = std::accumulate(inDimsPad.begin(), inDimsPad.end(), 
                                        srcPrec.size(), std::multiplies<size_t>());
    
    m_padded_input.resize(padded_size, 0);
    
    // Convert to 5D for unified processing
    const auto srcDim5d = to5Dim(srcDims);
    const auto srcDimPad5d = to5Dim(inDimsPad);
    const auto srcDataSize = srcPrec.size();
    size_t dimSize = srcDims.size();
    
    int padB0 = (dimSize > 2) ? m_attrs.padBegin[0] : 0;
    int padB1 = (dimSize > 2) ? m_attrs.padBegin[1] : 0;
    int padB2 = (dimSize == 5) ? m_attrs.padBegin[dimSize - 3] : 0;
    int padB3 = m_attrs.padBegin[dimSize - 2];
    int padB4 = m_attrs.padBegin[dimSize - 1];
    
    // Calculate strides for source and padded tensors
    std::vector<size_t> srcStrides(5), padStrides(5);
    srcStrides[4] = 1;
    padStrides[4] = 1;
    for (int i = 3; i >= 0; i--) {
        srcStrides[i] = srcStrides[i + 1] * srcDim5d[i + 1];
        padStrides[i] = padStrides[i + 1] * srcDimPad5d[i + 1];
    }
    
    // Copy data with padding
    parallel_for4d(srcDim5d[0], srcDim5d[1], srcDim5d[2], srcDim5d[3], [&](int n, int c, int d, int h) {
        const uint8_t* src_ptr = src_data + (n * srcStrides[0] + c * srcStrides[1] + 
                                            d * srcStrides[2] + h * srcStrides[3]) * srcDataSize;
        uint8_t* dst_ptr = m_padded_input.data() + ((n + padB0) * padStrides[0] + (c + padB1) * padStrides[1] +
                                                    (d + padB2) * padStrides[2] + (h + padB3) * padStrides[3] + 
                                                    padB4) * srcDataSize;
        cpu_memcpy(dst_ptr, src_ptr, srcDim5d[4] * srcDataSize);
    });
    
    return m_padded_input.data();
}

void RefInterpolateExecutor::buildTblCubic(const VectorDims& srcDimPad5d, const VectorDims& dstDim5d,
                                          const std::vector<float>& dataScales) {
    int dimSize = dataRank;
    float fy = (dataScales.size() > dimSize - 2) ? (1.0f / dataScales[dimSize - 2]) : 1.f;
    float fx = (dataScales.size() > dimSize - 1) ? (1.0f / dataScales[dimSize - 1]) : 1.f;
    int IH = srcDimPad5d[3];
    int IW = srcDimPad5d[4];
    int OH = dstDim5d[3];
    int OW = dstDim5d[4];

    const int idxNum = 1;
    auxTable.resize((CUBIC_GRID_LEN + idxNum) * OW + (CUBIC_GRID_LEN + idxNum) * OH);
    
    auto xOrigin = auxTable.data();
    auto xFactor = reinterpret_cast<float*>(&auxTable[OW]);
    auto yOrigin = &auxTable[(CUBIC_GRID_LEN + idxNum) * OW];
    auto yFactor = reinterpret_cast<float*>(&auxTable[(CUBIC_GRID_LEN + idxNum) * OW + OH]);

    for (int ox = 0; ox < OW; ox++) {
        float ix = coordTransToInput(ox, fx, IW, OW);
        int ix_r = static_cast<int>(std::floor(ix));
        xOrigin[ox] = ix_r;
        float dx = ix - ix_r;
        
        std::vector<float> coffes_x = getCubicCoeffs(dx, m_attrs.cubeCoeff);
        for (int i = 0; i < CUBIC_GRID_LEN; i++) {
            xFactor[ox * CUBIC_GRID_LEN + i] = coffes_x[i];
        }
    }

    for (int oy = 0; oy < OH; oy++) {
        float iy = coordTransToInput(oy, fy, IH, OH);
        int iy_r = static_cast<int>(std::floor(iy));
        yOrigin[oy] = iy_r;
        float dy = iy - iy_r;
        
        std::vector<float> coffes_y = getCubicCoeffs(dy, m_attrs.cubeCoeff);
        for (int i = 0; i < CUBIC_GRID_LEN; i++) {
            yFactor[oy * CUBIC_GRID_LEN + i] = coffes_y[i];
        }
    }
}

void RefInterpolateExecutor::cubicRef(const uint8_t* in_ptr_, uint8_t* out_ptr_,
                                    int B, int C, int IH, int IW, int OH, int OW) {
    const int idxNum = 1;
    auto* xOrigin = static_cast<int*>(auxTable.data());
    auto* xFactor = reinterpret_cast<float*>(&auxTable[OW]);
    auto* yOrigin = static_cast<int*>(&auxTable[(CUBIC_GRID_LEN + idxNum) * OW]);
    auto* yFactor = reinterpret_cast<float*>(&auxTable[(CUBIC_GRID_LEN + idxNum) * OW + OH]);

    const auto* in_ptr_f32 = reinterpret_cast<const float*>(in_ptr_);
    auto* out_ptr_f32 = reinterpret_cast<float*>(out_ptr_);

    parallel_for4d(B, C, OH, OW, [&](size_t n, size_t c, size_t oy, size_t ox) {
        const float* in_ptr_nc = in_ptr_f32 + (IW * IH * C * n + IW * IH * c);
        float* out_ptr_nc = out_ptr_f32 + (OW * OH * C * n + OW * OH * c);

        int iy = yOrigin[oy];
        int ix = xOrigin[ox];

        float retY = 0.F;
        for (int y = iy - 1, i = 0; y <= iy + 2; y++, i++) {
            int yInRange = std::max(0, std::min(y, IH - 1));
            const float* in_ptr_nch = in_ptr_nc + IW * yInRange;
            float retX = 0.F;
            for (int x = ix - 1, j = 0; x <= ix + 2; x++, j++) {
                int xInRange = std::max(0, std::min(x, IW - 1));
                retX += xFactor[ox * CUBIC_GRID_LEN + j] * in_ptr_nch[xInRange];
            }
            retY += yFactor[oy * CUBIC_GRID_LEN + i] * retX;
        }
        out_ptr_nc[oy * OW + ox] = retY;
    });
}

std::vector<float> RefInterpolateExecutor::getCubicCoeffs(float mantissa, float a) {
    float m = std::fabs(mantissa);
    std::vector<float> coeffs(4, 0.F);
    coeffs[0] = a * (m - 1.0) * (m - 1.0) * m;
    coeffs[1] = ((a + 2.0) * m - (a + 3.0)) * m * m + 1.0;
    coeffs[2] = (((-a - 2.0) * m + (2.0 * a + 3.0)) * m - a) * m;
    coeffs[3] = -a * m * m * (m - 1.0);
    return coeffs;
}

float RefInterpolateExecutor::getPillowBilinearCoeffs(float m) {
    if (m < 0.0f) {
        m = -m;
    }
    if (m < 1.0f) {
        return 1.0f - m;
    }
    return 0.0f;
}

float RefInterpolateExecutor::getPillowBicubicCoeffs(float m) {
    float a = -0.5f;
    if (m < 0.0f) {
        m = -m;
    }
    if (m < 1.0f) {
        return ((a + 2) * m - (a + 3)) * m * m + 1.0f;
    }
    if (m < 2.0f) {
        return ((a * m - 5 * a) * m + 8 * a) * m - 4 * a;
    }
    return 0.0f;
}

void RefInterpolateExecutor::buildTblPillow(const VectorDims& srcDimPad5d, const VectorDims& dstDim5d,
                                           const std::vector<float>& dataScales) {
    // Initialize thread number
    m_threads_num = parallel_get_max_threads();
    
    // For pillow, we need to get the correct scales based on which dimensions are being resized
    // dataScales contains scales for all dimensions (1.0 for unchanged dimensions)
    float fy = 1.0f;
    float fx = 1.0f;
    
    
    // NCHWAsNHWC means axes (1,2) correspond to logical H,W but physical C,H
    // So when NCHWAsNHWC=true:
    // - dataScales[1] is the scale for C dimension (which acts as logical H) 
    // - dataScales[2] is the scale for H dimension (which acts as logical W)
    if (m_attrs.NCHWAsNHWC && dataRank == 4) {
        // For NCHWAsNHWC with 4D tensor:
        // Physical layout is NCHW but logical layout is NHWC
        // When axes=(1,2), we scale logical H,W which are physical C,H
        // Looking at debug: dataScales: 1 1 2 4, but we expect [1,2,4,1] for [N,C,H,W]
        // This suggests dataScales maps differently - maybe spatial only?
        // Let's try using the actual dimension ratios for now
        float scaleC = static_cast<float>(dstDim5d[1]) / static_cast<float>(srcDimPad5d[1]); // C dimension scale
        float scaleH = static_cast<float>(dstDim5d[3]) / static_cast<float>(srcDimPad5d[3]); // H dimension scale
        fx = scaleH; // X pass scales logical W (physical H)
        fy = scaleC; // Y pass scales logical H (physical C)
        // std::cout << "NCHWAsNHWC: scaleC=" << scaleC << " scaleH=" << scaleH << " fx=" << fx << " fy=" << fy << std::endl;
    } else {
        // Normal case: Find scales for spatial dimensions (H and W)
        if (dataRank >= 3) {
            fy = dataScales[dataRank - 2]; // Height scale
        }
        if (dataRank >= 2) {
            fx = dataScales[dataRank - 1]; // Width scale
        }
    }
    
    int IH, IW, OH, OW;
    if (m_attrs.NCHWAsNHWC && dataRank == 4) {
        // For NCHWAsNHWC: physical (N,C,H,W) -> logical (N,H,W,C)
        IH = srcDimPad5d[1]; // physical C -> logical H
        IW = srcDimPad5d[3]; // physical H -> logical W
        OH = dstDim5d[1];    // physical C -> logical H
        OW = dstDim5d[3];    // physical H -> logical W
    } else {
        IH = srcDimPad5d[3];
        IW = srcDimPad5d[4];
        OH = dstDim5d[3];
        OW = dstDim5d[4];
    }
    
    // Debug output
    // std::cout << "buildTblPillow: IH=" << IH << " IW=" << IW << " OH=" << OH << " OW=" << OW 
    //           << " fy=" << fy << " fx=" << fx << " dataRank=" << dataRank 
    //           << " mode=" << (m_attrs.mode == InterpolateMode::bilinear_pillow ? "bilinear" : "bicubic")
    //           << " coordTransMode=" << static_cast<int>(m_attrs.coordTransMode) 
    //           << " NCHWAsNHWC=" << m_attrs.NCHWAsNHWC << std::endl;
    
    // Print all dataScales
    // std::cout << "dataScales: ";
    // for (auto s : dataScales) {
    //     std::cout << s << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "srcDimPad5d: ";
    // for (auto d : srcDimPad5d) {
    //     std::cout << d << " ";
    // }
    // std::cout << "\ndstDim5d: ";
    // for (auto d : dstDim5d) {
    //     std::cout << d << " ";
    // }
    // std::cout << std::endl;

    struct filterArgs {
        float (*weightGen)(float m);
        float ScaleClipReciprocal;
        float filterRadius;
        int filterLen;
    };

    // pillowScale: e.g. 2.0 means down sample 2 times
    auto generateArgs = [&](float pillowScale) -> filterArgs {
        float scaleClip = pillowScale < 1.0f ? 1.0f : pillowScale;
        float filterRadius = (m_attrs.mode == InterpolateMode::bilinear_pillow) 
            ? PILLOW_BILINEAR_WINDOW_SCALE * scaleClip
            : PILLOW_BICUBIC_WINDOW_SCALE * scaleClip;
        int filterLen = static_cast<int>(std::ceil(filterRadius * 2)) + 1;
        
        filterArgs args{
            (m_attrs.mode == InterpolateMode::bilinear_pillow)
                ? RefInterpolateExecutor::getPillowBilinearCoeffs
                : RefInterpolateExecutor::getPillowBicubicCoeffs,
            1.0f / scaleClip,
            filterRadius,
            filterLen
        };
        return args;
    };

    filterArgs filterArgsX = generateArgs(1.0f / fx);
    filterArgs filterArgsY = generateArgs(1.0f / fy);

    // index with Run Length Coding(start+len for each ow/oh)
    size_t weightLen = filterArgsX.filterLen * OW + filterArgsY.filterLen * OH;
    size_t boundLen = 2 * OW + 2 * OH;
    auxTable.resize(2 + weightLen + boundLen);
    size_t offset = 0;
    auxTable[offset] = filterArgsX.filterLen;
    auxTable[offset + 1] = filterArgsY.filterLen;
    offset += 2;
    auto* weightX = reinterpret_cast<float*>(&auxTable[offset]);
    offset += filterArgsX.filterLen * OW;
    auto* weightY = reinterpret_cast<float*>(&auxTable[offset]);
    offset += filterArgsY.filterLen * OH;
    auto* indexX = static_cast<int*>(&auxTable[offset]);
    offset += 2 * OW;
    auto* indexY = static_cast<int*>(&auxTable[offset]);

    auto generateTbl = [&](int inLen, int outLen, float fScale, filterArgs args, float* weightTbl, int* idxTbl) {
        int min = 0;
        int max = 0;
        for (int ox = 0; ox < outLen; ox++) {
            // For pillow interpolation, use direct coordinate calculation to match reference
            // This matches the pillow library's calculation: center = (xx + 0.5) * scale
            float ixCenter = (ox + 0.5f) * (1.0f / fScale);
            min = static_cast<int>(ixCenter - args.filterRadius + 0.5f);
            if (min < 0) {
                min = 0;
            }
            max = static_cast<int>(ixCenter + args.filterRadius + 0.5f);
            if (max > inLen) {
                max = inLen;
            }
            // use [min, max) range of input to get output
            // below let max become len
            max -= min;
            idxTbl[2 * ox] = min;
            idxTbl[2 * ox + 1] = max;

            size_t offset = ox * args.filterLen;
            float weightSum = 0;
            int ix = 0;
            for (ix = 0; ix < max; ix++) {
                // use distance to center as a parameter to compute weight
                float w = args.weightGen((ix + min - ixCenter + 0.5f) * args.ScaleClipReciprocal);
                weightTbl[offset + ix] = w;
                weightSum += w;
            }
            if (weightSum != 0) {
                for (ix = 0; ix < max; ix++) {
                    weightTbl[offset + ix] /= weightSum;
                }
            }

            // filterlen is maximum possible len, set others to 0 for possible uniform process(vector)
            for (; ix < args.filterLen; ix++) {
                weightTbl[offset + ix] = 0.f;
            }
        }
    };

    // fx and fy are already in the format dstDim/srcDim
    generateTbl(IW, OW, fx, filterArgsX, weightX, indexX);
    generateTbl(IH, OH, fy, filterArgsY, weightY, indexY);
    
    // Debug output - check indices
    // std::cout << "indexX: ";
    // for (int i = 0; i < std::min(10, 2 * OW); i++) {
    //     std::cout << indexX[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "indexY: ";
    // for (int i = 0; i < std::min(10, 2 * OH); i++) {
    //     std::cout << indexY[i] << " ";
    // }
    // std::cout << std::endl;
    
    // Create working buffer if needed
    bool xPass = IW != OW;
    bool yPass = IH != OH;
    if (xPass && yPass) {
        if (m_attrs.NCHWAsNHWC && dataRank == 4) {
            // For NCHWAsNHWC mode with physical NCHW layout:
            // We're scaling C and H dimensions
            // X pass: scales H dimension from IH to OW (logical W = physical H)
            // Intermediate buffer shape: (N, C, OW, W) where OW is the scaled H dimension
            size_t bufferSize = srcDimPad5d[0] * srcDimPad5d[1] * OW * srcDimPad5d[4] * srcDataSize;
            pillow_working_buf.resize(bufferSize);
        } else {
            // For normal mode, buffer needs to hold B * C * IH * OW elements
            pillow_working_buf.resize(srcDimPad5d[0] * srcDimPad5d[1] * IH * OW * srcDataSize);
        }
    }
}

void RefInterpolateExecutor::pillowRef(const uint8_t* in_ptr_, uint8_t* out_ptr_,
                                      int B, int C, int IH, int IW, int OH, int OW) {
    size_t offset = 0;
    int filterLenX = auxTable[offset];
    int filterLenY = auxTable[offset + 1];
    offset += 2;
    auto* weightX = reinterpret_cast<float*>(&auxTable[offset]);
    offset += filterLenX * OW;
    auto* weightY = reinterpret_cast<float*>(&auxTable[offset]);
    offset += filterLenY * OH;
    auto* indexX = static_cast<int*>(&auxTable[offset]);
    offset += 2 * OW;
    auto* indexY = static_cast<int*>(&auxTable[offset]);

    // workBuffer needed when both pass is true
    bool xPass = IW != OW;
    bool yPass = IH != OH;

    auto bc_loop = [&](size_t b, size_t c) {
        const uint8_t* in_ptr_nc = in_ptr_ + (IW * IH * C * b + IW * IH * c) * srcDataSize;
        uint8_t* out_ptr_nc = out_ptr_ + (OW * OH * C * b + OW * OH * c) * srcDataSize;
        uint8_t* xpass_out_ptr_nc = nullptr;
        const uint8_t* ypass_in_ptr_nc = nullptr;
        
        if (xPass && yPass) {
            // Always use simple indexing for working buffer
            xpass_out_ptr_nc = &pillow_working_buf[(OW * IH * C * b + OW * IH * c) * srcDataSize];
            ypass_in_ptr_nc = &pillow_working_buf[(OW * IH * C * b + OW * IH * c) * srcDataSize];
        } else if (xPass && !yPass) {
            xpass_out_ptr_nc = out_ptr_nc;
        } else if (!xPass && yPass) {
            ypass_in_ptr_nc = in_ptr_nc;
        } else if (!xPass && !yPass) {
            cpu_memcpy(out_ptr_nc, in_ptr_nc, OH * OW * srcDataSize);
            return;
        }
        
        float result = 0;
        int f = 0;
        int filterS = 0;
        int filterL = 0;
        float* weight = nullptr;
        
        if (xPass) {
            for (size_t ih = 0; ih < static_cast<size_t>(IH); ih++) {
                for (size_t ow = 0; ow < static_cast<size_t>(OW); ow++) {
                    filterS = indexX[ow * 2];
                    filterL = indexX[ow * 2 + 1];
                    weight = (&weightX[ow * filterLenX]);
                    result = 0.f;
                    
                    if (m_attrs.inPrc == ov::element::f32) {
                        auto* in_f32 = reinterpret_cast<const float*>(in_ptr_nc);
                        for (f = 0; f < filterL; f++) {
                            int idx = f + filterS;
                            if (idx >= 0 && idx < IW) {
                                float pixel = in_f32[ih * IW + idx];
                                result += pixel * weight[f];
                            }
                        }
                        reinterpret_cast<float*>(xpass_out_ptr_nc)[ih * OW + ow] = result;
                    } else {
                        // Handle other data types
                        for (f = 0; f < filterL; f++) {
                            int idx = f + filterS;
                            if (idx >= 0 && idx < IW) {
                                size_t offset = (ih * IW + idx) * srcDataSize;
                                float pixel = 0;
                                // Simple conversion for now - can be expanded for other types
                                if (srcDataSize == 4) {
                                    pixel = *reinterpret_cast<const float*>(&in_ptr_nc[offset]);
                                } else if (srcDataSize == 1) {
                                    pixel = static_cast<float>(in_ptr_nc[offset]);
                                }
                                result += pixel * weight[f];
                            }
                        }
                        
                        if (!isFloatCompatible(m_attrs.outPrc)) {
                            result = std::round(result);
                        }
                        
                        size_t out_idx = (ih * OW + ow) * srcDataSize;
                        if (srcDataSize == 4) {
                            *reinterpret_cast<float*>(&xpass_out_ptr_nc[out_idx]) = result;
                        } else if (srcDataSize == 1) {
                            xpass_out_ptr_nc[out_idx] = static_cast<uint8_t>(std::max(0.f, std::min(255.f, result)));
                        }
                    }
                }
            }
        }
        
        if (yPass) {
            for (size_t oh = 0; oh < static_cast<size_t>(OH); oh++) {
                filterS = indexY[oh * 2];
                filterL = indexY[oh * 2 + 1];
                weight = (&weightY[oh * filterLenY]);
                
                for (size_t ow = 0; ow < static_cast<size_t>(OW); ow++) {
                    result = 0.f;
                    
                    if (m_attrs.inPrc == ov::element::f32) {
                        auto* in_f32 = reinterpret_cast<const float*>(ypass_in_ptr_nc);
                        for (f = 0; f < filterL; f++) {
                            int idx = f + filterS;
                            if (idx >= 0 && idx < IH) {
                                float pixel = in_f32[idx * OW + ow];
                                result += pixel * weight[f];
                            }
                        }
                        reinterpret_cast<float*>(out_ptr_nc)[oh * OW + ow] = result;
                    } else {
                        for (f = 0; f < filterL; f++) {
                            int idx = f + filterS;
                            if (idx >= 0 && idx < IH) {
                                size_t offset = (idx * OW + ow) * srcDataSize;
                                float pixel = 0;
                                if (srcDataSize == 4) {
                                    pixel = *reinterpret_cast<const float*>(&ypass_in_ptr_nc[offset]);
                                } else if (srcDataSize == 1) {
                                    pixel = static_cast<float>(ypass_in_ptr_nc[offset]);
                                }
                                result += pixel * weight[f];
                            }
                        }
                        
                        if (!isFloatCompatible(m_attrs.outPrc)) {
                            result = std::round(result);
                        }
                        
                        size_t out_idx = (oh * OW + ow) * srcDataSize;
                        if (srcDataSize == 4) {
                            *reinterpret_cast<float*>(&out_ptr_nc[out_idx]) = result;
                        } else if (srcDataSize == 1) {
                            out_ptr_nc[out_idx] = static_cast<uint8_t>(std::max(0.f, std::min(255.f, result)));
                        }
                    }
                }
            }
        }
    };

    // Simple sequential execution for debugging
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            bc_loop(b, c);
        }
    }
}

void RefInterpolateExecutor::pillowRefNCHWAsNHWC(const uint8_t* in_ptr_, uint8_t* out_ptr_,
                                                int B, int logicalC, int logicalIH, int logicalIW, 
                                                int logicalOH, int logicalOW) {
    // For NCHWAsNHWC mode: we're treating physical NCHW as logical NHWC
    // Physical layout: (B, C, H, W) = (B, logicalIH, logicalIW, logicalC)
    // We're scaling logical H,W which are physical C,H dimensions
    // X pass scales logical W (physical H): logicalIW -> logicalOW
    // Y pass scales logical H (physical C): logicalIH -> logicalOH
    // std::cout << "pillowRefNCHWAsNHWC: B=" << B << " logicalC=" << logicalC 
    //           << " logicalIH=" << logicalIH << " logicalIW=" << logicalIW 
    //           << " logicalOH=" << logicalOH << " logicalOW=" << logicalOW << std::endl;
    
    size_t offset = 0;
    int filterLenX = auxTable[offset];
    int filterLenY = auxTable[offset + 1];
    offset += 2;
    
    auto* weightX = reinterpret_cast<float*>(&auxTable[offset]);
    offset += filterLenX * logicalOW;
    auto* weightY = reinterpret_cast<float*>(&auxTable[offset]);
    offset += filterLenY * logicalOH;
    auto* indexX = static_cast<int*>(&auxTable[offset]);
    offset += 2 * logicalOW;
    auto* indexY = static_cast<int*>(&auxTable[offset]);

    bool xPass = logicalIW != logicalOW;  // Logical W dimension (physical H)
    bool yPass = logicalIH != logicalOH;  // Logical H dimension (physical C)

    auto b_loop = [&](size_t b) {
        // For physical NCHW layout:
        // Input: (B, logicalIH, logicalIW, logicalC) = (B, C_phys, H_phys, W_phys)
        // Output: (B, logicalOH, logicalOW, logicalC) = (B, C_out, H_out, W_phys)
        const uint8_t* in_ptr_b = in_ptr_ + b * logicalIH * logicalIW * logicalC * srcDataSize;
        uint8_t* out_ptr_b = out_ptr_ + b * logicalOH * logicalOW * logicalC * srcDataSize;
        

        uint8_t* xpass_out_ptr_b = nullptr;
        const uint8_t* ypass_in_ptr_b = nullptr;

        if (xPass && yPass) {
            // After X pass (scaling physical H dimension), intermediate shape is (B, C, H_out, W)
            // Which in logical terms is (B, logicalIH, logicalOW, logicalC)
            size_t buffer_size = static_cast<size_t>(logicalIH) * logicalOW * logicalC;
            
            xpass_out_ptr_b = &pillow_working_buf[b * buffer_size * srcDataSize];
            ypass_in_ptr_b = &pillow_working_buf[b * buffer_size * srcDataSize];
        } else if (xPass && !yPass) {
            xpass_out_ptr_b = out_ptr_b;
        } else if (!xPass && yPass) {
            ypass_in_ptr_b = in_ptr_b;
        } else if (!xPass && !yPass) {
            cpu_memcpy(out_ptr_b, in_ptr_b, logicalOH * logicalOW * logicalC * srcDataSize);
            return;
        }

        float result = 0;
        int filterS = 0;
        int filterL = 0;
        float* weight = nullptr;

        // X pass: scale logical W (physical H) dimension
        if (xPass) {
            if (m_attrs.inPrc == ov::element::f32) {
                auto* in_f32 = reinterpret_cast<const float*>(in_ptr_b);
                auto* out_f32 = reinterpret_cast<float*>(xpass_out_ptr_b);
                
                // For each element in the output
                for (int c = 0; c < logicalIH; c++) {      // physical C dimension (not scaled in X pass)
                    for (int h = 0; h < logicalOW; h++) {  // physical H dimension (scaled in X pass)
                        filterS = indexX[h * 2];
                        filterL = indexX[h * 2 + 1];
                        weight = &weightX[h * filterLenX];
                        
                        for (int w = 0; w < logicalC; w++) {  // physical W dimension (not scaled)
                            result = 0.0f;
                            
                            // Apply filter along H dimension
                            for (int f = 0; f < filterL; f++) {
                                int src_h = filterS + f;
                                if (src_h >= 0 && src_h < logicalIW) {
                                    // Physical memory layout: NCHW
                                    size_t src_idx = c * logicalIW * logicalC + src_h * logicalC + w;
                                    result += in_f32[src_idx] * weight[f];
                                }
                            }
                            
                            // Write to output
                            size_t dst_idx = c * logicalOW * logicalC + h * logicalC + w;
                            out_f32[dst_idx] = result;
                        }
                    }
                }
            } else {
                OPENVINO_THROW("NCHWAsNHWC pillow interpolation only supports f32 precision");
            }
        }

        // Y pass: scale logical H (physical C) dimension
        if (yPass) {
            if (m_attrs.inPrc == ov::element::f32) {
                auto* in_f32 = reinterpret_cast<const float*>(ypass_in_ptr_b);
                auto* out_f32 = reinterpret_cast<float*>(out_ptr_b);
                
                // For each element in the output
                for (int c = 0; c < logicalOH; c++) {      // physical C dimension (scaled in Y pass)
                    filterS = indexY[c * 2];
                    filterL = indexY[c * 2 + 1];
                    weight = &weightY[c * filterLenY];
                    
                    for (int h = 0; h < logicalOW; h++) {  // physical H dimension (already scaled in X pass)
                        for (int w = 0; w < logicalC; w++) {  // physical W dimension (not scaled)
                            result = 0.0f;
                            
                            // Apply filter along C dimension
                            for (int f = 0; f < filterL; f++) {
                                int src_c = filterS + f;
                                if (src_c >= 0 && src_c < logicalIH) {
                                    // After X pass, intermediate buffer has shape: (C, H_out, W)
                                    size_t src_idx = src_c * logicalOW * logicalC + h * logicalC + w;
                                    result += in_f32[src_idx] * weight[f];
                                }
                            }
                            
                            // Write to output
                            size_t dst_idx = c * logicalOW * logicalC + h * logicalC + w;
                            out_f32[dst_idx] = result;
                        }
                    }
                }
            } else {
                OPENVINO_THROW("NCHWAsNHWC pillow interpolation only supports f32 precision");
            }
        }
    };

    // Process all batches
    for (int b = 0; b < B; b++) {
        b_loop(b);
    }
}


}  // namespace intel_cpu
}  // namespace ov