// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interpolate.hpp"

#include <algorithm>
#include <cmath>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/common/cpu_memcpy.h"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "utils/general_utils.h"

using namespace ov::intel_cpu;

bool ov::intel_cpu::InterpolateExecutor::init(const InterpolateAttrs& interpolateAttrs,
                                              const std::vector<MemoryDescPtr>& srcDescs,
                                              const std::vector<MemoryDescPtr>& dstDescs,
                                              [[maybe_unused]] const dnnl::primitive_attr& attr) {
    const auto& srcDims = srcDescs[0]->getShape().getStaticDims();
    const auto& dstDims = dstDescs[0]->getShape().getStaticDims();
    interpAttrs = interpolateAttrs;
    srcDim5d = to5Dim(srcDims);
    const auto toDimVec = [](const std::vector<int>& pads) {
        VectorDims dims(pads.size(), 0);
        for (size_t i = 0; i < pads.size(); ++i) {
            dims[i] = static_cast<Dim>(pads[i]);
        }
        return dims;
    };
    padBegin5d = to5Dim(toDimVec(interpolateAttrs.padBegin));
    padEnd5d = to5Dim(toDimVec(interpolateAttrs.padEnd));
    srcDimPad5d = to5Dim(getPaddedInputShape(srcDims, interpolateAttrs.padBegin, interpolateAttrs.padEnd));
    dstDim5d = to5Dim(dstDims);
    srcDataSize = interpolateAttrs.inPrc.size();
    dstDataSize = interpolateAttrs.outPrc.size();
    dataRank = srcDims.size();
    spatialDimSize = getSpatialDimsNum(dataRank);

    switch (interpAttrs.mode) {
    case InterpolateMode::nearest: {
        buildTblNN(srcDimPad5d,
                   dstDim5d,
                   interpAttrs.dataScales,
                   interpolateAttrs.layout,
                   interpolateAttrs.nearestMode);
        break;
    }
    case InterpolateMode::linear_onnx: {
        buildTblLinearOnnx(srcDimPad5d, dstDim5d, interpAttrs.dataScales, interpolateAttrs.layout);
        break;
    }
    case InterpolateMode::linear: {
        static constexpr int LINEAR_KERNEL = 2;
        buildTblLinear(srcDimPad5d, dstDim5d, interpAttrs.dataScales, LINEAR_KERNEL, interpolateAttrs.antialias);
        break;
    }
    case InterpolateMode::cubic: {
        buildTblCubic(srcDimPad5d,
                      dstDim5d,
                      interpAttrs.dataScales,
                      interpolateAttrs.cubeCoeff,
                      interpolateAttrs.layout);
        break;
    }
    default: {
        OPENVINO_THROW("Interpolate executor does not support interpolate mode: ", interpAttrs.mode);
        break;
    }
    }
    return true;
}
// =====================================================================================================================
// index layout:
// d_0............d_OD-1, h_0..............h_OH-1, w_0................w_OW-1
void ov::intel_cpu::InterpolateExecutor::buildTblNN(const VectorDims& srcDimPad5d,
                                                    const VectorDims& dstDim5d,
                                                    const std::vector<float>& dataScales,
                                                    [[maybe_unused]] InterpolateLayoutType layout,
                                                    InterpolateNearestMode nearestMode) {
    const int dimSize = dataRank;
    float fz = (dimSize == 5) ? dataScales[dimSize - 3] : 1.F;
    float fy = dataScales[dimSize - 2];
    float fx = dataScales[dimSize - 1];
    const bool usePaddedCoords = interpAttrs.hasPad;
    const size_t coordID = usePaddedCoords ? srcDimPad5d[2] : srcDim5d[2];
    const size_t coordIH = usePaddedCoords ? srcDimPad5d[3] : srcDim5d[3];
    const size_t coordIW = usePaddedCoords ? srcDimPad5d[4] : srcDim5d[4];
    size_t OD = dstDim5d[2];
    size_t OH = dstDim5d[3];
    size_t OW = dstDim5d[4];

    indexTable.resize(OD + OH + OW);
    bool isDDownsample = fz < 1;
    bool isHDownsample = fy < 1;
    bool isWDownsample = fx < 1;
    const bool hasPad = interpAttrs.hasPad;
    const auto finalizeIndex = [&](int idx, size_t dimIdx) -> int {
        if (!hasPad) {
            return clipCoord(idx, static_cast<int>(srcDim5d[dimIdx]));
        }
        return clipCoord(idx, static_cast<int>(srcDimPad5d[dimIdx]));
    };
    for (int oz = 0; oz < static_cast<int>(OD); oz++) {
        float iz = coordTransToInput(oz, fz, static_cast<int>(coordID), OD);
        indexTable[oz] = nearestRound(iz, isDDownsample, nearestMode);
        indexTable[oz] = finalizeIndex(indexTable[oz], 2);
    }
    for (int oy = 0; oy < static_cast<int>(OH); oy++) {
        float iy = coordTransToInput(oy, fy, static_cast<int>(coordIH), OH);
        indexTable[OD + oy] = nearestRound(iy, isHDownsample, nearestMode);
        indexTable[OD + oy] = finalizeIndex(indexTable[OD + oy], 3);
    }
    for (int ox = 0; ox < static_cast<int>(OW); ox++) {
        float ix = coordTransToInput(ox, fx, static_cast<int>(coordIW), OW);
        indexTable[OD + OH + ox] = nearestRound(ix, isWDownsample, nearestMode);
        indexTable[OD + OH + ox] = finalizeIndex(indexTable[OD + OH + ox], 4);
    }
}

// scale is float(outShape) / float(inShape)
// strictly consistent with onnx calc manner(div scale, not multiply inverse), given this is done offline
// the slight precison diff can produce obvious wrong value due to "nearest round" behavior for NN mode
float ov::intel_cpu::InterpolateExecutor::coordTransToInput(int outCoord,
                                                            float scale,
                                                            int inShape,
                                                            int outShape) const {
    if (scale == 1.0F || (inShape == outShape)) {
        return static_cast<float>(outCoord);
    }
    switch (interpAttrs.coordTransMode) {
    case InterpolateCoordTransMode::half_pixel: {
        return (static_cast<float>(outCoord) + 0.5F) / scale - 0.5F;
    }
    case InterpolateCoordTransMode::pytorch_half_pixel: {
        if (outShape > 1) {
            return (static_cast<float>(outCoord) + 0.5F) / scale - 0.5F;
        }
        return 0;
    }
    case InterpolateCoordTransMode::asymmetric: {
        return static_cast<float>(outCoord) / scale;
    }
    case InterpolateCoordTransMode::tf_half_pixel_for_nn: {
        return (static_cast<float>(outCoord) + 0.5F) / scale;
    }
    case InterpolateCoordTransMode::align_corners: {
        if (outShape > 1) {
            return static_cast<float>(outCoord) * (static_cast<float>(inShape - 1) / static_cast<float>(outShape - 1));
        }
        return 0;
    }
    default: {
        OPENVINO_THROW("Interpolate executor does not support specified coordinate transformation mode");
        break;
    }
    }
}

int ov::intel_cpu::InterpolateExecutor::nearestRound(float originCoord,
                                                     bool isDownsample,
                                                     InterpolateNearestMode nearestMode) {
    switch (nearestMode) {
    case InterpolateNearestMode::round_prefer_floor: {
        if (originCoord == (static_cast<float>(static_cast<int>(originCoord)) + 0.5F)) {
            return static_cast<int>(std::floor(originCoord));
        }
        return static_cast<int>(std::round(originCoord));
    }
    case InterpolateNearestMode::round_prefer_ceil: {
        return static_cast<int>(std::round(originCoord));
    }
    case InterpolateNearestMode::floor: {
        return static_cast<int>(std::floor(originCoord));
    }
    case InterpolateNearestMode::ceil: {
        return static_cast<int>(std::ceil(originCoord));
    }
    case InterpolateNearestMode::simple: {
        if (isDownsample) {
            return static_cast<int>(std::ceil(originCoord));
        }
        return static_cast<int>(originCoord);
    }
    default: {
        OPENVINO_THROW("Interpolate executor does not support specified nearest round mode");
        break;
    }
    }
}

void ov::intel_cpu::InterpolateExecutor::linearOnnxCF(int outCoord,
                                                      float scale,
                                                      int inShape,
                                                      int outShape,
                                                      int& index0,
                                                      int& index1,
                                                      float& weight0,
                                                      float& weight1,
                                                      bool allowHalo) {
    float inCoord = coordTransToInput(outCoord, scale, inShape, outShape);
    if (!allowHalo) {
        inCoord = std::max(0.0F, std::min(inCoord, static_cast<float>(inShape - 1)));
    }
    float floorCoord = std::floor(inCoord);
    index0 = static_cast<int>(floorCoord);
    index1 = index0 + 1;
    if (!allowHalo) {
        index0 = std::max(0, std::min(index0, inShape - 1));
        index1 = std::max(0, std::min(index1, inShape - 1));
    }

    float t = inCoord - floorCoord;
    weight1 = t;
    weight0 = 1.0F - t;
    if (index0 == index1) {
        weight0 = 0.5F;
        weight1 = 0.5F;
    }
}

void ov::intel_cpu::InterpolateExecutor::buildTblLinearOnnx(const VectorDims& srcDimPad5d,
                                                            const VectorDims& dstDim5d,
                                                            const std::vector<float>& dataScales,
                                                            InterpolateLayoutType layout) {
    int dimSize = dataRank;
    float fz = (spatialDimSize > 2) ? dataScales[dimSize - 3] : 1.F;
    float fy = (spatialDimSize > 1) ? dataScales[dimSize - 2] : 1.F;
    float fx = dataScales[dimSize - 1];
    const bool usePaddedCoords = interpAttrs.hasPad;
    int coordID = usePaddedCoords ? static_cast<int>(srcDimPad5d[2]) : static_cast<int>(srcDim5d[2]);
    int coordIH = usePaddedCoords ? static_cast<int>(srcDimPad5d[3]) : static_cast<int>(srcDim5d[3]);
    int coordIW = usePaddedCoords ? static_cast<int>(srcDimPad5d[4]) : static_cast<int>(srcDim5d[4]);
    int IH = srcDimPad5d[3];
    int IW = srcDimPad5d[4];
    int OD = dstDim5d[2];
    int OH = dstDim5d[3];
    int OW = dstDim5d[4];
    const bool allowHalo = interpAttrs.hasPad;
    const auto finalizeIndex = [&](int idx, size_t dimIdx) -> int {
        if (!interpAttrs.hasPad) {
            return clipCoord(idx, static_cast<int>(srcDim5d[dimIdx]));
        }
        return clipCoord(idx, static_cast<int>(srcDimPad5d[dimIdx]));
    };

    std::vector<int*> indexPtr(MAX_INPUT_INTERPOLATE, nullptr);
    std::vector<float*> weightPtr(MAX_INPUT_INTERPOLATE, nullptr);
    if (layout == InterpolateLayoutType::planar) {
        // FrontTopLeft:0, FrontTopRight:1, FrontBottomLeft:2, FrontBottomRight:3,
        // EndTopLeft:4,   EndTopRight:5,   EndBottomLeft:6,   EndBottomRight:7
        // weight: Left:0, ritht:1, top:2, bottom:3, front:4, end:5
        int eltInGrid = [&]() -> int {
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
        indexTable.resize(idxType * scratchLen);

        indexPtr[0] = static_cast<int*>(indexTable.data());
        indexPtr[1] = static_cast<int*>(&indexTable[OW * OH * OD]);
        weightPtr[0] = reinterpret_cast<float*>(&indexTable[scratchLen]);
        weightPtr[1] = reinterpret_cast<float*>(&indexTable[scratchLen + OW * OH * OD]);
        if (spatialDimSize > 1) {
            indexPtr[2] = static_cast<int*>(&indexTable[2 * OW * OH * OD]);
            indexPtr[3] = static_cast<int*>(&indexTable[3 * OW * OH * OD]);
            weightPtr[2] = reinterpret_cast<float*>(&indexTable[scratchLen + 2 * OW * OH * OD]);
            weightPtr[3] = reinterpret_cast<float*>(&indexTable[scratchLen + 3 * OW * OH * OD]);
        }
        if (spatialDimSize > 2) {
            indexPtr[4] = static_cast<int*>(&indexTable[4 * OW * OH * OD]);
            indexPtr[5] = static_cast<int*>(&indexTable[5 * OW * OH * OD]);
            indexPtr[6] = static_cast<int*>(&indexTable[6 * OW * OH * OD]);
            indexPtr[7] = static_cast<int*>(&indexTable[7 * OW * OH * OD]);
            weightPtr[4] = reinterpret_cast<float*>(&indexTable[scratchLen + 4 * OW * OH * OD]);
            weightPtr[5] = reinterpret_cast<float*>(&indexTable[scratchLen + 5 * OW * OH * OD]);
        }
        int scale = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41) ? srcDataSize : 1;

        for (int oz = 0; oz < OD; oz++) {
            int izF = 0;
            int izE = 0;
            float weightF = NAN;
            float weightE = NAN;
            linearOnnxCF(oz, fz, coordID, OD, izF, izE, weightF, weightE, allowHalo);
            const int izFPad = finalizeIndex(izF, 2);
            const int izEPad = finalizeIndex(izE, 2);
            int idxOz = oz * OH * OW;
            for (int oy = 0; oy < OH; oy++) {
                int iyT = 0;
                int iyB = 0;
                float weightT = NAN;
                float weightB = NAN;
                linearOnnxCF(oy, fy, coordIH, OH, iyT, iyB, weightT, weightB, allowHalo);
                const int iyTPad = finalizeIndex(iyT, 3);
                const int iyBPad = finalizeIndex(iyB, 3);
                int idxOzOy = idxOz + oy * OW;
                for (int ox = 0; ox < OW; ox++) {
                    int ixL = 0;
                    int ixR = 0;
                    float weightL = NAN;
                    float weightR = NAN;
                    linearOnnxCF(ox, fx, coordIW, OW, ixL, ixR, weightL, weightR, allowHalo);
                    const int ixLPad = finalizeIndex(ixL, 4);
                    const int ixRPad = finalizeIndex(ixR, 4);
                    int idxOzOyOx = idxOzOy + ox;
                    indexPtr[0][idxOzOyOx] = (izFPad * IH * IW + iyTPad * IW + ixLPad) * scale;
                    indexPtr[1][idxOzOyOx] = (izFPad * IH * IW + iyTPad * IW + ixRPad) * scale;
                    weightPtr[0][idxOzOyOx] = weightL;
                    weightPtr[1][idxOzOyOx] = weightR;
                    if (spatialDimSize > 1) {
                        indexPtr[2][idxOzOyOx] = (izFPad * IH * IW + iyBPad * IW + ixLPad) * scale;
                        indexPtr[3][idxOzOyOx] = (izFPad * IH * IW + iyBPad * IW + ixRPad) * scale;
                        weightPtr[2][idxOzOyOx] = weightT;
                        weightPtr[3][idxOzOyOx] = weightB;
                    }
                    if (spatialDimSize > 2) {
                        indexPtr[4][idxOzOyOx] = (izEPad * IH * IW + iyTPad * IW + ixLPad) * scale;
                        indexPtr[5][idxOzOyOx] = (izEPad * IH * IW + iyTPad * IW + ixRPad) * scale;
                        indexPtr[6][idxOzOyOx] = (izEPad * IH * IW + iyBPad * IW + ixLPad) * scale;
                        indexPtr[7][idxOzOyOx] = (izEPad * IH * IW + iyBPad * IW + ixRPad) * scale;
                        weightPtr[4][idxOzOyOx] = weightF;
                        weightPtr[5][idxOzOyOx] = weightE;
                    }
                }
            }
        }
    } else {
        // index: left:OW right:OW Top:OH Bottom:OH, Front:OD, End:OD
        // weight:same as index
        size_t scratchLen = rnd_up(OW + OW + OH + OH + OD + OD, 16);
        int idxType = 2;
        indexTable.resize(idxType * scratchLen);
        indexPtr[0] = static_cast<int*>(indexTable.data());
        indexPtr[1] = static_cast<int*>(&indexTable[OW]);
        indexPtr[2] = static_cast<int*>(&indexTable[2 * OW]);
        indexPtr[3] = static_cast<int*>(&indexTable[2 * OW + OH]);
        indexPtr[4] = static_cast<int*>(&indexTable[2 * OW + 2 * OH]);
        indexPtr[5] = static_cast<int*>(&indexTable[2 * OW + 2 * OH + OD]);

        weightPtr[0] = reinterpret_cast<float*>(&indexTable[scratchLen]);
        weightPtr[1] = reinterpret_cast<float*>(&indexTable[scratchLen + OW]);
        weightPtr[2] = reinterpret_cast<float*>(&indexTable[scratchLen + 2 * OW]);
        weightPtr[3] = reinterpret_cast<float*>(&indexTable[scratchLen + 2 * OW + OH]);
        weightPtr[4] = reinterpret_cast<float*>(&indexTable[scratchLen + 2 * OW + 2 * OH]);
        weightPtr[5] = reinterpret_cast<float*>(&indexTable[scratchLen + 2 * OW + 2 * OH + OD]);

        for (int ox = 0; ox < OW; ox++) {
            linearOnnxCF(ox, fx, coordIW, OW, indexPtr[0][ox], indexPtr[1][ox], weightPtr[0][ox], weightPtr[1][ox], allowHalo);
            indexPtr[0][ox] = finalizeIndex(indexPtr[0][ox], 4);
            indexPtr[1][ox] = finalizeIndex(indexPtr[1][ox], 4);
        }
        for (int oy = 0; oy < OH; oy++) {
            linearOnnxCF(oy, fy, coordIH, OH, indexPtr[2][oy], indexPtr[3][oy], weightPtr[2][oy], weightPtr[3][oy], allowHalo);
            indexPtr[2][oy] = finalizeIndex(indexPtr[2][oy], 3);
            indexPtr[3][oy] = finalizeIndex(indexPtr[3][oy], 3);
        }
        for (int oz = 0; oz < OD; oz++) {
            linearOnnxCF(oz, fz, coordID, OD, indexPtr[4][oz], indexPtr[5][oz], weightPtr[4][oz], weightPtr[5][oz], allowHalo);
            indexPtr[4][oz] = finalizeIndex(indexPtr[4][oz], 2);
            indexPtr[5][oz] = finalizeIndex(indexPtr[5][oz], 2);
        }
    }
}

// table layout:
// wd .........wd, wh............wh, ww.............ww, id...........id, ih............ih, iw..............iw
//                        |                                                      |
//                   wh0.....wh_diameter                                    ih0.....ih_diameter
void ov::intel_cpu::InterpolateExecutor::buildTblLinear(const VectorDims& srcDimPad5d,
                                                        const VectorDims& dstDim5d,
                                                        const std::vector<float>& dataScales,
                                                        int kernel_width,
                                                        bool antialias) {
    int dimSize = dataRank;
    float fz = (dimSize == 5) ? dataScales[dimSize - 3] : 1.F;
    float fy = dataScales[dimSize - 2];
    float fx = dataScales[dimSize - 1];
    const bool usePaddedCoords = interpAttrs.hasPad;
    size_t coordID = usePaddedCoords ? srcDimPad5d[2] : srcDim5d[2];
    size_t coordIH = usePaddedCoords ? srcDimPad5d[3] : srcDim5d[3];
    size_t coordIW = usePaddedCoords ? srcDimPad5d[4] : srcDim5d[4];
    size_t OD = dstDim5d[2];
    size_t OH = dstDim5d[3];
    size_t OW = dstDim5d[4];

    if (coordIW != OW || coordIH != OH || coordID != OD) {
        float ax = antialias ? fx : 1.0F;
        float ay = antialias ? fy : 1.0F;
        float az = antialias ? fz : 1.0F;

        int rx = (fx > 1.0F) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ax));
        int ry = (fy > 1.0F) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ay));
        int rz = (fz > 1.0F) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / az));

        int diaOD = 2 * rz + 1;
        int diaOH = 2 * ry + 1;
        int diaOW = 2 * rx + 1;
        int sizeOD = OD * diaOD;
        int sizeOH = OH * diaOH;
        int sizeOW = OW * diaOW;
        indexTable.resize((sizeOD + sizeOH + sizeOW) * 2);
        auto* weightTable = reinterpret_cast<float*>(indexTable.data());
        auto* weightOD = (&weightTable[0]);
        auto* weightOH = (&weightTable[sizeOD]);
        auto* weightOW = (&weightTable[sizeOD + sizeOH]);

        auto* idxTable = static_cast<int*>(&indexTable[sizeOD + sizeOH + sizeOW]);
        auto* idxOD = (&idxTable[0]);
        auto* idxOH = (&idxTable[sizeOD]);
        auto* idxOW = (&idxTable[sizeOD + sizeOH]);

        const auto finalizeIndex = [&](int idx, size_t dimIdx) -> int {
            if (!interpAttrs.hasPad) {
                return clipCoord(idx, static_cast<int>(srcDim5d[dimIdx]));
            }
            return clipCoord(idx, static_cast<int>(srcDimPad5d[dimIdx]));
        };

        for (int oz = 0; oz < static_cast<int>(OD); oz++) {
            float iz = coordTransToInput(oz, fz, static_cast<int>(coordID), OD);
            auto iz_r = static_cast<int>(std::round(iz));
            for (int r = iz_r - rz, i = 0; r <= iz_r + rz; r++, i++) {
                idxOD[oz * diaOD + i] = finalizeIndex(r, 2);
                if (!interpAttrs.hasPad && (r < 0 || r >= static_cast<int>(coordID))) {
                    weightOD[oz * diaOD + i] = 0.F;
                } else {
                    float dz = iz - static_cast<float>(r);
                    weightOD[oz * diaOD + i] = az * triangleCoeff(az * dz);
                }
            }
        }
        for (int oy = 0; oy < static_cast<int>(OH); oy++) {
            float iy = coordTransToInput(oy, fy, static_cast<int>(coordIH), OH);
            auto iy_r = static_cast<int>(std::round(iy));
            for (int r = iy_r - ry, i = 0; r <= iy_r + ry; r++, i++) {
                idxOH[oy * diaOH + i] = finalizeIndex(r, 3);
                if (!interpAttrs.hasPad && (r < 0 || r >= static_cast<int>(coordIH))) {
                    weightOH[oy * diaOH + i] = 0.F;
                } else {
                    float dy = iy - static_cast<float>(r);
                    weightOH[oy * diaOH + i] = ay * triangleCoeff(ay * dy);
                }
            }
        }
        for (int ox = 0; ox < static_cast<int>(OW); ox++) {
            float ix = coordTransToInput(ox, fx, static_cast<int>(coordIW), OW);
            auto ix_r = static_cast<int>(std::round(ix));
            for (int r = ix_r - rx, i = 0; r <= ix_r + rx; r++, i++) {
                idxOW[ox * diaOW + i] = finalizeIndex(r, 4);
                if (!interpAttrs.hasPad && (r < 0 || r >= static_cast<int>(coordIW))) {
                    weightOW[ox * diaOW + i] = 0.F;
                } else {
                    float dx = ix - static_cast<float>(r);
                    weightOW[ox * diaOW + i] = ax * triangleCoeff(ax * dx);
                }
            }
        }
    }
}

std::vector<float> ov::intel_cpu::InterpolateExecutor::getCubicCoeffs(float mantissa, float a) {
    float m = std::fabs(mantissa);
    std::vector<float> coeffs(4, 0.F);

    coeffs[0] = a * (m - 1.0F) * (m - 1.0F) * m;
    coeffs[1] = ((a + 2.0F) * m - (a + 3.0F)) * m * m + 1.0F;
    coeffs[2] = (((-a - 2.0F) * m + (2.0F * a + 3.0F)) * m - a) * m;
    coeffs[3] = -a * m * m * (m - 1.0F);
    return coeffs;
}

// table layout:
// OW      OW         OW         OW         OW          OH       OH           OH           OH           OH
// x_idx   x_weight0  x_weight1  x_weight2  x_weight3   y_idx    y_weight0    y_weight1    y_weight2    y_weight3
void ov::intel_cpu::InterpolateExecutor::buildTblCubic(const VectorDims& srcDimPad5d,
                                                       const VectorDims& dstDim5d,
                                                       const std::vector<float>& dataScales,
                                                       float cubicCoeff,
                                                       InterpolateLayoutType layout) {
    int dimSize = dataRank;
    float fy = dataScales[dimSize - 2];
    float fx = dataScales[dimSize - 1];
    int IH = srcDimPad5d[3];
    int IW = srcDimPad5d[4];
    int OH = dstDim5d[3];
    int OW = dstDim5d[4];

    // idxNum for index, CUBIC_GRID_LEN for weight
    const int idxNum = 1;
    size_t idxWeightSize = (CUBIC_GRID_LEN + idxNum) * OW + (CUBIC_GRID_LEN + idxNum) * OH;
    if (layout != InterpolateLayoutType::planar) {
        indexTable.resize(idxWeightSize);
    } else {
        size_t sequenceSize = 2 * OH * OW;
        indexTable.resize(idxWeightSize + sequenceSize);
    }

    int tblAdvance = 0;
    auto* xOrigin = static_cast<int*>(&indexTable[tblAdvance]);
    tblAdvance += OW;
    auto* xFactor = reinterpret_cast<float*>(&indexTable[tblAdvance]);
    for (int ox = 0; ox < OW; ox++) {
        float ix = coordTransToInput(ox, fx, IW, OW);
        auto ix_r = static_cast<int>(std::floor(ix));
        xOrigin[ox] = ix_r;
        float m = ix - static_cast<float>(ix_r);
        std::vector<float> coffes = getCubicCoeffs(m, cubicCoeff);
        xFactor[CUBIC_GRID_LEN * ox] = coffes[0];
        xFactor[CUBIC_GRID_LEN * ox + 1] = coffes[1];
        xFactor[CUBIC_GRID_LEN * ox + 2] = coffes[2];
        xFactor[CUBIC_GRID_LEN * ox + 3] = coffes[3];
    }

    tblAdvance += CUBIC_GRID_LEN * OW;
    auto* yOrigin = static_cast<int*>(&indexTable[tblAdvance]);
    tblAdvance += OH;
    auto* yFactor = reinterpret_cast<float*>(&indexTable[tblAdvance]);
    for (int oy = 0; oy < OH; oy++) {
        float iy = coordTransToInput(oy, fy, IH, OH);
        auto iy_r = static_cast<int>(std::floor(iy));
        yOrigin[oy] = iy_r;
        float m = iy - static_cast<float>(iy_r);
        std::vector<float> coffes = getCubicCoeffs(m, cubicCoeff);
        yFactor[CUBIC_GRID_LEN * oy] = coffes[0];
        yFactor[CUBIC_GRID_LEN * oy + 1] = coffes[1];
        yFactor[CUBIC_GRID_LEN * oy + 2] = coffes[2];
        yFactor[CUBIC_GRID_LEN * oy + 3] = coffes[3];
    }

    if (layout == InterpolateLayoutType::planar) {
        tblAdvance += CUBIC_GRID_LEN * OH;
        auto* sequenceOH = static_cast<int*>(&indexTable[tblAdvance]);
        tblAdvance += OH * OW;
        auto* sequenceOW = static_cast<int*>(&indexTable[tblAdvance]);
        for (int h = 0; h < OH; ++h) {
            int offset = h * OW;
            for (int w = 0; w < OW; ++w) {
                sequenceOH[offset + w] = h * sizeof(int);
                sequenceOW[offset + w] = w * sizeof(int);
            }
        }
    }
}

// shapeND: n     c     d     h    w
// blockND: ncdhw cdhw  dhw   hw   w    1
// index  : 0      1    2     3    4    5
inline VectorDims getBlockND(const VectorDims& shape) {
    int shapeRank = shape.size();
    VectorDims blockND(shapeRank + 1, 1);
    for (int i = shapeRank - 1; i >= 0; i--) {
        blockND[i] = shape[i] * blockND[i + 1];
    }
    return blockND;
}

const uint8_t* ov::intel_cpu::InterpolateExecutor::padPreprocess(const std::vector<MemoryCPtr>& src,
                                                                 const std::vector<MemoryPtr>& dst) {
    const uint8_t* src_data_origin = src[0]->getDataAs<uint8_t>();

    const auto& srcDim = src[0]->getStaticDims();
    const auto& dstDim = dst[0]->getStaticDims();
    size_t dimSize = srcDim.size();
    auto srcDimPad = getSrcDimPad5d();

    const auto srcDim5d = to5Dim(srcDim);
    const auto srcDimPad5d = to5Dim(srcDimPad);
    const auto dstDim5d = to5Dim(dstDim);
    const auto srcDataSize = src[0]->getDesc().getPrecision().size();

    const uint8_t* src_data = src_data_origin;
    if (interpAttrs.hasPad) {
        int padB0 = (dimSize > 2) ? interpAttrs.padBegin[0] : 0;
        int padB1 = (dimSize > 2) ? interpAttrs.padBegin[1] : 0;
        int padB2 = (dimSize == 5) ? interpAttrs.padBegin[dimSize - 3] : 0;
        int padB3 = interpAttrs.padBegin[dimSize - 2];
        int padB4 = interpAttrs.padBegin[dimSize - 1];

        VectorDims inShapeBlock = getBlockND(srcDim5d);
        VectorDims inShapePadBlock = getBlockND(srcDimPad5d);

        if (interpAttrs.layout == InterpolateLayoutType::planar) {
            m_srcPadded.assign(inShapePadBlock[0] * srcDataSize, 0);
            auto* src_data_pad = static_cast<uint8_t*>(m_srcPadded.data());
            parallel_for4d(srcDim5d[0], srcDim5d[1], srcDim5d[2], srcDim5d[3], [&](int n, int c, int d, int h) {
                const uint8_t* src = src_data_origin + (inShapeBlock[1] * n + inShapeBlock[2] * c +
                                                        inShapeBlock[3] * d + inShapeBlock[4] * h) *
                                                           srcDataSize;
                uint8_t* srcPad =
                    src_data_pad + (inShapePadBlock[1] * (n + padB0) + inShapePadBlock[2] * (c + padB1) +
                                    inShapePadBlock[3] * (d + padB2) + inShapePadBlock[4] * (h + padB3) + padB4) *
                                       srcDataSize;
                cpu_memcpy(srcPad, src, srcDim5d[4] * srcDataSize);
            });
            if (std::getenv("OV_DEBUG_INTERP_PAD")) {
                static bool logged_planar = false;
                if (!logged_planar) {
                    logged_planar = true;
                    const size_t preview = std::min<size_t>(4, srcDim5d[4]);
                    const size_t offsetBytes =
                        (inShapePadBlock[1] * padB0 + inShapePadBlock[2] * padB1 + inShapePadBlock[3] * padB2 +
                         inShapePadBlock[4] * padB3 + padB4) *
                        srcDataSize;
                    std::fprintf(stderr, "[InterpolateExecutor] planar pad preview:");
                    for (size_t i = 0; i < preview * srcDataSize; ++i) {
                        std::fprintf(stderr, " %02x", src_data_pad[offsetBytes + i]);
                    }
                    std::fprintf(stderr, " | src:");
                    for (size_t i = 0; i < preview * srcDataSize; ++i) {
                        std::fprintf(stderr, " %02x", src_data_origin[i]);
                    }
                    std::fprintf(stderr, "\n");
                }
            }
            src_data = src_data_pad;
        } else if (interpAttrs.layout == InterpolateLayoutType::by_channel) {
            m_srcPadded.assign(inShapePadBlock[0] * srcDataSize, 0);
            auto* src_data_pad = static_cast<uint8_t*>(m_srcPadded.data());
            parallel_for4d(srcDim5d[0], srcDim5d[2], srcDim5d[3], srcDim5d[4], [&](int n, int d, int h, int w) {
                const uint8_t* src = src_data_origin +
                                     (inShapeBlock[1] * n +
                                      (inShapeBlock[3] * d + inShapeBlock[4] * h + inShapeBlock[5] * w) * srcDim5d[1]) *
                                         srcDataSize;
                uint8_t* srcPad = src_data_pad + (inShapePadBlock[1] * (n + padB0) +
                                                  (inShapePadBlock[3] * (d + padB2) + inShapePadBlock[4] * (h + padB3) +
                                                   inShapePadBlock[5] * (w + padB4)) *
                                                      srcDimPad5d[1] +
                                                  padB1) *
                                                     srcDataSize;
                cpu_memcpy(srcPad, src, srcDim5d[1] * srcDataSize);
            });
            if (std::getenv("OV_DEBUG_INTERP_PAD")) {
                static bool logged_by_ch = false;
                if (!logged_by_ch) {
                    logged_by_ch = true;
                    const size_t preview = std::min<size_t>(4, srcDim5d[1]);
                    const size_t offsetBytes =
                        (inShapePadBlock[1] * padB0 +
                         (inShapePadBlock[3] * padB2 + inShapePadBlock[4] * padB3 + inShapePadBlock[5] * padB4) *
                             srcDimPad5d[1] +
                         padB1) *
                        srcDataSize;
                    std::fprintf(stderr, "[InterpolateExecutor] by_channel pad preview:");
                    for (size_t i = 0; i < preview * srcDataSize; ++i) {
                        std::fprintf(stderr, " %02x", src_data_pad[offsetBytes + i]);
                    }
                    std::fprintf(stderr, " | src:");
                    for (size_t i = 0; i < preview * srcDataSize; ++i) {
                        std::fprintf(stderr, " %02x", src_data_origin[i]);
                    }
                    std::fprintf(stderr, "\n");
                }
            }
            src_data = src_data_pad;
        } else if (interpAttrs.layout == InterpolateLayoutType::block) {
            size_t blkSize = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ? 16 : 8;
            size_t CB = div_up(srcDimPad5d[1], blkSize);
            size_t eltsTotal = srcDimPad5d[0] * CB * srcDimPad5d[2] * srcDimPad5d[3] * srcDimPad5d[4] * blkSize;
            m_srcPadded.assign(eltsTotal * srcDataSize, 0x0);
            auto* src_data_pad = static_cast<uint8_t*>(m_srcPadded.data());
            OPENVINO_ASSERT(srcDim5d[0] == srcDimPad5d[0] && srcDim5d[1] == srcDimPad5d[1],
                            "Interpolate executor does not support padding on batch and channel dimensions");
            parallel_for5d(
                srcDim5d[0],
                CB,
                srcDim5d[2],
                srcDim5d[3],
                srcDim5d[4],
                [&](int n, int cb, int d, int h, int w) {
                    const uint8_t* src = src_data_origin +
                                         (n * CB * srcDim5d[2] * srcDim5d[3] * srcDim5d[4] * blkSize) * srcDataSize +
                                         (cb * srcDim5d[2] * srcDim5d[3] * srcDim5d[4] * blkSize) * srcDataSize +
                                         (d * srcDim5d[3] * srcDim5d[4] * blkSize) * srcDataSize +
                                         (h * srcDim5d[4] * blkSize) * srcDataSize + (w * blkSize) * srcDataSize;
                    uint8_t* srcPad =
                        src_data_pad +
                        (n * CB * srcDimPad5d[2] * srcDimPad5d[3] * srcDimPad5d[4] * blkSize) * srcDataSize +
                        (cb * srcDimPad5d[2] * srcDimPad5d[3] * srcDimPad5d[4] * blkSize) * srcDataSize +
                        ((d + padB2) * srcDimPad5d[3] * srcDimPad5d[4] * blkSize) * srcDataSize +
                        ((h + padB3) * srcDimPad5d[4] * blkSize) * srcDataSize + ((w + padB4) * blkSize) * srcDataSize;
                    cpu_memcpy(srcPad, src, blkSize * srcDataSize);
                });
            if (std::getenv("OV_DEBUG_INTERP_PAD")) {
                static bool logged_block = false;
                if (!logged_block) {
                    logged_block = true;
                    const size_t previewBytes = std::min<size_t>(blkSize, 4UL) * srcDataSize;
                    const size_t offsetBytes = ((padB2 * srcDimPad5d[3] * srcDimPad5d[4] * blkSize) +
                                                (padB3 * srcDimPad5d[4] * blkSize) + (padB4 * blkSize)) *
                                               srcDataSize;
                    std::fprintf(stderr, "[InterpolateExecutor] block pad preview:");
                    for (size_t i = 0; i < previewBytes; ++i) {
                        std::fprintf(stderr, " %02x", src_data_pad[offsetBytes + i]);
                    }
                    std::fprintf(stderr, "\n");
                }
            }
            src_data = src_data_pad;
        }
    } else {
        m_srcPadded.clear();
        src_data = src_data_origin;
    }
    return src_data;
}
