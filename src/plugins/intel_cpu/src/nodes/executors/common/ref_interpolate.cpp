// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ref_interpolate.hpp"
#include "ie_parallel.hpp"
#include "nodes/common/cpu_memcpy.h"
#include "utils/bfloat16.hpp"

void ov::intel_cpu::InterpolateRefExecutor::exec(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_) {
    size_t N = srcDimPad5d[0], C = srcDimPad5d[1], ID = srcDimPad5d[2], IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    size_t OD = dstDim5d[2], OH = dstDim5d[3], OW = dstDim5d[4];

    switch (mode) {
        case InterpolateMode::nearest: {
            NNRef(in_ptr_, out_ptr_, N, C, ID, IH, IW, OD, OH, OW);
            break;
        }
        case InterpolateMode::linear_onnx: {
            linearOnnxRef(in_ptr_, out_ptr_, N, C, ID, IH, IW, OD, OH, OW);
            break;
        }
        case InterpolateMode::cubic: {
            cubicRef(in_ptr_, out_ptr_, N, C, IH, IW, OH, OW);
            break;
        }
        case InterpolateMode::linear: {
            float fz = (dataRank == 5) ? dataScales[dataRank - 3] : 1.f;
            float fy = dataScales[dataRank - 2];
            float fx = dataScales[dataRank - 1];

            bool isDownsample = (fx < 1.f) || (fy < 1.f) || (fz < 1.f);
            int kernel_width = 2;
            linearInterpolation(in_ptr_, out_ptr_, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW, kernel_width, isDownsample && antialias);
            break;
        }
        default: {
            IE_THROW() << "Interpolate layer has unsupported interpolate mode: " << mode;
        }
    }
}

void ov::intel_cpu::InterpolateRefExecutor::NNRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                                int OD, int OH, int OW) {
    int *index_d = static_cast<int*>(&indexTable[0]);
    int *index_h = static_cast<int*>(&indexTable[OD]);
    int *index_w = static_cast<int*>(&indexTable[OD + OH]);

    const float *in_ptr_f32 = reinterpret_cast<const float *>(in_ptr_);
    float *out_ptr_f32 = reinterpret_cast<float *>(out_ptr_);

    parallel_for3d(B, C, OD, [&](size_t b, size_t c, size_t od) {
        const float *in_ptr = in_ptr_f32 + (IW * IH * ID * C * b + IW * IH * ID * c + IW * IH * index_d[od]);
        float *out_ptr = out_ptr_f32 + (OW * OH * OD * C * b + OW * OH * OD * c + OW * OH * od);
        for (int oh = 0; oh < OH; oh++) {
            const float *in_ptr_h = in_ptr + (IW * index_h[oh]);
            float *out_ptr_h = out_ptr + (OW * oh);
            for (int ow = 0; ow < OW; ow++) {
                out_ptr_h[ow] = in_ptr_h[index_w[ow]];
            }
        }
    });
}

void ov::intel_cpu::InterpolateRefExecutor::linearOnnxRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                                        int OD, int OH, int OW) {
    std::vector<int*> indexPtr(MAX_INPUT_INTERPOLATE, 0);
    std::vector<float*> weightPtr(MAX_INPUT_INTERPOLATE, 0);
    // FrontTopLeft:0, FrontTopRight:1, FrontBottomLeft:2, FrontBottomRight:3,
    // EndTopLeft:4,   EndTopRight:5,   EndBottomLeft:6,   EndBottomRight:7
    // weight: Left:0, ritht:1, top:2, bottom:3, front:4, end:5

    int eltInGrid = (spatialDimSize > 2) ? MAX_INPUT_INTERPOLATE : ((spatialDimSize > 1) ? 4 : 2);
    int scratchLen = rnd_up(eltInGrid * OW * OH * OD, 16);

    indexPtr[0] = static_cast<int*>(&indexTable[0]);
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

    const float *in_ptr_f32 = reinterpret_cast<const float *>(in_ptr_);
    float *out_ptr_f32 = reinterpret_cast<float *>(out_ptr_);

    parallel_for2d(B, C, [&](size_t b, size_t c) {
        float *out_ptr_nc = out_ptr_f32 + (OD * OH * OW * C * b + OD * OH * OW * c);
        const float *in_ptr_nc = in_ptr_f32 + (ID * IH * IW * C * b + ID * IH * IW * c);
        // do not combined 1d/2d to 3d unified process to get rid of invalid computing.
        switch (spatialDimSize) {
            case 1:
                for (int i = 0; i < OW; i++) {
                    float src0 = in_ptr_nc[indexPtr[0][i]];
                    float src1 = in_ptr_nc[indexPtr[1][i]];

                    out_ptr_nc[i] = src0 * weightPtr[0][i] +
                                    src1 * weightPtr[1][i];
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

                    // float dstValue =
                    // weightPtr[4][i] * weightPtr[2][i] * weightPtr[0][i] * src000 +
                    // weightPtr[4][i] * weightPtr[2][i] * weightPtr[1][i] * src001 +
                    // weightPtr[4][i] * weightPtr[3][i] * weightPtr[0][i] * src010 +
                    // weightPtr[4][i] * weightPtr[3][i] * weightPtr[1][i] * src011 +
                    // weightPtr[5][i] * weightPtr[2][i] * weightPtr[0][i] * src100 +
                    // weightPtr[5][i] * weightPtr[2][i] * weightPtr[1][i] * src101 +
                    // weightPtr[5][i] * weightPtr[3][i] * weightPtr[0][i] * src110 +
                    // weightPtr[5][i] * weightPtr[3][i] * weightPtr[1][i] * src111;

                    out_ptr_nc[i] =
                            weightPtr[4][i] * (weightPtr[2][i] * (weightPtr[0][i] * src000 +
                                                                  weightPtr[1][i] * src001) +
                                               weightPtr[3][i] * (weightPtr[0][i] * src010 +
                                                                  weightPtr[1][i] * src011)) +
                            weightPtr[5][i] * (weightPtr[2][i] * (weightPtr[0][i] * src100 +
                                                                  weightPtr[1][i] * src101) +
                                               weightPtr[3][i] * (weightPtr[0][i] * src110 +
                                                                  weightPtr[1][i] * src111));
                }
                break;
            default:
                break;
        }
    });
}

void ov::intel_cpu::InterpolateRefExecutor::cubicRef(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int IH, int IW, int OH, int OW) {
    const int idxNum = 1;
    int *xOrigin = static_cast<int*>(&indexTable[0]);
    float *xFactor = reinterpret_cast<float*>(&indexTable[OW]);
    int *yOrigin = static_cast<int*>(&indexTable[(CUBIC_GRID_LEN + idxNum) * OW]);
    float *yFactor = reinterpret_cast<float*>(&indexTable[(CUBIC_GRID_LEN + idxNum) * OW + OH]);

    const float *in_ptr_f32 = reinterpret_cast<const float *>(in_ptr_);
    float *out_ptr_f32 = reinterpret_cast<float *>(out_ptr_);

    parallel_for4d(B, C, OH, OW, [&](size_t n, size_t c, size_t oy, size_t ox) {
        const float *in_ptr_nc = in_ptr_f32 + (IW * IH * C * n + IW * IH * c);
        float *out_ptr_nc = out_ptr_f32 + (OW * OH * C * n + OW * OH * c);

        int iy = yOrigin[oy];
        int ix = xOrigin[ox];

        float retY = 0.f;
        for (int y = iy - 1, i = 0; y <= iy + 2; y++, i++) {
            int yInRange = std::max(0, std::min(y, IH - 1));
            const float *in_ptr_nch = in_ptr_nc + IW * yInRange;
            float retX = 0.f;
            for (int x = ix - 1, j = 0; x <= ix + 2; x++, j++) {
                int xInRange = std::max(0, std::min(x, IW - 1));
                retX += xFactor[ox * CUBIC_GRID_LEN + j] * in_ptr_nch[xInRange];
            }
            retY += yFactor[oy * CUBIC_GRID_LEN + i] * retX;
        }
        out_ptr_nc[oy * OW + ox] = retY;
    });
}

float ov::intel_cpu::InterpolateRefExecutor::getValue(const uint8_t *base, size_t offset, InferenceEngine::Precision prec) {
    const uint8_t *baseOffset = base + offset;
    switch (prec) {
        case Precision::U8: {
            return static_cast<float>(*baseOffset);
            break;
        }
        case Precision::I8: {
            const int8_t *valuePtr = reinterpret_cast<const int8_t *>(baseOffset);
            return static_cast<float>(*valuePtr);
            break;
        }
        case Precision::BF16: {
            const uint16_t *valuePtr = reinterpret_cast<const uint16_t *>(baseOffset);
            return bfloat16_t::from_bits(*valuePtr);
            break;
        }
        case Precision::FP32: {
            const float *valuePtr = reinterpret_cast<const float *>(baseOffset);
            return *valuePtr;
            break;
        }
        default: {
            IE_THROW() << "Interpolate layer does not support precision: " << prec;
            break;
        }
    }
}

void ov::intel_cpu::InterpolateRefExecutor::setValue(uint8_t *base, size_t offset, float value, InferenceEngine::Precision prec) {
    uint8_t *baseOffset = base + offset;
    switch (prec) {
        case Precision::U8: {
            uint8_t data = static_cast<uint8_t>(value < 0 ? 0 : value);
            cpu_memcpy(baseOffset, &data, 1);
            break;
        }
        case Precision::I8: {
            int8_t data = static_cast<int8_t>(value);
            cpu_memcpy(baseOffset, &data, 1);
            break;
        }
        case Precision::BF16: {
            uint16_t data = bfloat16_t(value).to_bits();
            cpu_memcpy(baseOffset, &data, 2);
            break;
        }
        case Precision::FP32: {
            cpu_memcpy(baseOffset, &value, sizeof(float));
            break;
        }
        default: {
            IE_THROW() << "Interpolate layer does not support precision: " << prec;
            break;
        }
    }
}

void ov::intel_cpu::InterpolateRefExecutor::linearInterpolation(const uint8_t *in_ptr_, uint8_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                                              float fx, float fy, float fz, int OD, int OH, int OW, int kernel_width, bool antialias) {
    if (IW == OW && IH == OH && ID == OD) {
        size_t spatialDimSize = IW * IH * ID;
        // TODO: enable when fusing into interp with linear mode will support
        if (/*fusedWith.empty() &&*/ inputPrec == outputPrec) {
            size_t size = B * C * spatialDimSize * srcDataSize;
            cpu_memcpy(out_ptr_, in_ptr_, size);
        } else {
            parallel_for2d(B, C, [&](size_t b, size_t c) {
                const uint8_t *in_ptr_nc = in_ptr_ + (spatialDimSize * C * b + spatialDimSize * c) * srcDataSize;
                uint8_t *out_ptr_nc = out_ptr_ + (spatialDimSize * C * b + spatialDimSize * c) * dstDataSize;
                for (size_t i = 0; i < spatialDimSize; i++) {
                    float dstValue = getValue(in_ptr_nc, i * srcDataSize, inputPrec);
                    setValue(out_ptr_nc, i * dstDataSize, dstValue, outputPrec);
                }
            });
        }
        return;
    }

    float ax = antialias ? fx : 1.0f;
    float ay = antialias ? fy : 1.0f;
    float az = antialias ? fz : 1.0f;

    int rx = (fx > 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ax));
    int ry = (fy > 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ay));
    int rz = (fz > 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / az));

    int diaOD = 2 * rz + 1;
    int diaOH = 2 * ry + 1;
    int diaOW = 2 * rx + 1;
    int sizeOD = OD * diaOD;
    int sizeOH = OH * diaOH;
    int sizeOW = OW * diaOW;

    float *weightTable = reinterpret_cast<float*>(&indexTable[0]);
    float *weightOD = static_cast<float*>(&weightTable[0]);
    float *weightOH = static_cast<float*>(&weightTable[sizeOD]);
    float *weightOW = static_cast<float*>(&weightTable[sizeOD + sizeOH]);

    int *idxTable = static_cast<int*>(&indexTable[sizeOD + sizeOH + sizeOW]);
    int *idxOD = static_cast<int*>(&idxTable[0]);
    int *idxOH = static_cast<int*>(&idxTable[sizeOD]);
    int *idxOW = static_cast<int*>(&idxTable[sizeOD + sizeOH]);

    parallel_for2d(B, C, [&](size_t b, size_t c) {
        const uint8_t *in_ptr_nc = in_ptr_ + (IW * IH * ID * C * b + IW * IH * ID * c) * srcDataSize;
        uint8_t *out_ptr_nc = out_ptr_ + (OW * OH * OD * C * b + OW * OH * OD * c) * dstDataSize;
        for (size_t oz = 0; oz < OD; oz++) {
            uint8_t *out_ptr_ncd = out_ptr_nc + (OW * OH * oz) * dstDataSize;
            for (size_t oy = 0; oy < OH; oy++) {
                uint8_t *out_ptr_ncdh = out_ptr_ncd + (OW * oy) * dstDataSize;
                for (size_t ox = 0; ox < OW; ox++) {
                    float sum = 0.f;
                    float wsum = 0.f;

                    // this comment explains the original algo.
                    // for (int z = iz_r - rz; z <= iz_r + rz; z++) {
                    //    for (int y = iy_r - ry; y <= iy_r + ry; y++) {
                    //        for (int x = ix_r - rx; x <= ix_r + rx; x++) {
                    //            bool is_continue =  z < 0                     ||
                    //                                y < 0                     ||
                    //                                x < 0                     ||
                    //                                z >= static_cast<int>(ID) ||
                    //                                y >= static_cast<int>(IH) ||
                    //                                x >= static_cast<int>(IW);
                    //            if (is_continue)
                    //                continue;

                    //            float dx = ix - x;
                    //            float dy = iy - y;
                    //            float dz = iz - z;

                    //            float w = ax * triangleCoeff(ax * dx) *
                    //                      ay * triangleCoeff(ay * dy) *
                    //                      az * triangleCoeff(az * dz);

                    //            sum += w * getValue(in_ptr_nc, (z * IH * IW + y * IW + x) * srcDataSize, inputPrec);
                    //            wsum += w;
                    //        }
                    //    }
                    //}

                    for (int iz = 0; iz < diaOD; iz++) {
                        if (weightOD[oz * diaOD + iz] == 0.f)
                            continue;
                        for (int iy = 0; iy < diaOH; iy++) {
                            if (weightOH[oy * diaOH + iy] == 0.f) {
                                continue;
                            }
                            for (int ix = 0; ix < diaOW; ix++) {
                                if (weightOW[ox * diaOW + ix] == 0.f) {
                                    continue;
                                }
                                float w = weightOD[oz * diaOD + iz] * weightOH[oy * diaOH + iy] * weightOW[ox * diaOW + ix];
                                float value = getValue(in_ptr_nc,
                                                       (idxOD[oz * diaOD + iz] * IH * IW + idxOH[oy * diaOH + iy] * IW + idxOW[ox * diaOW + ix])
                                                       * srcDataSize, inputPrec);

                                sum += w * value;
                                wsum += w;
                            }
                        }
                    }

                    if (!wsum) {
                        setValue(out_ptr_ncdh, ox * dstDataSize, 0.f, outputPrec);
                    } else {
                        float dst_value = sum / wsum;
                        setValue(out_ptr_ncdh, ox * dstDataSize, dst_value, outputPrec);
                    }
                }
            }
        }
    });
}

