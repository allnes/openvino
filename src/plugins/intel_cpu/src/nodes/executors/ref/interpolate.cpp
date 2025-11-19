// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interpolate.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <oneapi/dnnl/dnnl.hpp>
#include <cstring>
#include <vector>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/common/cpu_memcpy.h"
#include "nodes/executors/interpolate.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "utils/general_utils.h"
#include <common/float16.hpp>

using namespace ov::intel_cpu;

static inline bool isFloatCompatible(ov::element::Type prc) {
    return prc == ov::element::f32 || prc == ov::element::bf16 || prc == ov::element::f16 || prc == ov::element::f64;
}

bool InterpolateRefExecutor::init(const InterpolateAttrs& interpolateAttrs,
                                  const std::vector<MemoryDescPtr>& srcDescs,
                                  const std::vector<MemoryDescPtr>& dstDescs,
                                  const dnnl::primitive_attr& attr) {
    // Reuse common shape-agnostic initialization (tables, dims, etc.)
    return InterpolateExecutor::init(interpolateAttrs, srcDescs, dstDescs, attr);
}

// Basic typed load/store for fp32/bf16/u8/s8
float InterpolateRefExecutor::getValue(const uint8_t* base, size_t offset, ov::element::Type prec) {
    switch (prec) {
    case ov::element::f32:
        return *reinterpret_cast<const float*>(base + offset);
    case ov::element::f16: {
        auto v = *reinterpret_cast<const dnnl::impl::float16_t*>(base + offset);
        return static_cast<float>(v);
    }
    case ov::element::bf16: {
        uint16_t v = *reinterpret_cast<const uint16_t*>(base + offset);
        uint32_t tmp = static_cast<uint32_t>(v) << 16;
        float f;
        std::memcpy(&f, &tmp, sizeof(float));
        return f;
    }
    case ov::element::u8:
        return static_cast<float>(*reinterpret_cast<const uint8_t*>(base + offset));
    case ov::element::i8:
        return static_cast<float>(*reinterpret_cast<const int8_t*>(base + offset));
    default:
        OPENVINO_THROW("InterpolateRefExecutor: unsupported input precision");
    }
}

void InterpolateRefExecutor::setValue(uint8_t* base, size_t offset, float value, ov::element::Type prec) {
    switch (prec) {
    case ov::element::f32: {
        *reinterpret_cast<float*>(base + offset) = value;
        break;
    }
    case ov::element::f16: {
        dnnl::impl::float16_t v = static_cast<dnnl::impl::float16_t>(value);
        *reinterpret_cast<dnnl::impl::float16_t*>(base + offset) = v;
        break;
    }
    case ov::element::bf16: {
        // simple round-to-nearest-even
        uint32_t tmp;
        std::memcpy(&tmp, &value, sizeof(float));
        uint16_t v = static_cast<uint16_t>((tmp + 0x8000) >> 16);
        *reinterpret_cast<uint16_t*>(base + offset) = v;
        break;
    }
    case ov::element::u8: {
        int iv = static_cast<int>(std::nearbyint(value));
        iv = std::max(0, std::min(255, iv));
        *reinterpret_cast<uint8_t*>(base + offset) = static_cast<uint8_t>(iv);
        break;
    }
    case ov::element::i8: {
        int iv = static_cast<int>(std::nearbyint(value));
        iv = std::max(-128, std::min(127, iv));
        *reinterpret_cast<int8_t*>(base + offset) = static_cast<int8_t>(iv);
        break;
    }
    default:
        OPENVINO_THROW("InterpolateRefExecutor: unsupported output precision");
    }
}

void InterpolateRefExecutor::NNRef(const uint8_t* in_ptr_,
                                   uint8_t* out_ptr_,
                                   int B,
                                   int C,
                                   int ID,
                                   int IH,
                                   int IW,
                                   int OD,
                                   int OH,
                                   int OW) {
    // indexTable layout per buildTblNN: [OD][OH][OW]
    auto* index_d = const_cast<int*>(reinterpret_cast<const int*>(indexTable.data()));
    auto* index_h = const_cast<int*>(reinterpret_cast<const int*>(&indexTable[OD]));
    auto* index_w = const_cast<int*>(reinterpret_cast<const int*>(&indexTable[OD + OH]));

    const auto inputPrec = interpAttrs.inPrc;
    const auto outputPrec = interpAttrs.outPrc;
    const size_t srcDataSizeL = srcDataSize;
    const size_t dstDataSizeL = dstDataSize;

    ov::parallel_for3d(B, C, OD, [&](size_t b, size_t c, size_t od) {
        const uint8_t* in_ptr = in_ptr_ + (IW * IH * ID * C * b + IW * IH * ID * c + IW * IH * index_d[od]) * srcDataSizeL;
        uint8_t* out_ptr = out_ptr_ + (OW * OH * OD * C * b + OW * OH * OD * c + OW * OH * od) * dstDataSizeL;
        for (int oh = 0; oh < OH; oh++) {
            const uint8_t* in_ptr_h = in_ptr + (IW * index_h[oh]) * srcDataSizeL;
            uint8_t* out_ptr_h = out_ptr + (OW * oh) * dstDataSizeL;
            for (int ow = 0; ow < OW; ow++) {
                float v = getValue(in_ptr_h, static_cast<size_t>(index_w[ow]) * srcDataSizeL, inputPrec);
                setValue(out_ptr_h, static_cast<size_t>(ow) * dstDataSizeL, v, outputPrec);
            }
        }
    });
}

void InterpolateRefExecutor::linearOnnxRef(const uint8_t* in_ptr_,
                                           uint8_t* out_ptr_,
                                           int B,
                                           int C,
                                           int ID,
                                           int IH,
                                           int IW,
                                           int OD,
                                           int OH,
                                           int OW) {
    // The buildTblLinearOnnx stores two index grids (L/R) and two weight grids per axis.
    // We follow the planar scratch layout used in the common init.
    std::vector<int*> indexPtr(MAX_INPUT_INTERPOLATE, nullptr);
    std::vector<float*> weightPtr(MAX_INPUT_INTERPOLATE, nullptr);

    int eltInGrid = 2;
    if (spatialDimSize > 1) eltInGrid = 4;
    if (spatialDimSize > 2) eltInGrid = MAX_INPUT_INTERPOLATE;
    int scratchLen = rnd_up(eltInGrid * OW * OH * OD, 16);

    indexPtr[0] = const_cast<int*>(reinterpret_cast<const int*>(indexTable.data()));
    indexPtr[1] = const_cast<int*>(reinterpret_cast<const int*>(&indexTable[OW * OH * OD]));
    weightPtr[0] = reinterpret_cast<float*>(const_cast<int*>(&indexTable[scratchLen]));
    weightPtr[1] = reinterpret_cast<float*>(const_cast<int*>(&indexTable[scratchLen + OW * OH * OD]));
    if (spatialDimSize > 1) {
        indexPtr[2] = const_cast<int*>(reinterpret_cast<const int*>(&indexTable[2 * OW * OH * OD]));
        indexPtr[3] = const_cast<int*>(reinterpret_cast<const int*>(&indexTable[3 * OW * OH * OD]));
        weightPtr[2] = reinterpret_cast<float*>(const_cast<int*>(&indexTable[scratchLen + 2 * OW * OH * OD]));
        weightPtr[3] = reinterpret_cast<float*>(const_cast<int*>(&indexTable[scratchLen + 3 * OW * OH * OD]));
    }
    if (spatialDimSize > 2) {
        indexPtr[4] = const_cast<int*>(reinterpret_cast<const int*>(&indexTable[4 * OW * OH * OD]));
        indexPtr[5] = const_cast<int*>(reinterpret_cast<const int*>(&indexTable[5 * OW * OH * OD]));
        indexPtr[6] = const_cast<int*>(reinterpret_cast<const int*>(&indexTable[6 * OW * OH * OD]));
        indexPtr[7] = const_cast<int*>(reinterpret_cast<const int*>(&indexTable[7 * OW * OH * OD]));
        weightPtr[4] = reinterpret_cast<float*>(const_cast<int*>(&indexTable[scratchLen + 4 * OW * OH * OD]));
        weightPtr[5] = reinterpret_cast<float*>(const_cast<int*>(&indexTable[scratchLen + 5 * OW * OH * OD]));
    }

    const auto inputPrec = interpAttrs.inPrc;
    const auto outputPrec = interpAttrs.outPrc;
    const size_t srcDataSizeL = srcDataSize;
    const size_t dstDataSizeL = dstDataSize;

    if (std::getenv("OV_DEBUG_INTERP_PAD")) {
        static bool logged_planar = false;
        if (!logged_planar) {
            logged_planar = true;
            std::fprintf(stderr,
                         "[InterpolateRefExecutor] layout=%d spatial=%d IH=%d IW=%d firstXIdx=%d firstYIdx=%d\n",
                         static_cast<int>(interpAttrs.layout),
                         spatialDimSize,
                         IH,
                         IW,
                         indexPtr[0][0],
                         (spatialDimSize > 1) ? indexPtr[2][0] : -1);
        }
    }

    switch (spatialDimSize) {
    case 1: {
        const bool byCh = interpAttrs.layout == InterpolateLayoutType::by_channel;
        ov::parallel_for4d(B, C, OD, OH, [&](size_t b, size_t c, size_t d, size_t h) {
            uint8_t* out_ptr_nc = byCh ? out_ptr_ + (OD * OH * OW * C * b + OH * OW * d + OW * h) * dstDataSizeL
                                       : out_ptr_ + (OD * OH * OW * C * b + OD * OH * OW * c + OH * OW * d + OW * h) * dstDataSizeL;
            const uint8_t* in_ptr_nc = byCh ? in_ptr_ + (ID * IH * IW * C * b + IH * IW * d + IW * h) * srcDataSizeL
                                            : in_ptr_ + (ID * IH * IW * C * b + ID * IH * IW * c + IH * IW * d + IW * h) * srcDataSizeL;
            auto off = [&](int idx) {
                if (byCh) {
                    return (static_cast<size_t>(idx) * C + c) * srcDataSizeL;
                } else {
                    return static_cast<size_t>(idx) * srcDataSizeL;
                }
            };
            auto ooff = [&](int i) {
                if (byCh) {
                    return (static_cast<size_t>(i) * C + c) * dstDataSizeL;
                } else {
                    return static_cast<size_t>(i) * dstDataSizeL;
                }
            };
            for (int i = 0; i < OW; i++) {
                float src0 = getValue(in_ptr_nc, off(indexPtr[0][i]), inputPrec);
                float src1 = getValue(in_ptr_nc, off(indexPtr[1][i]), inputPrec);
                float dst = src0 * weightPtr[0][i] + src1 * weightPtr[1][i];
                setValue(out_ptr_nc, ooff(i), dst, outputPrec);
            }
        });
        break;
    }
    case 2:
        ov::parallel_for3d(B, C, OD, [&](size_t b, size_t c, size_t d) {
            const bool byCh = interpAttrs.layout == InterpolateLayoutType::by_channel;
            const uint8_t* in_ptr_nc = byCh ? in_ptr_ + (ID * IH * IW * C * b + IH * IW * d) * srcDataSizeL
                                            : in_ptr_ + (ID * IH * IW * C * b + ID * IH * IW * C * c + IH * IW * d) * srcDataSizeL;
            uint8_t* out_ptr_nc = byCh ? out_ptr_ + (OD * OH * OW * C * b + OH * OW * d) * dstDataSizeL
                                       : out_ptr_ + (OD * OH * OW * C * b + OD * OH * OW * C * c + OH * OW * d) * dstDataSizeL;
            if (std::getenv("OV_DEBUG_INTERP_PAD") && b == 0 && c == 0 && d == 0) {
                const int dump = std::min(8, OH * OW);
                std::fprintf(stderr, "[InterpolateRefExecutor] idxL/R/T/B preview:");
                for (int i = 0; i < dump; ++i) {
                    std::fprintf(stderr,
                                 " [%d,%d,%d,%d]",
                                 indexPtr[0][i],
                                 indexPtr[1][i],
                                 indexPtr[2][i],
                                 indexPtr[3][i]);
                }
                std::fprintf(stderr, "\n");
            }
            for (int i = 0; i < OH * OW; i++) {
                auto off = [&](int idx){ return byCh ? (static_cast<size_t>(idx) * C + c) * srcDataSizeL
                                                     : static_cast<size_t>(idx) * srcDataSizeL; };
                float src00 = getValue(in_ptr_nc, off(indexPtr[0][i]), inputPrec);
                float src01 = getValue(in_ptr_nc, off(indexPtr[1][i]), inputPrec);
                float src10 = getValue(in_ptr_nc, off(indexPtr[2][i]), inputPrec);
                float src11 = getValue(in_ptr_nc, off(indexPtr[3][i]), inputPrec);
                if (std::getenv("OV_DEBUG_INTERP_PAD") && b == 0 && c == 0 && d == 0 && i < 8) {
                    std::fprintf(stderr,
                                 "[InterpolateRefExecutor] samples i=%d: %.4f %.4f %.4f %.4f | weights %.4f %.4f %.4f %.4f\n",
                                 i,
                                 src00,
                                 src01,
                                 src10,
                                 src11,
                                 weightPtr[0][i],
                                 weightPtr[1][i],
                                 weightPtr[2][i],
                                 weightPtr[3][i]);
                }
                float v = src00 * weightPtr[0][i] + src01 * weightPtr[1][i];
                float w = src10 * weightPtr[0][i] + src11 * weightPtr[1][i];
                float dst = v * weightPtr[2][i] + w * weightPtr[3][i];
                const size_t ooff = byCh ? (static_cast<size_t>(i) * C + c) * dstDataSizeL
                                         : static_cast<size_t>(i) * dstDataSizeL;
                setValue(out_ptr_nc, ooff, dst, outputPrec);
            }
        });
        break;
    case 3: {
        const bool byCh = interpAttrs.layout == InterpolateLayoutType::by_channel;
        ov::parallel_for2d(B, C, [&](size_t b, size_t c) {
            uint8_t* out_ptr_nc = byCh ? out_ptr_ + (OD * OH * OW * C * b) * dstDataSizeL
                                       : out_ptr_ + (OD * OH * OW * C * b + OD * OH * OW * c) * dstDataSizeL;
            const uint8_t* in_ptr_nc = byCh ? in_ptr_ + (ID * IH * IW * C * b) * srcDataSizeL
                                            : in_ptr_ + (ID * IH * IW * C * b + ID * IH * IW * c) * srcDataSizeL;
            auto off = [&](int idx) {
                if (byCh) {
                    return (static_cast<size_t>(idx) * C + c) * srcDataSizeL;
                } else {
                    return static_cast<size_t>(idx) * srcDataSizeL;
                }
            };
            auto ooff = [&](int i) {
                if (byCh) {
                    return (static_cast<size_t>(i) * C + c) * dstDataSizeL;
                } else {
                    return static_cast<size_t>(i) * dstDataSizeL;
                }
            };
            for (int i = 0; i < OD * OH * OW; i++) {
                // Trilinear via separable weights
                float f00 = getValue(in_ptr_nc, off(indexPtr[0][i]), inputPrec) * weightPtr[0][i] +
                            getValue(in_ptr_nc, off(indexPtr[1][i]), inputPrec) * weightPtr[1][i];
                float f01 = getValue(in_ptr_nc, off(indexPtr[2][i]), inputPrec) * weightPtr[0][i] +
                            getValue(in_ptr_nc, off(indexPtr[3][i]), inputPrec) * weightPtr[1][i];
                float front = f00 * weightPtr[2][i] + f01 * weightPtr[3][i];

                float b00 = getValue(in_ptr_nc, off(indexPtr[4][i]), inputPrec) * weightPtr[0][i] +
                            getValue(in_ptr_nc, off(indexPtr[5][i]), inputPrec) * weightPtr[1][i];
                float b01 = getValue(in_ptr_nc, off(indexPtr[6][i]), inputPrec) * weightPtr[0][i] +
                            getValue(in_ptr_nc, off(indexPtr[7][i]), inputPrec) * weightPtr[1][i];
                float back = b00 * weightPtr[2][i] + b01 * weightPtr[3][i];

                float dst = front * weightPtr[4][i] + back * weightPtr[5][i];
                setValue(out_ptr_nc, ooff(i), dst, outputPrec);
            }
        });
        break;
    }
    default:
        OPENVINO_THROW("InterpolateRefExecutor: unsupported spatialDimSize for linear_onnx");
    }
}

void InterpolateRefExecutor::linearRef(const uint8_t* in_ptr_,
                                       uint8_t* out_ptr_,
                                       int B,
                                       int C,
                                       int ID,
                                       int IH,
                                       int IW,
                                       int OD,
                                       int OH,
                                       int OW) {
    // Fallback to linearOnnxRef which is spec-compatible for our reference
    linearOnnxRef(in_ptr_, out_ptr_, B, C, ID, IH, IW, OD, OH, OW);
}

void InterpolateRefExecutor::pillowRef(const uint8_t* in_ptr_,
                                       uint8_t* out_ptr_,
                                       int B,
                                       int C,
                                       int IH,
                                       int IW,
                                       int OH,
                                       int OW) {
    if (pillowTable.empty()) {
        OPENVINO_THROW("Pillow interpolation tables are not initialized");
    }
    size_t offset = 0;
    const int filterLenX = pillowTable[offset];
    const int filterLenY = pillowTable[offset + 1];
    offset += 2;
    auto* weightX = reinterpret_cast<float*>(&pillowTable[offset]);
    offset += static_cast<size_t>(filterLenX) * OW;
    auto* weightY = reinterpret_cast<float*>(&pillowTable[offset]);
    offset += static_cast<size_t>(filterLenY) * OH;
    auto* indexX = static_cast<int*>(&pillowTable[offset]);
    offset += 2 * OW;
    auto* indexY = static_cast<int*>(&pillowTable[offset]);

    const bool xPass = IW != OW;
    const bool yPass = IH != OH;
    const auto inputPrec = interpAttrs.inPrc;
    const auto outputPrec = interpAttrs.outPrc;
    const size_t srcDataSizeL = srcDataSize;
    const size_t dstDataSizeL = dstDataSize;

    if (xPass && yPass) {
        OPENVINO_ASSERT(!pillow_working_buf.empty(),
                        "Pillow interpolation requires intermediate buffer but it is not allocated");
    }

    const auto bc_loop = [&](size_t b, size_t c) {
        const uint8_t* in_ptr_nc =
            in_ptr_ + (static_cast<size_t>(IW) * IH * C * b + static_cast<size_t>(IW) * IH * c) * srcDataSizeL;
        uint8_t* out_ptr_nc =
            out_ptr_ + (static_cast<size_t>(OW) * OH * C * b + static_cast<size_t>(OW) * OH * c) * dstDataSizeL;

        uint8_t* xpass_out_ptr_nc = nullptr;
        const uint8_t* ypass_in_ptr_nc = nullptr;

        if (xPass && yPass) {
            const size_t parallel_num = static_cast<size_t>(B) * static_cast<size_t>(C);
            if (parallel_num < m_threads_num) {
                const size_t base =
                    (static_cast<size_t>(b) * static_cast<size_t>(C) + c) * static_cast<size_t>(OW) * IH;
                xpass_out_ptr_nc = pillow_working_buf.data() + base * srcDataSizeL;
            } else {
                const size_t threadIdx = static_cast<size_t>(parallel_get_thread_num());
                const size_t buffer_size = static_cast<size_t>(OW) * IH;
                xpass_out_ptr_nc = pillow_working_buf.data() + threadIdx * buffer_size * srcDataSizeL;
            }
            ypass_in_ptr_nc = xpass_out_ptr_nc;
        } else if (xPass && !yPass) {
            xpass_out_ptr_nc = out_ptr_nc;
        } else if (!xPass && yPass) {
            ypass_in_ptr_nc = in_ptr_nc;
        } else {
            cpu_memcpy(out_ptr_nc, in_ptr_nc, static_cast<size_t>(OH) * OW * dstDataSizeL);
            return;
        }

        if (xPass) {
            for (int ih = 0; ih < IH; ++ih) {
                for (int ow = 0; ow < OW; ++ow) {
                    const int filterS = indexX[ow * 2];
                    const int filterL = indexX[ow * 2 + 1];
                    const float* weight = &weightX[ow * filterLenX];
                    float result = 0.F;
                    for (int f = 0; f < filterL; ++f) {
                        const float pixel = getValue(in_ptr_nc,
                                                     (static_cast<size_t>(ih) * IW + filterS + f) * srcDataSizeL,
                                                     inputPrec);
                        result += pixel * weight[f];
                    }
                    if (!isFloatCompatible(outputPrec)) {
                        result = static_cast<float>(static_cast<int>(result >= 0.0F ? result + 0.5F : result - 0.5F));
                    }
                    setValue(xpass_out_ptr_nc,
                             (static_cast<size_t>(ih) * OW + ow) * dstDataSizeL,
                             result,
                             outputPrec);
                }
            }
        }

        if (yPass) {
            for (int oh = 0; oh < OH; ++oh) {
                const int filterS = indexY[oh * 2];
                const int filterL = indexY[oh * 2 + 1];
                const float* weight = &weightY[oh * filterLenY];
                for (int ow = 0; ow < OW; ++ow) {
                    float result = 0.F;
                    for (int f = 0; f < filterL; ++f) {
                        const float pixel = getValue(ypass_in_ptr_nc,
                                                     (static_cast<size_t>(filterS + f) * OW + ow) * srcDataSizeL,
                                                     inputPrec);
                        result += pixel * weight[f];
                    }
                    if (!isFloatCompatible(outputPrec)) {
                        result = static_cast<float>(static_cast<int>(result >= 0.0F ? result + 0.5F : result - 0.5F));
                    }
                    setValue(out_ptr_nc, (static_cast<size_t>(oh) * OW + ow) * dstDataSizeL, result, outputPrec);
                }
            }
        }
    };

    parallel_nt_static(static_cast<int>(m_threads_num), [&](const int ithr, const int nthr) {
        for_2d(ithr, nthr, B, C, bc_loop);
    });
}

void InterpolateRefExecutor::pillowRefNCHWAsNHWC(const uint8_t* in_ptr_,
                                                 uint8_t* out_ptr_,
                                                 int B,
                                                 int C,
                                                 int IH,
                                                 int IW,
                                                 int OH,
                                                 int OW) {
    if (pillowTable.empty()) {
        OPENVINO_THROW("Pillow interpolation tables are not initialized");
    }
    size_t offset = 0;
    const int filterLenX = pillowTable[offset];
    const int filterLenY = pillowTable[offset + 1];
    offset += 2;
    auto* weightX = reinterpret_cast<float*>(&pillowTable[offset]);
    offset += static_cast<size_t>(filterLenX) * OW;
    auto* weightY = reinterpret_cast<float*>(&pillowTable[offset]);
    offset += static_cast<size_t>(filterLenY) * OH;
    auto* indexX = static_cast<int*>(&pillowTable[offset]);
    offset += 2 * OW;
    auto* indexY = static_cast<int*>(&pillowTable[offset]);

    const bool xPass = IW != OW;
    const bool yPass = IH != OH;
    const auto inputPrec = interpAttrs.inPrc;
    const auto outputPrec = interpAttrs.outPrc;
    const size_t srcDataSizeL = srcDataSize;
    const size_t dstDataSizeL = dstDataSize;

    if (xPass && yPass) {
        OPENVINO_ASSERT(!pillow_working_buf.empty(),
                        "Pillow interpolation requires intermediate buffer but it is not allocated");
    }

    const auto b_loop = [&](size_t b) {
        const uint8_t* in_ptr_b = in_ptr_ + static_cast<size_t>(b) * IH * IW * C * srcDataSizeL;
        uint8_t* out_ptr_b = out_ptr_ + static_cast<size_t>(b) * OH * OW * C * dstDataSizeL;

        uint8_t* xpass_out_ptr_b = nullptr;
        const uint8_t* ypass_in_ptr_b = nullptr;

        if (xPass && yPass) {
            const size_t buffer_size = static_cast<size_t>(IH) * OW * C;
            const size_t parallel_num = static_cast<size_t>(B);
            if (parallel_num < m_threads_num) {
                xpass_out_ptr_b = pillow_working_buf.data() + static_cast<size_t>(b) * buffer_size * srcDataSizeL;
            } else {
                const size_t threadIdx = static_cast<size_t>(parallel_get_thread_num());
                xpass_out_ptr_b = pillow_working_buf.data() + threadIdx * buffer_size * srcDataSizeL;
            }
            ypass_in_ptr_b = xpass_out_ptr_b;
        } else if (xPass && !yPass) {
            xpass_out_ptr_b = out_ptr_b;
        } else if (!xPass && yPass) {
            ypass_in_ptr_b = in_ptr_b;
        } else {
            cpu_memcpy(out_ptr_b, in_ptr_b, static_cast<size_t>(OH) * OW * C * dstDataSizeL);
            return;
        }

        if (xPass) {
            for (int ih = 0; ih < IH; ++ih) {
                for (int ow = 0; ow < OW; ++ow) {
                    const int filterS = indexX[ow * 2];
                    const int filterL = indexX[ow * 2 + 1];
                    const float* weight = &weightX[ow * filterLenX];
                    for (int c = 0; c < C; ++c) {
                        float result = 0.F;
                        for (int f = 0; f < filterL; ++f) {
                            const float pixel = getValue(in_ptr_b,
                                                         ((static_cast<size_t>(ih) * IW + filterS + f) * C + c) *
                                                             srcDataSizeL,
                                                         inputPrec);
                            result += pixel * weight[f];
                        }
                        if (!isFloatCompatible(outputPrec)) {
                            result =
                                static_cast<float>(static_cast<int>(result >= 0.0F ? result + 0.5F : result - 0.5F));
                        }
                        setValue(xpass_out_ptr_b,
                                 ((static_cast<size_t>(ih) * OW + ow) * C + c) * dstDataSizeL,
                                 result,
                                 outputPrec);
                    }
                }
            }
        }

        if (yPass) {
            for (int oh = 0; oh < OH; ++oh) {
                const int filterS = indexY[oh * 2];
                const int filterL = indexY[oh * 2 + 1];
                const float* weight = &weightY[oh * filterLenY];
                for (int ow = 0; ow < OW; ++ow) {
                    for (int c = 0; c < C; ++c) {
                        float result = 0.F;
                        for (int f = 0; f < filterL; ++f) {
                            const float pixel = getValue(ypass_in_ptr_b,
                                                         (((filterS + f) * OW + ow) * C + c) * srcDataSizeL,
                                                         inputPrec);
                            result += pixel * weight[f];
                        }
                        if (!isFloatCompatible(outputPrec)) {
                            result =
                                static_cast<float>(static_cast<int>(result >= 0.0F ? result + 0.5F : result - 0.5F));
                        }
                        setValue(out_ptr_b,
                                 ((static_cast<size_t>(oh) * OW + ow) * C + c) * dstDataSizeL,
                                 result,
                                 outputPrec);
                    }
                }
            }
        }
    };

    parallel_nt_static(static_cast<int>(m_threads_num), [&](const int ithr, const int nthr) {
        for_1d(ithr, nthr, B, b_loop);
    });
}

void InterpolateRefExecutor::exec(const std::vector<MemoryCPtr>& src,
                                  const std::vector<MemoryPtr>& dst,
                                  [[maybe_unused]] const void* post_ops_data_) {
    const auto* in_ptr_ = padPreprocess(src, dst);
    auto* out_ptr_ = dst[0]->getDataAs<uint8_t>();

    size_t N = srcDimPad5d[0];
    size_t C = srcDimPad5d[1];
    size_t ID = srcDimPad5d[2];
    size_t IH = srcDimPad5d[3];
    size_t IW = srcDimPad5d[4];
    size_t OD = dstDim5d[2];
    size_t OH = dstDim5d[3];
    size_t OW = dstDim5d[4];

    const bool isPurePadSizes = interpAttrs.hasPad &&
                                interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::sizes &&
                                srcDimPad5d == dstDim5d;
    if (isPurePadSizes) {
        if (std::getenv("OV_DEBUG_INTERP_PAD")) {
            std::fprintf(stderr,
                         "[InterpolateRefExecutor] pure pad sizes fast-path: bytes=%zu\n",
                         dst[0]->getDesc().getCurrentMemSize());
        }
        const size_t bytesToCopy = dst[0]->getDesc().getCurrentMemSize();
        cpu_memcpy(out_ptr_, in_ptr_, bytesToCopy);
        return;
    }

    switch (interpAttrs.mode) {
    case InterpolateMode::nearest:
        NNRef(in_ptr_, out_ptr_, static_cast<int>(N), static_cast<int>(C), static_cast<int>(ID), static_cast<int>(IH), static_cast<int>(IW),
              static_cast<int>(OD), static_cast<int>(OH), static_cast<int>(OW));
        break;
    case InterpolateMode::linear_onnx:
        linearOnnxRef(in_ptr_, out_ptr_, static_cast<int>(N), static_cast<int>(C), static_cast<int>(ID), static_cast<int>(IH),
                      static_cast<int>(IW), static_cast<int>(OD), static_cast<int>(OH), static_cast<int>(OW));
        break;
    case InterpolateMode::linear:
        linearRef(in_ptr_, out_ptr_, static_cast<int>(N), static_cast<int>(C), static_cast<int>(ID), static_cast<int>(IH), static_cast<int>(IW),
                  static_cast<int>(OD), static_cast<int>(OH), static_cast<int>(OW));
        break;
    case InterpolateMode::bilinear_pillow:
    case InterpolateMode::bicubic_pillow:
        if (interpAttrs.NCHWAsNHWC) {
            pillowRefNCHWAsNHWC(in_ptr_,
                                out_ptr_,
                                static_cast<int>(N),
                                static_cast<int>(C),
                                static_cast<int>(IH),
                                static_cast<int>(IW),
                                static_cast<int>(OH),
                                static_cast<int>(OW));
        } else {
            pillowRef(in_ptr_,
                      out_ptr_,
                      static_cast<int>(N),
                      static_cast<int>(C),
                      static_cast<int>(IH),
                      static_cast<int>(IW),
                      static_cast<int>(OH),
                      static_cast<int>(OW));
        }
        break;
    default:
        OPENVINO_THROW("InterpolateRefExecutor: mode not supported in ref backend: ", static_cast<int>(interpAttrs.mode));
    }
}
