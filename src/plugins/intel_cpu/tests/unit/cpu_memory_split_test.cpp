// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>

#include "cpu_memory.h"
#include "memory_desc/cpu_blocked_memory_desc.h"

namespace ov::intel_cpu::test {

TEST(CpuMemorySplit, HorizontalPackedWeightsPreserveNibbleBoundaries) {
    const dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    for (const auto type : {ov::element::i4, ov::element::u4}) {
        for (const size_t rows : {18U, 19U, 21U}) {
            for (const size_t columns : {11U, 32U, 33U}) {
                const auto desc = std::make_shared<CpuBlockedMemoryDesc>(type, Shape{rows, columns});
                const auto source = std::make_shared<Memory>(engine, desc);
                auto* input = static_cast<uint8_t*>(source->getData());
                const size_t elements = rows * columns;
                for (size_t byte = 0; byte < (elements + 1) / 2; ++byte) {
                    const auto low = (byte * 2 * 7 + 3) % 16;
                    const auto high = ((byte * 2 + 1) * 7 + 3) % 16;
                    input[byte] = static_cast<uint8_t>(low | (high << 4));
                }
                for (const int ranks : {2, 3}) {
                    const size_t partition_rows = rows / ranks;
                    for (int rank = 0; rank < ranks; ++rank) {
                        SCOPED_TRACE(::testing::Message()
                                     << type << '/' << rows << 'x' << columns << '/' << rank << ':' << ranks);
                        const auto partition = split_horizontal(engine, source, 0, rank, ranks);
                        const size_t count =
                            (rank == ranks - 1 ? rows - rank * partition_rows : partition_rows) * columns;
                        ASSERT_EQ(partition->getShape().getElementsCount(), count);
                        const auto* output = static_cast<const uint8_t*>(partition->getData());
                        const size_t offset = rank * partition_rows * columns;
                        for (size_t element = 0; element < count; ++element) {
                            EXPECT_EQ((output[element / 2] >> (4 * (element % 2))) & 15U,
                                      ((offset + element) * 7 + 3) % 16);
                        }
                        if (offset % 2 != 0 && count % 2 != 0) {
                            EXPECT_EQ(output[count / 2] & 0xf0U, 0U);
                        }
                    }
                }
            }
        }
    }
}

}  // namespace ov::intel_cpu::test
