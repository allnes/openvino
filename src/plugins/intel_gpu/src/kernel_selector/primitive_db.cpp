// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_db.h"
#include <cassert>
#include <algorithm>
#include <vector>
#include <utility>
#include <stdexcept>

#ifndef NDEBUG
#include <fstream>
#include <iostream>
#endif

namespace kernel_selector {
namespace gpu {
namespace cache {

primitive_db::primitive_db()
    : primitives({
#include "ks_primitive_db.inc"
      }),
      batch_headers({
#include "ks_primitive_db_batch_headers.inc"
      }) {
#ifdef ENABLE_EXPERIMENTAL_PORTABLE_GPU
    // Generic Eltwise kernels only depend on fetch_data and its common preamble.
    std::map<std::string, code> portable_batch_headers;
    portable_batch_headers.emplace("common", std::move(batch_headers.at("common")));
    portable_batch_headers.emplace("fetch_data", std::move(batch_headers.at("fetch_data")));
    batch_headers = std::move(portable_batch_headers);
#endif
}

std::vector<code> primitive_db::get(const primitive_id& id) const {
#ifndef NDEBUG
    {
        std::string filename = id + ".cl";
        std::ifstream kernel_file{filename, std::ios::in | std::ios::binary};
        if (kernel_file.is_open()) {
            code ret;
            auto beg = kernel_file.tellg();
            kernel_file.seekg(0, std::ios::end);
            auto end = kernel_file.tellg();
            kernel_file.seekg(0, std::ios::beg);

            ret.resize((size_t)(end - beg));
            kernel_file.read(&ret[0], (size_t)(end - beg));

            return {std::move(ret)};
        }
    }
#endif
    try {
        const auto codes = primitives.equal_range(id);
        std::vector<code> temp;
        std::for_each(codes.first, codes.second, [&](const std::pair<const std::string, std::string>& c) {
            temp.push_back(c.second);
        });

        if (temp.size() != 1) {
            throw std::runtime_error("cannot find the kernel " + id + " in primitive database.");
        }

        return temp;
    } catch (...) {
        throw std::runtime_error("cannot find the kernel " + id + " in primitive database.");
    }
}
}  // namespace cache
}  // namespace gpu
}  // namespace kernel_selector
