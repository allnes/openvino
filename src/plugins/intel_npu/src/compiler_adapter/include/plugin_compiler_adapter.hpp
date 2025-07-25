// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/icompiler.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "ze_graph_ext_wrappers.hpp"

namespace intel_npu {

class PluginCompilerAdapter final : public ICompilerAdapter {
public:
    PluginCompilerAdapter(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct);

    std::shared_ptr<IGraph> compile(const std::shared_ptr<const ov::Model>& model, const Config& config) const override;

    std::shared_ptr<IGraph> compileWS(const std::shared_ptr<ov::Model>& model, const Config& config) const override;

    std::shared_ptr<IGraph> parse(
        ov::Tensor mainBlob,
        const bool blobAllocatedByPlugin,
        const Config& config,
        std::optional<std::vector<ov::Tensor>> initBlobs = std::nullopt,
        const std::optional<std::shared_ptr<const ov::Model>>& model = std::nullopt) const override;

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) const override;

    std::vector<std::string> get_supported_options() const override;

    bool is_option_supported(std::string optname) const override;

    uint32_t get_version() const override;

private:
    std::shared_ptr<ZeroInitStructsHolder> _zeroInitStruct;

    std::shared_ptr<ZeGraphExtWrappers> _zeGraphExt;
    ov::SoPtr<ICompiler> _compiler;

    Logger _logger;
};

}  // namespace intel_npu
