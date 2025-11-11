#pragma once

#include <memory>
#include <string>

namespace ov {
class Model;
}

namespace ov::intel_cpu::android_cache {

std::shared_ptr<ov::Model> prepare_model_for_cache(const std::shared_ptr<ov::Model>& model);

void restore_sinks_from_cache(const std::shared_ptr<ov::Model>& model);

}  // namespace ov::intel_cpu::android_cache
