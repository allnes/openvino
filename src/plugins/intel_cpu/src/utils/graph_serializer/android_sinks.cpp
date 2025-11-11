#include "android_sinks.hpp"

#include <pugixml.hpp>

#include <memory>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/op/util/variable_extension.hpp"

namespace ov::intel_cpu::android_cache {
namespace {
void append_cache_debug(const std::string& msg) {
#if defined(__ANDROID__)
    (void)msg;
#else
    std::ofstream log("/data/local/tmp/ov_cache_debug.log", std::ios::app);
    if (log.is_open()) {
        log << msg << std::endl;
    }
#endif
}

#if defined(__ANDROID__)

std::string extract_variable_id(const std::shared_ptr<ov::op::Sink>& sink) {
    if (const auto& assign_v6 = std::dynamic_pointer_cast<ov::op::v6::Assign>(sink)) {
        if (const auto& variable = assign_v6->get_variable()) {
            return variable->get_info().variable_id;
        }
    }
    if (const auto& assign_v3 = std::dynamic_pointer_cast<ov::op::v3::Assign>(sink)) {
        return assign_v3->get_variable_id();
    }
    if (const auto& var_ext = ov::as_type_ptr<ov::op::util::VariableExtension>(sink)) {
        return var_ext->get_variable_id();
    }
    return {};
}

std::string make_keep_result_name(const std::string& sink_name) {
    return sink_name + "__cpu_cache_keep";
}

std::shared_ptr<ov::op::util::Variable> find_variable(const std::shared_ptr<ov::Model>& model,
                                                      const std::string& variable_id,
                                                      const ov::Output<ov::Node>& source) {
    for (const auto& variable : model->get_variables()) {
        if (variable && variable->get_info().variable_id == variable_id) {
            return variable;
        }
    }

    ov::op::util::VariableInfo info;
    info.variable_id = variable_id;
    info.data_shape = source.get_partial_shape();
    info.data_type = source.get_element_type();
    auto variable = std::make_shared<ov::op::util::Variable>(info);
    model->add_variables({variable});
    return variable;
}
#endif
}  // namespace

std::shared_ptr<ov::Model> prepare_model_for_cache(const std::shared_ptr<ov::Model>& model) {
#if !defined(__ANDROID__)
    return model;
#else
    append_cache_debug("[android_cache] prepare start sinks=" + std::to_string(model->get_sinks().size()));
    auto clone = std::const_pointer_cast<ov::Model>(model->clone());

    pugi::xml_document sinks_doc;
    auto sinks_root = sinks_doc.append_child("sinks");

    ov::ResultVector keep_results;
    for (const auto& sink : clone->get_sinks()) {
        auto sink_node = sinks_root.append_child("sink");
        sink_node.append_attribute("name").set_value(sink->get_friendly_name().c_str());

        if (!sink->input_values().empty()) {
            const auto& in = sink->input_value(0);
            sink_node.append_attribute("input_name")
                .set_value(in.get_node_shared_ptr()->get_friendly_name().c_str());
            sink_node.append_attribute("input_port").set_value(static_cast<unsigned>(in.get_index()));
        } else {
            sink_node.append_attribute("input_name").set_value("");
            sink_node.append_attribute("input_port").set_value(0);
        }

        const std::string variable_id = extract_variable_id(sink);
        sink_node.append_attribute("variable_id").set_value(variable_id.c_str());

        const auto version = ov::is_type<ov::op::v6::Assign>(sink) ? 6U : 3U;
        sink_node.append_attribute("version").set_value(static_cast<unsigned>(version));

        const auto keep_name = make_keep_result_name(sink->get_friendly_name());
        sink_node.append_attribute("keep_result_name").set_value(keep_name.c_str());

        if (!sink->input_values().empty()) {
            auto keep_res = std::make_shared<ov::op::v0::Result>(sink->input_value(0));
            keep_res->set_friendly_name(keep_name);
            keep_results.push_back(keep_res);
        }
    }

    if (!keep_results.empty()) {
        clone->add_results(keep_results);
    }

    if (!sinks_root.empty()) {
        std::ostringstream meta_stream;
        sinks_doc.save(meta_stream);
        clone->get_rt_info()["intel_cpu_cache_sinks"] = meta_stream.str();
        append_cache_debug("[android_cache] metadata bytes=" + std::to_string(meta_stream.str().size()));
    }

    return clone;
#endif
}

void restore_sinks_from_cache(const std::shared_ptr<ov::Model>& model) {
#if !defined(__ANDROID__)
    (void)model;
#else
    auto rt_it = model->get_rt_info().find("intel_cpu_cache_sinks");
    if (rt_it == model->get_rt_info().end()) {
        append_cache_debug("[android_cache] metadata missing");
        return;
    }

    const auto metadata = rt_it->second.as<std::string>();
    pugi::xml_document sinks_doc;
    if (!sinks_doc.load_string(metadata.c_str())) {
        model->get_rt_info().erase("intel_cpu_cache_sinks");
        append_cache_debug("[android_cache] metadata parse failed");
        return;
    }

    std::unordered_map<std::string, std::shared_ptr<ov::Node>> nodes_by_name;
    for (const auto& node : model->get_ops()) {
        nodes_by_name.emplace(node->get_friendly_name(), node);
    }

    ov::SinkVector sink_nodes;
    std::vector<std::string> keep_results_to_remove;

    const auto sinks_root = sinks_doc.child("sinks");
    for (auto sink = sinks_root.child("sink"); sink; sink = sink.next_sibling("sink")) {
        const auto sink_name = sink.attribute("name").as_string();
        const auto input_name = sink.attribute("input_name").as_string();
        const auto input_port = static_cast<size_t>(sink.attribute("input_port").as_uint());
        const auto variable_id = sink.attribute("variable_id").as_string();
        const auto version = sink.attribute("version").as_uint(3);
        const auto keep_name = sink.attribute("keep_result_name").as_string();

        auto src_it = nodes_by_name.find(input_name);
        if (src_it == nodes_by_name.end()) {
            continue;
        }
        const auto& source_node = src_it->second;
        if (input_port >= source_node->get_output_size()) {
            continue;
        }

        ov::Output<ov::Node> source_output = source_node->output(input_port);
        auto variable = find_variable(model, variable_id, source_output);

        std::shared_ptr<ov::op::Sink> sink_op;
        if (version == 6) {
            sink_op = std::make_shared<ov::op::v6::Assign>(source_output, variable);
        } else {
            sink_op = std::make_shared<ov::op::v3::Assign>(source_output, variable_id);
        }
        sink_op->set_friendly_name(sink_name);
        sink_nodes.push_back(sink_op);

        if (keep_name && keep_name[0]) {
            keep_results_to_remove.emplace_back(keep_name);
        }
    }

    if (!sink_nodes.empty()) {
        model->add_sinks(sink_nodes);
    }

    if (!keep_results_to_remove.empty()) {
        auto results = model->get_results();
        for (const auto& keep_name : keep_results_to_remove) {
            for (const auto& result : results) {
                if (result->get_friendly_name() == keep_name) {
                    model->remove_result(result);
                    break;
                }
            }
        }
    }

    model->get_rt_info().erase(rt_it);
    append_cache_debug("[android_cache] restored sinks=" + std::to_string(sink_nodes.size()) +
                       " keep_removed=" + std::to_string(keep_results_to_remove.size()));
#endif
}

}  // namespace ov::intel_cpu::android_cache
