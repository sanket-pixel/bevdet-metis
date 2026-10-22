// SPDX-License-Identifier: AGPL-3.0-or-later
#include "common/config.h"

#include <set>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace bev {
namespace {
void CheckKeys(const YAML::Node& node, const std::set<std::string>& expected) {
    if (!node.IsMap())
        throw std::invalid_argument("Expected a YAML mapping");
    std::set<std::string> actual;
    for (const auto& entry : node)
        actual.insert(entry.first.as<std::string>());
    if (actual != expected)
        throw std::invalid_argument("Unexpected or missing YAML settings");
}
} // namespace

PipelineConfig LoadConfig(const std::filesystem::path& path) {
    const auto root = YAML::LoadFile(path.string());
    CheckKeys(root, {"inference", "data", "postprocessing"});
    CheckKeys(root["inference"], {"model_directory", "compiled_model_directory", "threads"});
    CheckKeys(root["data"], {"directory", "scene"});
    CheckKeys(root["postprocessing"], {"score_threshold", "nms_threshold", "max_detections"});
    const auto parent = std::filesystem::absolute(path).parent_path();
    const auto resolve = [&](const YAML::Node& node) {
        return std::filesystem::weakly_canonical(parent / node.as<std::string>());
    };
    PipelineConfig config;
    config.inference.model_directory = resolve(root["inference"]["model_directory"]);
    config.inference.compiled_model_directory =
        resolve(root["inference"]["compiled_model_directory"]);
    config.inference.threads = root["inference"]["threads"].as<int>();
    config.data.directory = resolve(root["data"]["directory"]);
    config.data.scene = root["data"]["scene"].as<std::string>();
    config.postprocessing.score_threshold = root["postprocessing"]["score_threshold"].as<float>();
    config.postprocessing.nms_threshold = root["postprocessing"]["nms_threshold"].as<double>();
    config.postprocessing.max_detections = root["postprocessing"]["max_detections"].as<int>();
    const auto& post = config.postprocessing;
    if (config.inference.threads < 1 || config.inference.threads > 32 ||
        !(post.score_threshold > 0 && post.score_threshold < 1) ||
        !(post.nms_threshold > 0 && post.nms_threshold < 1) || post.max_detections < 1 ||
        post.max_detections > 500 || config.data.scene.empty())
        throw std::invalid_argument("Invalid runtime configuration");
    return config;
}
} // namespace bev
