// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <filesystem>
#include <string>

namespace bev {
struct InferenceConfig {
    std::filesystem::path model_directory;
    std::filesystem::path compiled_model_directory;
    int threads = 4;
};
struct DataConfig {
    std::filesystem::path directory;
    std::string scene;
};
struct PostprocessingConfig {
    float score_threshold = 0.4f;
    double nms_threshold = 0.2;
    int max_detections = 500;
};
struct PipelineConfig {
    InferenceConfig inference;
    DataConfig data;
    PostprocessingConfig postprocessing;
};
PipelineConfig LoadConfig(const std::filesystem::path& path);
} // namespace bev
