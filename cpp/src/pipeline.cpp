// SPDX-License-Identifier: AGPL-3.0-or-later
#include "pipeline.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace bev {
namespace {
using Clock = std::chrono::steady_clock;
double Milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

Tensor PrepareNeck(const Tensor& packed, const FeatureResize& resize) {
    if (packed.shape != std::array<int, 4>{1, 640, 64, 64})
        throw std::invalid_argument("Invalid packed BEV backbone shape");
    Tensor skip({1, 128, 64, 64}), deep({1, 512, 16, 16});
    std::copy_n(packed.data.data(), skip.data.size(), skip.data.data());
    for (int c = 0; c < 512; ++c)
        for (int y = 0; y < 16; ++y)
            for (int x = 0; x < 16; ++x)
                deep.data[(c * 16 + y) * 16 + x] =
                    packed.data[((c + 128) * 64 + y * 4) * 64 + x * 4];
    const auto resized = resize.Process(deep, 64, 64);
    Tensor output({1, 640, 64, 64});
    std::copy(skip.data.begin(), skip.data.end(), output.data.begin());
    std::copy(resized.data.begin(), resized.data.end(), output.data.begin() + skip.data.size());
    return output;
}

void SaveIndices(const std::filesystem::path& directory, const std::string& name,
                 const std::vector<std::int64_t>& data) {
    std::ofstream binary(directory / (name + ".bin"), std::ios::binary);
    binary.exceptions(std::ios::failbit | std::ios::badbit);
    binary.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(std::int64_t));
    std::ifstream previous(directory / "tensors.json");
    auto metadata = nlohmann::json::parse(previous);
    metadata[name] = {{"shape", {data.size()}}, {"dtype", "int64"}, {"file", name + ".bin"}};
    std::ofstream output(directory / "tensors.json");
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output << metadata.dump(2) << '\n';
}
} // namespace

Pipeline::Pipeline(const PipelineConfig& config)
    : image_(context_, config.inference, "image"),
      bev_backbone_(context_, config.inference, "bev_backbone"),
      bev_neck_(context_, config.inference, "bev_neck"),
      detection_head_(context_, config.inference, "detection_head"),
      postprocessing_(config.postprocessing) {
    cv::setNumThreads(config.inference.threads);
}

std::vector<Detection> Pipeline::Process(const Frame& frame, Timings* timings) {
    Timings measurements;
    const auto start = Clock::now();
    prepared_ = preprocessing_.Process(frame);
    const auto prepared = Clock::now();
    auto Run = [&](const std::string& name, AipuInference& model,
                   const Tensor& input) -> const std::vector<Tensor>& {
        const auto begin = Clock::now();
        const auto& result = model.Process(input);
        measurements["inference_" + name] += Milliseconds(begin, Clock::now());
        for (const auto& [key, duration] : model.timings())
            measurements["inference_" + key] += duration;
        return result;
    };
    for (int camera = 0; camera < 6; ++camera)
        image_features_[camera] = Run("image", image_, prepared_.images[camera])[0];
    auto stage_start = Clock::now();
    bev_ = view_transform_.Process(image_features_, prepared_);
    measurements["inference_view_transform"] = Milliseconds(stage_start, Clock::now());
    backbone_features_ = Run("bev_backbone", bev_backbone_, bev_)[0];
    stage_start = Clock::now();
    neck_input_ = PrepareNeck(backbone_features_, feature_resize_);
    measurements["inference_host_resize"] = Milliseconds(stage_start, Clock::now());
    neck_features_ = Run("bev_neck", bev_neck_, neck_input_)[0];
    stage_start = Clock::now();
    head_input_ = feature_resize_.Process(neck_features_, 128, 128);
    measurements["inference_host_resize"] += Milliseconds(stage_start, Clock::now());
    const auto& outputs = Run("detection_head", detection_head_, head_input_);
    std::copy(outputs.begin(), outputs.end(), heads_.begin());
    const auto inferred = Clock::now();
    auto detections = postprocessing_.Process(heads_);
    const auto end = Clock::now();
    measurements["preprocessing"] = Milliseconds(start, prepared);
    measurements["inference"] = Milliseconds(prepared, inferred);
    measurements["postprocessing"] = Milliseconds(inferred, end);
    measurements["pipeline"] = Milliseconds(start, end);
    if (timings)
        *timings = std::move(measurements);
    return detections;
}

void Pipeline::SaveTensors(const std::filesystem::path& directory) const {
    Tensor transforms({1, 6, 4, 4});
    for (int camera = 0; camera < 6; ++camera)
        std::copy_n(prepared_.camera_to_keyego[camera].val, 16,
                    transforms.data.data() + camera * 16);
    SaveTensor(directory, "camera_to_keyego", transforms);
    for (int camera = 0; camera < 6; ++camera) {
        SaveTensor(directory, "image_" + std::to_string(camera), prepared_.images[camera]);
        SaveTensor(directory, "features_" + std::to_string(camera), image_features_[camera]);
    }
    SaveIndices(directory, "coors_img", prepared_.coors_img);
    SaveIndices(directory, "coors_depth", prepared_.coors_depth);
    SaveTensor(directory, "bev", bev_);
    SaveTensor(directory, "backbone", backbone_features_);
    SaveTensor(directory, "neck_input", neck_input_);
    SaveTensor(directory, "neck_features", neck_features_);
    SaveTensor(directory, "head_input", head_input_);
    for (int head = 0; head < 6; ++head)
        SaveTensor(directory, head_names[head], heads_[head]);
}
} // namespace bev
