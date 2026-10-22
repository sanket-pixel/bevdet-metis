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

Pipeline::Pipeline(const PipelineConfig& config, bool retain_tensors)
    : retain_tensors_(retain_tensors), image_(context_, config.inference, "image"),
      bev_backbone_(context_, config.inference, "bev_backbone"),
      bev_neck_(context_, config.inference, "bev_neck"),
      detection_head_(context_, config.inference, "detection_head"),
      preprocessing_(image_.input_info()), postprocessing_(config.postprocessing) {
    cv::setNumThreads(config.inference.threads);
}

std::vector<Detection> Pipeline::Process(const Frame& frame, Timings* timings) {
    Timings measurements;
    const auto start = Clock::now();
    prepared_ = preprocessing_.Process(frame, retain_tensors_);
    const auto prepared = Clock::now();
    measurements.insert(preprocessing_.timings().begin(), preprocessing_.timings().end());
    auto Run = [&](const std::string& name, AipuInference& model) {
        const auto begin = Clock::now();
        model.Process();
        measurements["inference_" + name] += Milliseconds(begin, Clock::now());
        for (const auto& [key, duration] : model.timings())
            measurements["inference_" + key] += duration;
    };
    for (int camera = 0; camera < 6; ++camera) {
        std::copy(prepared_.quantized_images[camera].begin(),
                  prepared_.quantized_images[camera].end(), image_.input_buffer().begin());
        if (retain_tensors_)
            image_.ValidateInput(prepared_.images[camera]);
        Run("image", image_);
        native_features_[camera] = image_.output_buffer();
        if (retain_tensors_)
            image_features_[camera] = image_.Decode()[0];
    }
    auto stage_start = Clock::now();
    view_transform_.Process(native_features_, image_.output_info(0), prepared_,
                            bev_backbone_.input_info(), bev_backbone_.input_buffer());
    measurements["inference_view_transform"] = Milliseconds(stage_start, Clock::now());
    if (retain_tensors_) {
        bev_ = view_transform_.Process(image_features_, prepared_);
        bev_backbone_.ValidateInput(bev_);
    }
    Run("bev_backbone", bev_backbone_);
    stage_start = Clock::now();
    feature_resize_.Process(bev_backbone_.output_buffer(), bev_backbone_.output_info(0),
                            bev_neck_.input_info(), bev_neck_.input_buffer(), true);
    measurements["inference_host_resize"] = Milliseconds(stage_start, Clock::now());
    if (retain_tensors_) {
        backbone_features_ = bev_backbone_.Decode()[0];
        neck_input_ = PrepareNeck(backbone_features_, feature_resize_);
        bev_neck_.ValidateInput(neck_input_);
    }
    Run("bev_neck", bev_neck_);
    stage_start = Clock::now();
    feature_resize_.Process(bev_neck_.output_buffer(), bev_neck_.output_info(0),
                            detection_head_.input_info(), detection_head_.input_buffer());
    measurements["inference_host_resize"] += Milliseconds(stage_start, Clock::now());
    if (retain_tensors_) {
        neck_features_ = bev_neck_.Decode()[0];
        head_input_ = feature_resize_.Process(neck_features_, 128, 128);
        detection_head_.ValidateInput(head_input_);
    }
    Run("detection_head", detection_head_);
    stage_start = Clock::now();
    const auto& outputs = detection_head_.Decode();
    std::copy(outputs.begin(), outputs.end(), heads_.begin());
    measurements["inference_dequantization"] += Milliseconds(stage_start, Clock::now());
    const auto inferred = Clock::now();
    auto detections = postprocessing_.Process(heads_, detection_head_.output_info(5),
                                              detection_head_.output_buffer(5));
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
    if (!retain_tensors_)
        throw std::logic_error("Enable tensor retention before saving traces");
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
