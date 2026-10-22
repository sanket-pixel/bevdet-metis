// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace bev {
inline constexpr std::array<const char*, 6> camera_names{"CAM_FRONT_LEFT",  "CAM_FRONT",
                                                         "CAM_FRONT_RIGHT", "CAM_BACK_LEFT",
                                                         "CAM_BACK",        "CAM_BACK_RIGHT"};
inline constexpr std::array<const char*, 6> head_names{"reg", "height", "dim",
                                                       "rot", "vel",    "heatmap"};
inline constexpr std::array<const char*, 10> class_names{
    "car",        "truck",   "construction_vehicle", "bus",         "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian",           "traffic_cone"};

struct Tensor {
    std::array<int, 4> shape{}; // NCHW, contiguous float32.
    std::vector<float> data;
    Tensor() = default;
    explicit Tensor(std::array<int, 4> shape);
};
using Heads = std::array<Tensor, 6>;

struct Camera {
    std::filesystem::path image_path;
    cv::Matx33f intrinsic;
    cv::Matx44d sensor_to_ego;
    cv::Matx44d ego_to_global;
};

struct Frame {
    std::string token;
    std::string scene;
    std::int64_t timestamp = 0;
    std::array<Camera, 6> cameras;
};

struct PreparedInput {
    std::array<Tensor, 6> images;
    std::array<std::vector<std::int8_t>, 6> quantized_images;
    std::array<cv::Matx44f, 6> camera_to_keyego;
    std::vector<std::int64_t> coors_img;
    std::vector<std::int64_t> coors_depth;
};

struct Detection {
    int class_id = 0;
    double score = 0;
    std::array<double, 3> center{};
    std::array<double, 3> dimensions{};
    double yaw = 0;
    std::array<double, 2> velocity{};
};

using Timings = std::map<std::string, double>;
std::string Sha256(const std::filesystem::path& path);
void SaveTensor(const std::filesystem::path& directory, const std::string& name,
                const Tensor& tensor);
} // namespace bev
