// Geometry conventions follow FastBEV++ (Apache-2.0); see THIRD_PARTY_NOTICES.md.
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "common/dataset.h"

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_map>

namespace bev {
namespace {
using Json = nlohmann::json;
using Table = std::unordered_map<std::string, Json>;

Table ReadTable(const std::filesystem::path& directory, const std::string& name) {
    std::ifstream stream(directory / "v1.0-mini" / (name + ".json"));
    if (!stream)
        throw std::runtime_error("Cannot read nuScenes table " + name);
    Table result;
    for (const auto& row : Json::parse(stream))
        result.emplace(row.at("token").get<std::string>(), row);
    return result;
}

cv::Matx44d Transform(const Json& row) {
    const auto q = row.at("rotation").get<std::array<double, 4>>();
    const auto t = row.at("translation").get<std::array<double, 3>>();
    const auto [w, x, y, z] = q;
    const double norm = w * w + x * x + y * y + z * z;
    cv::Matx44d result = cv::Matx44d::eye();
    result(0, 0) = 1 - 2 * (y * y + z * z) / norm;
    result(0, 1) = 2 * (x * y - z * w) / norm;
    result(0, 2) = 2 * (x * z + y * w) / norm;
    result(1, 0) = 2 * (x * y + z * w) / norm;
    result(1, 1) = 1 - 2 * (x * x + z * z) / norm;
    result(1, 2) = 2 * (y * z - x * w) / norm;
    result(2, 0) = 2 * (x * z - y * w) / norm;
    result(2, 1) = 2 * (y * z + x * w) / norm;
    result(2, 2) = 1 - 2 * (x * x + y * y) / norm;
    for (int i = 0; i < 3; ++i)
        result(i, 3) = t[i];
    return result;
}
} // namespace

Dataset::Dataset(const std::filesystem::path& directory) {
    const auto scenes = ReadTable(directory, "scene");
    const auto samples = ReadTable(directory, "sample");
    const auto data = ReadTable(directory, "sample_data");
    const auto calibrations = ReadTable(directory, "calibrated_sensor");
    const auto sensors = ReadTable(directory, "sensor");
    const auto poses = ReadTable(directory, "ego_pose");
    std::unordered_map<std::string, std::array<Camera, 6>> cameras;
    std::unordered_map<std::string, unsigned> present;
    for (const auto& [token, row] : data) {
        if (!row.at("is_key_frame").get<bool>())
            continue;
        const auto& calibration =
            calibrations.at(row.at("calibrated_sensor_token").get<std::string>());
        const auto channel = sensors.at(calibration.at("sensor_token").get<std::string>())
                                 .at("channel")
                                 .get<std::string>();
        const auto found = std::find(camera_names.begin(), camera_names.end(), channel);
        if (found == camera_names.end())
            continue;
        const auto index = std::distance(camera_names.begin(), found);
        const auto sample = row.at("sample_token").get<std::string>();
        auto& camera = cameras[sample][index];
        camera.image_path = directory / row.at("filename").get<std::string>();
        camera.sensor_to_ego = Transform(calibration);
        camera.ego_to_global = Transform(poses.at(row.at("ego_pose_token").get<std::string>()));
        const auto intrinsic = calibration.at("camera_intrinsic");
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                camera.intrinsic(r, c) = intrinsic.at(r).at(c).get<float>();
        present[sample] |= 1u << index;
    }
    for (const auto& [token, scene] : scenes) {
        const auto name = scene.at("name").get<std::string>();
        auto sample_token = scene.at("first_sample_token").get<std::string>();
        auto& frames = scenes_[name];
        while (!sample_token.empty()) {
            if (present[sample_token] != 63u)
                throw std::runtime_error("Sample is missing camera images");
            const auto& sample = samples.at(sample_token);
            frames.push_back({sample_token, name, sample.at("timestamp").get<std::int64_t>(),
                              cameras.at(sample_token)});
            sample_token = sample.at("next").get<std::string>();
        }
    }
}

std::vector<Frame> Dataset::frames(const std::string& scene, int start, int count) const {
    const auto found = scenes_.find(scene);
    if (found == scenes_.end())
        throw std::invalid_argument("Unknown nuScenes scene: " + scene);
    const auto& frames = found->second;
    if (start < 0 || count < 1 || std::size_t(start) > frames.size() ||
        std::size_t(count) > frames.size() - start)
        throw std::invalid_argument("Requested frames are outside the selected scene");
    return {frames.begin() + start, frames.begin() + start + count};
}
} // namespace bev
