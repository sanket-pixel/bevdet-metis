// Geometry conventions follow FastBEV++ (Apache-2.0); see THIRD_PARTY_NOTICES.md.
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "view_transform.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace bev {
Tensor ViewTransform::Process(const std::array<Tensor, 6>& features,
                              const PreparedInput& input) const {
    constexpr int pixels = 16 * 44, voxels = 128 * 128 * 7;
    if (input.coors_img.size() != voxels || input.coors_depth.size() != voxels)
        throw std::invalid_argument("Invalid BEV projection index count");
    std::vector<float> pixel_features(6 * pixels * 123);
    for (int camera = 0; camera < 6; ++camera) {
        if (features[camera].shape != std::array<int, 4>{1, 123, 16, 44})
            throw std::invalid_argument("Unexpected image feature shape");
        for (int pixel = 0; pixel < pixels; ++pixel)
            for (int channel = 0; channel < 123; ++channel)
                pixel_features[(camera * pixels + pixel) * 123 + channel] =
                    features[camera].data[channel * pixels + pixel];
    }
    std::fill_n(pixel_features.data(), 123, 0.0f);
    std::vector<float> depths(6 * pixels * 59);
    for (int pixel = 0; pixel < 6 * pixels; ++pixel)
        for (int depth = 0; depth < 59; ++depth) {
            const float logit = std::clamp(pixel_features[pixel * 123 + depth], -80.0f, 80.0f);
            depths[pixel * 59 + depth] = 1.0f / (1.0f + std::exp(-logit));
        }
    std::vector<float> volume(voxels * 64);
    for (int voxel = 0; voxel < voxels; ++voxel) {
        const auto pixel = input.coors_img[voxel], depth = input.coors_depth[voxel];
        if (pixel < 0 || pixel >= 6 * pixels || depth < 0 || depth >= 6 * pixels * 59)
            throw std::invalid_argument("BEV projection index is outside the camera feature bank");
        for (int channel = 0; channel < 64; ++channel)
            volume[voxel * 64 + channel] =
                pixel_features[pixel * 123 + 59 + channel] * depths[depth];
    }
    std::vector<float> reduced(128 * 128 * 64, 0);
    for (int cell = 0; cell < 128 * 128; ++cell)
        for (int z = 0; z < 7; ++z)
            for (int channel = 0; channel < 64; ++channel)
                reduced[cell * 64 + channel] += volume[(cell * 7 + z) * 64 + channel];
    cv::Mat channel_first;
    cv::transpose(cv::Mat(128 * 128, 64, CV_32FC1, reduced.data()), channel_first);
    Tensor output({1, 64, 128, 128});
    for (int channel = 0; channel < 64; ++channel) {
        cv::Mat source(128, 128, CV_32FC1, channel_first.ptr<float>(channel));
        cv::Mat target(128, 128, CV_32FC1, output.data.data() + channel * 128 * 128);
        cv::transpose(source, target);
    }
    return output;
}
} // namespace bev
