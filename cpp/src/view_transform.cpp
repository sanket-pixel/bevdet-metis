// Geometry conventions follow FastBEV++ (Apache-2.0); see THIRD_PARTY_NOTICES.md.
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "view_transform.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace bev {
void ViewTransform::Process(const std::array<std::vector<std::int8_t>, 6>& features,
                            const AipuTensorInfo& source, const PreparedInput& input,
                            const AipuTensorInfo& target, std::vector<std::int8_t>& output) const {
    constexpr int pixels = 16 * 44, cells = 128 * 128;
    if (source.shape != std::vector<std::int64_t>{1, 16, 44, 123} ||
        target.shape != std::vector<std::int64_t>{1, 128, 128, 64} ||
        input.coors_img.size() != cells * 7 || input.coors_depth.size() != cells * 7 ||
        output.size() != target.size_bytes)
        throw std::invalid_argument("Invalid native BEV transform shapes");
    std::array<float, 256> values, probabilities;
    for (int i = 0; i < 256; ++i) {
        values[i] = (float(i - 128) - float(source.zero_point)) * float(source.scale);
        probabilities[i] = 1.0f / (1.0f + std::exp(-std::clamp(values[i], -80.0f, 80.0f)));
    }
    const int source_channels = int(source.padded_shape[3]);
    std::vector<float> features_by_pixel(6 * pixels * 64);
    std::vector<float> depths(6 * pixels * 59);
    for (int camera = 0; camera < 6; ++camera) {
        if (features[camera].size() != source.size_bytes)
            throw std::invalid_argument("Invalid native image feature size");
        for (int y = 0; y < 16; ++y)
            for (int x = 0; x < 44; ++x) {
                const int pixel = camera * pixels + y * 44 + x;
                const auto* data = features[camera].data() +
                                   ((y + source.padding[1].first) * source.padded_shape[2] + x +
                                    source.padding[2].first) *
                                       source_channels +
                                   source.padding[3].first;
                for (int c = 0; c < 64; ++c)
                    features_by_pixel[pixel * 64 + c] =
                        pixel == 0 ? 0.0f : values[int(data[59 + c]) + 128];
                for (int d = 0; d < 59; ++d)
                    depths[pixel * 59 + d] = pixel == 0 ? 0.5f : probabilities[int(data[d]) + 128];
            }
    }
    cv::parallel_for_(cv::Range(0, 128), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y)
            for (int x = 0; x < 128; ++x) {
                std::array<float, 64> sums{};
                for (int z = 0; z < 7; ++z) {
                    const int voxel = (x * 128 + y) * 7 + z;
                    const auto pixel = input.coors_img[voxel], depth = input.coors_depth[voxel];
                    if (pixel < 0 || pixel >= 6 * pixels || depth < 0 || depth >= 6 * pixels * 59)
                        throw std::invalid_argument("BEV projection index out of range");
                    const float probability = depths[depth];
                    const auto* feature = features_by_pixel.data() + pixel * 64;
                    for (int c = 0; c < 64; ++c)
                        sums[c] += feature[c] * probability;
                }
                auto* destination = output.data() +
                                    ((y + target.padding[1].first) * target.padded_shape[2] + x +
                                     target.padding[2].first) *
                                        target.padded_shape[3] +
                                    target.padding[3].first;
                for (int c = 0; c < 64; ++c) {
                    const float affine = sums[c] / float(target.scale) + float(target.zero_point);
                    destination[c] =
                        std::int8_t(std::nearbyint(std::clamp(affine, -128.0f, 127.0f)));
                }
            }
    });
}

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
