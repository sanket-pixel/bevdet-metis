// Bilinear conventions follow ONNX Runtime (Copyright Microsoft Corporation,
// MIT license in licenses/onnxruntime_mit.txt).
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "feature_resize.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace bev {
namespace {
struct Sample {
    int first;
    int second;
    float first_distance;
    float second_distance;
};

std::vector<Sample> Samples(int input_size, int output_size) {
    std::vector<Sample> result(output_size);
    for (int i = 0; i < output_size; ++i) {
        const float coordinate =
            output_size == 1 ? 0.0f : float(i) * float(input_size - 1) / float(output_size - 1);
        const int first = std::min(int(coordinate), input_size - 1),
                  second = std::min(first + 1, input_size - 1);
        result[i] = {first, second, first == second ? 0.5f : std::abs(coordinate - float(first)),
                     first == second ? 0.5f : std::abs(coordinate - float(second))};
    }
    return result;
}
} // namespace

Tensor FeatureResize::Process(const Tensor& input, int height, int width) const {
    if (input.shape[0] != 1 || height < 1 || width < 1)
        throw std::invalid_argument("Invalid feature resize dimensions");
    const int channels = input.shape[1], input_height = input.shape[2],
              input_width = input.shape[3];
    const auto rows = Samples(input_height, height), columns = Samples(input_width, width);
    Tensor output({1, channels, height, width});
    cv::parallel_for_(cv::Range(0, channels), [&](const cv::Range& range) {
        for (int c = range.start; c < range.end; ++c) {
            const float* source = input.data.data() + c * input_height * input_width;
            float* target = output.data.data() + c * height * width;
            for (int y = 0; y < height; ++y) {
                const auto& row = rows[y];
                for (int x = 0; x < width; ++x) {
                    const auto& column = columns[x];
                    target[y * width + x] = column.second_distance * row.second_distance *
                                                source[row.first * input_width + column.first] +
                                            column.first_distance * row.second_distance *
                                                source[row.first * input_width + column.second] +
                                            column.second_distance * row.first_distance *
                                                source[row.second * input_width + column.first] +
                                            column.first_distance * row.first_distance *
                                                source[row.second * input_width + column.second];
                }
            }
        }
    });
    return output;
}
} // namespace bev
