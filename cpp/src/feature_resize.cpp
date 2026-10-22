// Bilinear conventions follow ONNX Runtime (Copyright Microsoft Corporation,
// MIT license in licenses/onnxruntime_mit.txt).
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "feature_resize.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace bev {
namespace {
void ResizeChannels(const std::int8_t* a, const std::int8_t* b, const std::int8_t* c,
                    const std::int8_t* d, std::int8_t* output, int first, int channels, float wa,
                    float wb, float wc, float wd, float scale, float zero, float target_scale,
                    float target_zero) {
    int channel = first;
#if defined(__AVX2__)
    const auto sv = _mm256_set1_ps(scale), zv = _mm256_set1_ps(zero);
    const auto ts = _mm256_set1_ps(target_scale), tz = _mm256_set1_ps(target_zero);
    const auto aw = _mm256_set1_ps(wa), bw = _mm256_set1_ps(wb);
    const auto cw = _mm256_set1_ps(wc), dw = _mm256_set1_ps(wd);
    auto Load = [&](const std::int8_t* pointer) {
        const auto bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pointer));
        return _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes)), zv),
                             sv);
    };
    for (; channel + 8 <= channels; channel += 8) {
        const auto av = _mm256_mul_ps(aw, Load(a + channel));
        const auto bv = _mm256_mul_ps(bw, Load(b + channel));
        const auto cv = _mm256_mul_ps(cw, Load(c + channel));
        const auto dv = _mm256_mul_ps(dw, Load(d + channel));
        const auto value = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(av, bv), cv), dv);
        const auto affine = _mm256_add_ps(_mm256_div_ps(value, ts), tz);
        const auto clipped =
            _mm256_min_ps(_mm256_max_ps(affine, _mm256_set1_ps(-128)), _mm256_set1_ps(127));
        const auto integers = _mm256_cvtps_epi32(clipped);
        const auto words = _mm_packs_epi32(_mm256_castsi256_si128(integers),
                                           _mm256_extracti128_si256(integers, 1));
        const auto bytes = _mm_packs_epi16(words, words);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(output + channel), bytes);
    }
#endif
    for (; channel < channels; ++channel) {
        const float av = (float(a[channel]) - zero) * scale;
        const float bv = (float(b[channel]) - zero) * scale;
        const float cv = (float(c[channel]) - zero) * scale;
        const float dv = (float(d[channel]) - zero) * scale;
        const float value = wa * av + wb * bv + wc * cv + wd * dv;
        const float affine = value / target_scale + target_zero;
        output[channel] = std::int8_t(std::nearbyint(std::clamp(affine, -128.0f, 127.0f)));
    }
}

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

void FeatureResize::Process(const std::vector<std::int8_t>& input, const AipuTensorInfo& source,
                            const AipuTensorInfo& target, std::vector<std::int8_t>& output,
                            bool packed_backbone) const {
    if (source.shape.size() != 4 || target.shape.size() != 4)
        throw std::invalid_argument("Expected rank-four feature tensors");
    const int channels = int(target.shape[3]), height = int(target.shape[1]),
              width = int(target.shape[2]);
    if (source.shape[0] != 1 || target.shape[0] != 1 || source.shape[3] != channels ||
        input.size() != source.size_bytes || output.size() != target.size_bytes ||
        (packed_backbone && (source.shape != std::vector<std::int64_t>{1, 64, 64, 640} ||
                             target.shape != source.shape)))
        throw std::invalid_argument("Invalid native feature resize shapes");
    const auto rows = Samples(packed_backbone ? 16 : int(source.shape[1]), height);
    const auto columns = Samples(packed_backbone ? 16 : int(source.shape[2]), width);
    const int stride = packed_backbone ? 4 : 1;
    const int source_stride = int(source.padded_shape[3]);
    const auto source_row_stride = source.padded_shape[2] * source_stride;
    const auto* source_start = input.data() + source.padding[1].first * source_row_stride +
                               source.padding[2].first * source_stride + source.padding[3].first;
    const float scale = float(source.scale), zero = float(source.zero_point);
    const float target_scale = float(target.scale), target_zero = float(target.zero_point);
    cv::parallel_for_(cv::Range(0, height), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const auto& row = rows[y];
            for (int x = 0; x < width; ++x) {
                auto* destination = output.data() +
                                    ((y + target.padding[1].first) * target.padded_shape[2] + x +
                                     target.padding[2].first) *
                                        target.padded_shape[3] +
                                    target.padding[3].first;
                int first_channel = 0;
                if (packed_backbone) {
                    const auto* skip = source_start + y * source_row_stride + x * source_stride;
                    for (int c = 0; c < 128; ++c) {
                        const float value = (float(skip[c]) - zero) * scale;
                        const float affine = value / target_scale + target_zero;
                        destination[c] =
                            std::int8_t(std::nearbyint(std::clamp(affine, -128.0f, 127.0f)));
                    }
                    first_channel = 128;
                }
                const auto& column = columns[x];
                const auto* a = source_start + row.first * stride * source_row_stride +
                                column.first * stride * source_stride;
                const auto* b = source_start + row.first * stride * source_row_stride +
                                column.second * stride * source_stride;
                const auto* c = source_start + row.second * stride * source_row_stride +
                                column.first * stride * source_stride;
                const auto* d = source_start + row.second * stride * source_row_stride +
                                column.second * stride * source_stride;
                const float wa = column.second_distance * row.second_distance;
                const float wb = column.first_distance * row.second_distance;
                const float wc = column.second_distance * row.first_distance;
                const float wd = column.first_distance * row.first_distance;
                ResizeChannels(a, b, c, d, destination, first_channel, channels, wa, wb, wc, wd,
                               scale, zero, target_scale, target_zero);
            }
        }
    });
}

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
