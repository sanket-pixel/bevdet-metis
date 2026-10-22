// Geometry follows FastBEV++ (Apache-2.0). Bicubic resampling follows Pillow 12.2
// (Copyright Secret Labs AB, Fredrik Lundh, Jeffrey A. Clark and contributors;
// MIT-CMU license in licenses/pillow_mit_cmu.txt).
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "preprocessing.h"

#include <algorithm>
#include <cblas.h>
#include <chrono>
#include <cmath>
#include <future>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <stdexcept>

namespace bev {
namespace {
struct Coefficients {
    int start = 0;
    std::vector<int> weights;
};

double Cubic(double x) {
    x = std::abs(x);
    if (x < 1)
        return (1.5 * x - 2.5) * x * x + 1;
    if (x < 2)
        return (((x - 5) * x + 8) * x - 4) * -0.5;
    return 0;
}

std::vector<Coefficients> ResizeCoefficients(int input_size, int output_size) {
    const double scale = double(input_size) / output_size;
    const double filter_scale = std::max(1.0, scale);
    const double support = 2 * filter_scale;
    std::vector<Coefficients> result(output_size);
    for (int output = 0; output < output_size; ++output) {
        const double center = (output + 0.5) * scale;
        const int first = std::max(0, int(center - support + 0.5));
        const int last = std::min(input_size, int(center + support + 0.5));
        std::vector<double> weights(last - first);
        double sum = 0;
        for (int i = first; i < last; ++i) {
            weights[i - first] = Cubic((i - center + 0.5) * (1.0 / filter_scale));
            sum += weights[i - first];
        }
        auto& entry = result[output];
        entry.start = first;
        for (const auto weight : weights) {
            const double normalized = sum != 0 ? weight / sum : weight;
            entry.weights.push_back(int(normalized * (1 << 22) + (normalized < 0 ? -0.5 : 0.5)));
        }
    }
    return result;
}

cv::Mat ResizeBicubic(const cv::Mat& image, int width, int height, int crop_y) {
    const auto horizontal = ResizeCoefficients(image.cols, width);
    const auto vertical = ResizeCoefficients(image.rows, height);
    const int first_row = vertical[crop_y].start;
    const int last_row = vertical.back().start + int(vertical.back().weights.size());
    cv::Mat intermediate(last_row - first_row, width, CV_8UC4);
    cv::Mat output(256, width, CV_8UC4);
#if defined(__AVX2__)
    cv::Mat packed;
    cv::cvtColor(image.rowRange(first_row, last_row), packed, cv::COLOR_BGR2BGRA);
    int taps = 0;
    for (const auto& coefficient : horizontal)
        taps = std::max(taps, int(coefficient.weights.size()));
    std::vector<int> indices(taps * width), weights(taps * width, 0);
    for (int k = 0; k < taps; ++k)
        for (int x = 0; x < width; ++x) {
            const auto& coefficient = horizontal[x];
            indices[k * width + x] = std::min(coefficient.start + k, image.cols - 1);
            if (k < int(coefficient.weights.size()))
                weights[k * width + x] = coefficient.weights[k];
        }
#endif
    for (int y = first_row; y < last_row; ++y) {
        const auto* source = image.ptr<cv::Vec3b>(y);
        auto* target = intermediate.ptr<cv::Vec4b>(y - first_row);
        int x = 0;
#if defined(__AVX2__)
        const auto* packed_row = packed.ptr<int>(y - first_row);
        const auto mask = _mm256_set1_epi32(255), zero = _mm256_setzero_si256();
        for (; x + 8 <= width; x += 8) {
            auto b = _mm256_set1_epi32(1 << 21), g = b, r = b;
            for (int k = 0; k < taps; ++k) {
                const auto index = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(indices.data() + k * width + x));
                const auto weight = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(weights.data() + k * width + x));
                const auto pixel = _mm256_i32gather_epi32(packed_row, index, 4);
                b = _mm256_add_epi32(b, _mm256_mullo_epi32(_mm256_and_si256(pixel, mask), weight));
                g = _mm256_add_epi32(
                    g, _mm256_mullo_epi32(_mm256_and_si256(_mm256_srli_epi32(pixel, 8), mask),
                                          weight));
                r = _mm256_add_epi32(
                    r, _mm256_mullo_epi32(_mm256_and_si256(_mm256_srli_epi32(pixel, 16), mask),
                                          weight));
            }
            b = _mm256_min_epi32(_mm256_max_epi32(_mm256_srai_epi32(b, 22), zero), mask);
            g = _mm256_min_epi32(_mm256_max_epi32(_mm256_srai_epi32(g, 22), zero), mask);
            r = _mm256_min_epi32(_mm256_max_epi32(_mm256_srai_epi32(r, 22), zero), mask);
            const auto pixel = _mm256_or_si256(
                b, _mm256_or_si256(_mm256_slli_epi32(g, 8), _mm256_slli_epi32(r, 16)));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(target + x), pixel);
        }
#endif
        for (; x < width; ++x) {
            const auto& coefficient = horizontal[x];
            int b = 1 << 21, g = 1 << 21, r = 1 << 21;
            for (std::size_t k = 0; k < coefficient.weights.size(); ++k) {
                const auto pixel = source[coefficient.start + k];
                const auto weight = coefficient.weights[k];
                b += pixel[0] * weight;
                g += pixel[1] * weight;
                r += pixel[2] * weight;
            }
            target[x] = {static_cast<unsigned char>(std::clamp(b >> 22, 0, 255)),
                         static_cast<unsigned char>(std::clamp(g >> 22, 0, 255)),
                         static_cast<unsigned char>(std::clamp(r >> 22, 0, 255)), 0};
        }
    }
    std::vector<int> sums(width * 4);
    for (int y = 0; y < 256; ++y) {
        const auto& coefficient = vertical[crop_y + y];
        std::fill(sums.begin(), sums.end(), 1 << 21);
        for (std::size_t k = 0; k < coefficient.weights.size(); ++k) {
            const auto* source =
                intermediate.ptr<unsigned char>(coefficient.start + int(k) - first_row);
            const int weight = coefficient.weights[k];
            for (int i = 0; i < width * 4; ++i)
                sums[i] += source[i] * weight;
        }
        auto* target = output.ptr<unsigned char>(y);
        for (int i = 0; i < width * 4; ++i)
            target[i] = static_cast<unsigned char>(std::clamp(sums[i] >> 22, 0, 255));
    }
    return output;
}
} // namespace

Preprocessing::Preprocessing(const AipuTensorInfo& input_info) : input_info_(input_info) {
    openblas_set_num_threads(1);
    if (input_info_.shape != std::vector<std::int64_t>{1, 256, 704, 3})
        throw std::invalid_argument("Unexpected image model input shape");
    const std::array<float, 3> mean{123.675f, 116.28f, 103.53f};
    const std::array<float, 3> stddev{58.395f, 57.12f, 57.375f};
    for (int c = 0; c < 3; ++c)
        for (int value = 0; value < 256; ++value) {
            const float normalized = (float(value) - mean[c]) / stddev[c];
            normalization_[c][value] = normalized;
            const float affine =
                normalized / float(input_info_.scale) + float(input_info_.zero_point);
            quantization_[c][value] = std::int8_t(cvRound(std::clamp(affine, -128.0f, 127.0f)));
        }
}

void Preprocessing::ReadCamera(const Camera& camera, cv::Vec3f& adjustment, Tensor* image,
                               std::vector<std::int8_t>& quantized) const {
    const auto original =
        cv::imread(camera.image_path.string(), cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
    if (original.empty())
        throw std::runtime_error("Cannot read " + camera.image_path.string());
    const double resize = 704.0 / original.cols;
    const int width = int(original.cols * resize), height = int(original.rows * resize);
    if (height < 256)
        throw std::runtime_error("Camera image is too short after resizing");
    const int crop_x = std::max(0, width - 704) / 2, crop_y = height - 256;
    const auto cropped = ResizeBicubic(original, width, height, crop_y);
    quantized.assign(input_info_.size_bytes,
                     std::int8_t(std::clamp(input_info_.zero_point, -128, 127)));
    const int padded_width = int(input_info_.padded_shape[2]);
    const int padded_channels = int(input_info_.padded_shape[3]);
    const int top = int(input_info_.padding[1].first), left = int(input_info_.padding[2].first);
    const int channel_start = int(input_info_.padding[3].first);
    if (image)
        *image = Tensor({1, 3, 256, 704});
    for (int y = 0; y < 256; ++y) {
        const auto* source = cropped.ptr<cv::Vec4b>(y) + crop_x;
        auto* target =
            quantized.data() + ((y + top) * padded_width + left) * padded_channels + channel_start;
        for (int x = 0; x < 704; ++x)
            for (int c = 0; c < 3; ++c) {
                const auto value = source[x][c];
                target[x * padded_channels + c] = quantization_[c][value];
                if (image)
                    image->data[(c * 256 + y) * 704 + x] = normalization_[c][value];
            }
    }
    adjustment = {float(resize), float(crop_x), float(crop_y)};
}

const Timings& Preprocessing::timings() const { return timings_; }

PreparedInput Preprocessing::Process(const Frame& frame, bool retain_images) {
    const auto start = std::chrono::steady_clock::now();
    PreparedInput result;
    std::array<cv::Vec3f, 6> adjustments;
    auto images = std::async(std::launch::async, [&] {
        cv::parallel_for_(cv::Range(0, 6), [&](const cv::Range& range) {
            for (int camera = range.start; camera < range.end; ++camera)
                ReadCamera(frame.cameras[camera], adjustments[camera],
                           retain_images ? &result.images[camera] : nullptr,
                           result.quantized_images[camera]);
        });
        return std::chrono::steady_clock::now();
    });
    // Upstream casts poses/extrinsics to float32 before the double-precision composition.
    const cv::Matx44d key_global = cv::Matx44f(frame.cameras[0].ego_to_global);
    const auto key_from_world = key_global.inv(cv::DECOMP_LU);
    result.coors_img.assign(128 * 128 * 7, 0);
    result.coors_depth.assign(128 * 128 * 7, 0);
    std::array<std::vector<float>, 6> projections;
    constexpr int voxel_count = 128 * 128 * 7;
    std::vector<cv::Vec3f> voxels(voxel_count);
    for (int x = 0; x < 128; ++x)
        for (int y = 0; y < 128; ++y)
            for (int z = 0; z < 7; ++z)
                voxels[(x * 128 + y) * 7 + z] = {float(x) * 0.8f + (-51.2f - 0.8f / 2),
                                                 float(y) * 0.8f + (-51.2f - 0.8f / 2),
                                                 float(z) + (-2.5f - 1.0f / 2)};
    for (int camera = 0; camera < 6; ++camera) {
        const auto& source = frame.cameras[camera];
        const cv::Matx44d global = cv::Matx44f(source.ego_to_global);
        const cv::Matx44d sensor = cv::Matx44f(source.sensor_to_ego);
        result.camera_to_keyego[camera] = key_from_world * global * sensor;
        const cv::Matx44d transform = result.camera_to_keyego[camera];
        const cv::Matx44f inverse = transform.inv(cv::DECOMP_LU);
        cv::Matx<float, 3, 4> projection;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 4, 3, 1.0f, source.intrinsic.val,
                    3, inverse.val, 4, 0.0f, projection.val, 4);
        auto& projected = projections[camera];
        projected.resize(3 * voxel_count);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 3, voxel_count, 3, 1.0f,
                    projection.val, 4, voxels.front().val, 3, 0.0f, projected.data(), voxel_count);
        for (int r = 0; r < 3; ++r)
            for (int i = 0; i < voxel_count; ++i)
                projected[r * voxel_count + i] += projection(r, 3);
    }
    const auto images_ready = images.get();
    const auto indices_start = std::chrono::steady_clock::now();
    cv::parallel_for_(cv::Range(0, voxel_count), [&](const cv::Range& range) {
        for (int voxel = range.start; voxel < range.end; ++voxel) {
            for (int camera = 0; camera < 6; ++camera) {
                const auto& projected = projections[camera];
                const auto adjustment = adjustments[camera];
                const float depth = projected[2 * voxel_count + voxel];
                if (!(depth >= 1 && depth < 60))
                    continue;
                const float pixel_x = (projected[voxel] / depth) * adjustment[0] - adjustment[1];
                const float pixel_y =
                    (projected[voxel_count + voxel] / depth) * adjustment[0] - adjustment[2];
                const int column = int(pixel_x / 16), row = int(pixel_y / 16);
                if (column < 0 || column >= 44 || row < 0 || row >= 16)
                    continue;
                const int index = camera * 16 * 44 + row * 44 + column;
                result.coors_img[voxel] = index;
                result.coors_depth[voxel] = index * 59 + int(depth) - 1;
                break;
            }
        }
    });
    const auto end = std::chrono::steady_clock::now();
    timings_ = {{"preprocessing_images",
                 std::chrono::duration<double, std::milli>(images_ready - start).count()},
                {"preprocessing_indices",
                 std::chrono::duration<double, std::milli>(end - indices_start).count()}};
    return result;
}
} // namespace bev
