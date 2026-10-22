// Geometry follows FastBEV++ (Apache-2.0). Bicubic resampling follows Pillow 12.2
// (Copyright Secret Labs AB, Fredrik Lundh, Jeffrey A. Clark and contributors;
// MIT-CMU license in licenses/pillow_mit_cmu.txt).
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "preprocessing.h"

#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <opencv2/imgcodecs.hpp>
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

cv::Mat ResizeBicubic(const cv::Mat& image, int width, int height) {
    const auto horizontal = ResizeCoefficients(image.cols, width);
    const auto vertical = ResizeCoefficients(image.rows, height);
    cv::Mat intermediate(image.rows, width, CV_8UC3);
    cv::Mat output(height, width, CV_8UC3);
    for (int y = 0; y < image.rows; ++y) {
        const auto* source = image.ptr<cv::Vec3b>(y);
        auto* target = intermediate.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            const auto& coefficient = horizontal[x];
            for (int c = 0; c < 3; ++c) {
                int sum = 1 << 21;
                for (std::size_t k = 0; k < coefficient.weights.size(); ++k)
                    sum += source[coefficient.start + k][c] * coefficient.weights[k];
                target[x][c] = static_cast<unsigned char>(std::clamp(sum >> 22, 0, 255));
            }
        }
    }
    for (int y = 0; y < height; ++y) {
        const auto& coefficient = vertical[y];
        auto* target = output.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) {
                int sum = 1 << 21;
                for (std::size_t k = 0; k < coefficient.weights.size(); ++k)
                    sum += intermediate.ptr<cv::Vec3b>(coefficient.start + k)[x][c] *
                           coefficient.weights[k];
                target[x][c] = static_cast<unsigned char>(std::clamp(sum >> 22, 0, 255));
            }
        }
    }
    return output;
}
} // namespace

Preprocessing::Preprocessing() {
    openblas_set_num_threads(
        1); // Three-row projection matrices do not benefit from BLAS worker pools.
}

Tensor Preprocessing::ReadCamera(const Camera& camera, cv::Vec3f& adjustment) {
    const auto original =
        cv::imread(camera.image_path.string(), cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
    if (original.empty())
        throw std::runtime_error("Cannot read " + camera.image_path.string());
    const double resize = 704.0 / original.cols;
    const int width = int(original.cols * resize), height = int(original.rows * resize);
    if (height < 256)
        throw std::runtime_error("Camera image is too short after resizing");
    const int crop_x = std::max(0, width - 704) / 2, crop_y = height - 256;
    const auto resized = ResizeBicubic(original, width, height);
    const auto cropped = resized(cv::Rect(crop_x, crop_y, 704, 256));
    cv::Mat normalized;
    cropped.convertTo(normalized, CV_32FC3);
    const std::array<float, 3> mean{123.675f, 116.28f, 103.53f};
    const std::array<float, 3> stddev{58.395f, 57.12f, 57.375f};
    for (int y = 0; y < 256; ++y) {
        auto* row = normalized.ptr<cv::Vec3f>(y);
        for (int x = 0; x < 704; ++x)
            for (int c = 0; c < 3; ++c)
                row[x][c] = (row[x][c] - mean[c]) / stddev[c];
    }
    Tensor result({1, 3, 256, 704});
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < 256; ++y)
            for (int x = 0; x < 704; ++x)
                result.data[(c * 256 + y) * 704 + x] = normalized.at<cv::Vec3f>(y, x)[c];
    adjustment = {float(resize), float(crop_x), float(crop_y)};
    return result;
}

PreparedInput Preprocessing::Process(const Frame& frame) const {
    PreparedInput result;
    std::array<cv::Vec3f, 6> adjustments;
    cv::parallel_for_(cv::Range(0, 6), [&](const cv::Range& range) {
        for (int camera = range.start; camera < range.end; ++camera)
            result.images[camera] = ReadCamera(frame.cameras[camera], adjustments[camera]);
    });
    // Upstream casts poses/extrinsics to float32 before the double-precision composition.
    const cv::Matx44d key_global = cv::Matx44f(frame.cameras[0].ego_to_global);
    const auto key_from_world = key_global.inv(cv::DECOMP_LU);
    result.coors_img.assign(128 * 128 * 7, 0);
    result.coors_depth.assign(128 * 128 * 7, 0);
    std::vector<bool> assigned(128 * 128 * 7, false);
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
        std::vector<float> projected(3 * voxel_count);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 3, voxel_count, 3, 1.0f,
                    projection.val, 4, voxels.front().val, 3, 0.0f, projected.data(), voxel_count);
        for (int r = 0; r < 3; ++r)
            for (int i = 0; i < voxel_count; ++i)
                projected[r * voxel_count + i] += projection(r, 3);
        const auto adjustment = adjustments[camera];
        for (int x = 0; x < 128; ++x) {
            for (int y = 0; y < 128; ++y) {
                for (int z = 0; z < 7; ++z) {
                    const int voxel = (x * 128 + y) * 7 + z;
                    if (assigned[voxel])
                        continue;
                    const float depth = projected[2 * voxel_count + voxel];
                    if (!(depth >= 1 && depth < 60))
                        continue;
                    const float pixel_x =
                        (projected[voxel] / depth) * adjustment[0] - adjustment[1];
                    const float pixel_y =
                        (projected[voxel_count + voxel] / depth) * adjustment[0] - adjustment[2];
                    const int column = int(pixel_x / 16), row = int(pixel_y / 16);
                    if (column < 0 || column >= 44 || row < 0 || row >= 16)
                        continue;
                    const int index = camera * 16 * 44 + row * 44 + column;
                    result.coors_img[voxel] = index;
                    result.coors_depth[voxel] = index * 59 + int(depth) - 1;
                    assigned[voxel] = true;
                }
            }
        }
    }
    return result;
}
} // namespace bev
