// Decoding/NMS conventions follow FastBEV++ (Apache-2.0); see THIRD_PARTY_NOTICES.md.
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "postprocessing.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace bev {
namespace {
constexpr std::array<double, 10> nms_scale{1.0, 0.7, 0.7, 0.4, 0.55, 1.1, 1.0, 1.0, 1.5, 3.5};
struct Candidate {
    float score;
    int index;
};

// Match NumPy 1.26 introselect's tie order before stable sorting the top 500.
// Adapted from NumPy selection.cpp (BSD license in licenses/numpy_bsd.txt).
int MedianOfFive(Candidate* values) {
    auto Less = [&](int a, int b) { return values[a].score < values[b].score; };
    if (Less(1, 0))
        std::swap(values[1], values[0]);
    if (Less(4, 3))
        std::swap(values[4], values[3]);
    if (Less(3, 0))
        std::swap(values[3], values[0]);
    if (Less(4, 1))
        std::swap(values[4], values[1]);
    if (Less(2, 1))
        std::swap(values[2], values[1]);
    if (Less(3, 2))
        return Less(3, 1) ? 1 : 3;
    return 2;
}

void Select(Candidate* values, int count, int kth) {
    int low = 0, high = count - 1;
    if (kth < 3) {
        for (int i = 0; i <= kth; ++i) {
            int smallest = i;
            for (int j = i + 1; j < count; ++j)
                if (values[j].score < values[smallest].score)
                    smallest = j;
            std::swap(values[i], values[smallest]);
        }
        return;
    }
    if (kth == count - 1) {
        int largest = 0;
        for (int i = 1; i < count; ++i)
            if (!(values[i].score < values[largest].score))
                largest = i;
        std::swap(values[kth], values[largest]);
        return;
    }
    int depth = 0;
    for (int size = count; size > 1; size >>= 1)
        depth += 2;
    while (low + 1 < high) {
        int left = low + 1, right = high;
        if (depth > 0 || right - left < 5) {
            const int middle = low + (high - low) / 2;
            if (values[high].score < values[middle].score)
                std::swap(values[high], values[middle]);
            if (values[high].score < values[low].score)
                std::swap(values[high], values[low]);
            if (values[low].score < values[middle].score)
                std::swap(values[low], values[middle]);
            std::swap(values[middle], values[low + 1]);
        } else {
            const int medians = (right - left) / 5;
            for (int i = 0; i < medians; ++i) {
                const int begin = left + i * 5;
                const int median = MedianOfFive(values + begin);
                std::swap(values[begin + median], values[left + i]);
            }
            if (medians > 2)
                Select(values + left, medians, medians / 2);
            std::swap(values[left + medians / 2], values[low]);
            --left;
            ++right;
        }
        --depth;
        const float pivot = values[low].score;
        for (;;) {
            do {
                ++left;
            } while (values[left].score < pivot);
            do {
                --right;
            } while (pivot < values[right].score);
            if (right < left)
                break;
            std::swap(values[left], values[right]);
        }
        std::swap(values[low], values[right]);
        if (right >= kth)
            high = right - 1;
        if (right <= kth)
            low = left;
    }
    if (high == low + 1 && values[high].score < values[low].score)
        std::swap(values[high], values[low]);
}

std::vector<cv::Point2f> Corners(const Detection& box) {
    const double length = box.dimensions[0] * nms_scale[box.class_id];
    const double width = box.dimensions[1] * nms_scale[box.class_id];
    const double c = std::cos(box.yaw), s = std::sin(box.yaw);
    std::vector<cv::Point2f> result;
    for (const auto& point :
         std::array<cv::Point2d, 4>{{{0.5, 0.5}, {0.5, -0.5}, {-0.5, -0.5}, {-0.5, 0.5}}}) {
        const double x = point.x * length, y = point.y * width;
        result.emplace_back(float(x * c - y * s + box.center[0]),
                            float(x * s + y * c + box.center[1]));
    }
    return result;
}

std::vector<Detection> Suppress(const std::vector<Detection>& boxes,
                                const PostprocessingConfig& config) {
    std::vector<std::vector<cv::Point2f>> polygons;
    std::vector<cv::Rect2f> bounds;
    std::vector<double> areas;
    for (const auto& box : boxes) {
        polygons.push_back(Corners(box));
        const auto& points = polygons.back();
        float min_x = points[0].x, min_y = points[0].y, max_x = min_x, max_y = min_y;
        for (const auto& point : points) {
            min_x = std::min(min_x, point.x);
            min_y = std::min(min_y, point.y);
            max_x = std::max(max_x, point.x);
            max_y = std::max(max_y, point.y);
        }
        bounds.emplace_back(min_x, min_y, max_x - min_x, max_y - min_y);
        areas.push_back((box.dimensions[0] * nms_scale[box.class_id]) *
                        (box.dimensions[1] * nms_scale[box.class_id]));
    }
    std::vector<bool> removed(boxes.size(), false);
    std::vector<Detection> result;
    for (std::size_t i = 0; i < boxes.size() && int(result.size()) < config.max_detections; ++i) {
        if (removed[i])
            continue;
        result.push_back(boxes[i]);
        for (std::size_t j = i + 1; j < boxes.size(); ++j) {
            if (removed[j] || (bounds[i] & bounds[j]).area() <= 0)
                continue;
            std::vector<cv::Point2f> intersection;
            const double area =
                std::max(0.0f, cv::intersectConvexConvex(polygons[i], polygons[j], intersection));
            const double iou = area / std::max(1e-12, areas[i] + areas[j] - area);
            if (iou > config.nms_threshold)
                removed[j] = true;
        }
    }
    return result;
}
} // namespace

Postprocessing::Postprocessing(const PostprocessingConfig& config) : config_(config) {}

std::vector<Detection> Postprocessing::Process(const Heads& heads) const {
    if (heads[5].shape != std::array<int, 4>{1, 10, 128, 128})
        throw std::invalid_argument("Unexpected heatmap shape");
    std::vector<float> scores(10 * 128 * 128);
    for (std::size_t i = 0; i < scores.size(); ++i) {
        const float value = std::clamp(heads[5].data[i], -80.0f, 80.0f);
        scores[i] = 1.0f / (1.0f + std::exp(-value));
    }
    return Decode(heads, scores);
}

std::vector<Detection> Postprocessing::Process(const Heads& heads, const AipuTensorInfo& info,
                                               const std::vector<std::int8_t>& heatmap) const {
    if (info.shape != std::vector<std::int64_t>{1, 128, 128, 10} ||
        heatmap.size() != info.size_bytes)
        throw std::invalid_argument("Unexpected native heatmap shape");
    std::array<float, 256> probabilities;
    for (int i = 0; i < 256; ++i) {
        const float value = (float(i - 128) - float(info.zero_point)) * float(info.scale);
        probabilities[i] = 1.0f / (1.0f + std::exp(-std::clamp(value, -80.0f, 80.0f)));
    }
    std::vector<float> scores(10 * 128 * 128);
    for (int y = 0; y < 128; ++y)
        for (int x = 0; x < 128; ++x) {
            const auto* source =
                heatmap.data() +
                ((y + info.padding[1].first) * info.padded_shape[2] + x + info.padding[2].first) *
                    info.padded_shape[3] +
                info.padding[3].first;
            for (int c = 0; c < 10; ++c)
                scores[(c * 128 + y) * 128 + x] = probabilities[int(source[c]) + 128];
        }
    return Decode(heads, scores);
}

std::vector<Detection> Postprocessing::Decode(const Heads& heads,
                                              const std::vector<float>& scores) const {
    constexpr int pixels = 128 * 128;
    constexpr std::array<int, 6> channels{2, 1, 3, 2, 2, 10};
    for (int i = 0; i < 6; ++i)
        if (heads[i].shape != std::array<int, 4>{1, channels[i], 128, 128})
            throw std::invalid_argument("Unexpected detection head shape");
    std::vector<Candidate> candidates;
    candidates.reserve(scores.size());
    for (int i = 0; i < int(scores.size()); ++i)
        candidates.push_back({scores[i], i});
    Select(candidates.data(), int(candidates.size()), int(candidates.size()) - 500);
    candidates.erase(candidates.begin(), candidates.end() - 500);
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const auto& a, const auto& b) { return a.score > b.score; });
    std::vector<Detection> boxes;
    for (int i = 0; i < 500; ++i) {
        const auto& candidate = candidates[i];
        if (!(candidate.score > config_.score_threshold))
            continue;
        const int position = candidate.index % pixels, x = position % 128, y = position / 128;
        Detection box;
        box.class_id = candidate.index / pixels;
        box.score = candidate.score;
        box.center = {(double(x) + heads[0].data[position]) * 0.8 - 51.2,
                      (double(y) + heads[0].data[pixels + position]) * 0.8 - 51.2,
                      heads[1].data[position]};
        for (int c = 0; c < 3; ++c)
            box.dimensions[c] = std::exp(heads[2].data[c * pixels + position]);
        box.yaw = std::atan2(heads[3].data[position], heads[3].data[pixels + position]);
        box.velocity = {heads[4].data[position], heads[4].data[pixels + position]};
        bool valid = true;
        for (int c = 0; c < 3; ++c)
            valid &= std::isfinite(box.center[c]) && std::isfinite(box.dimensions[c]) &&
                     box.center[c] >= (c == 2 ? -10 : -61.2) &&
                     box.center[c] <= (c == 2 ? 10 : 61.2);
        valid &= std::isfinite(box.yaw) && std::isfinite(box.velocity[0]) &&
                 std::isfinite(box.velocity[1]);
        if (valid)
            boxes.push_back(box);
    }
    return Suppress(boxes, config_);
}
} // namespace bev
