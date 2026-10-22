// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include "common/types.h"
#include "inference/aipu_inference.h"

namespace bev {
class Preprocessing {
public:
    explicit Preprocessing(const AipuTensorInfo& input_info);
    PreparedInput Process(const Frame& frame, bool retain_images = false);
    const Timings& timings() const;

private:
    void ReadCamera(const Camera& camera, cv::Vec3f& adjustment, Tensor* image,
                    std::vector<std::int8_t>& quantized) const;
    Timings timings_;
    AipuTensorInfo input_info_;
    std::array<std::array<float, 256>, 3> normalization_;
    std::array<std::array<std::int8_t, 256>, 3> quantization_;
};
} // namespace bev
