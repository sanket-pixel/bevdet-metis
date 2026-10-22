// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include "common/config.h"
#include "common/types.h"
#include "feature_resize.h"
#include "inference/aipu_inference.h"
#include "postprocessing.h"
#include "preprocessing.h"
#include "view_transform.h"

namespace bev {
class Pipeline {
public:
    explicit Pipeline(const PipelineConfig& config);
    std::vector<Detection> Process(const Frame& frame, Timings* timings = nullptr);
    void SaveTensors(const std::filesystem::path& directory) const;

private:
    AipuContext context_;
    AipuInference image_;
    AipuInference bev_backbone_;
    AipuInference bev_neck_;
    AipuInference detection_head_;
    Preprocessing preprocessing_;
    ViewTransform view_transform_;
    FeatureResize feature_resize_;
    Postprocessing postprocessing_;
    PreparedInput prepared_;
    std::array<Tensor, 6> image_features_;
    Tensor bev_;
    Tensor backbone_features_;
    Tensor neck_input_;
    Tensor neck_features_;
    Tensor head_input_;
    Heads heads_;
};
} // namespace bev
