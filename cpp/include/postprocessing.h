// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include "common/config.h"
#include "common/types.h"
#include "inference/aipu_inference.h"

namespace bev {
class Postprocessing {
public:
    explicit Postprocessing(const PostprocessingConfig& config);
    std::vector<Detection> Process(const Heads& heads) const;
    std::vector<Detection> Process(const Heads& heads, const AipuTensorInfo& heatmap_info,
                                   const std::vector<std::int8_t>& heatmap) const;

private:
    std::vector<Detection> Decode(const Heads& heads, const std::vector<float>& scores) const;
    PostprocessingConfig config_;
};
} // namespace bev
