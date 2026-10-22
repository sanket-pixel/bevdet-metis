// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include "common/types.h"
#include "inference/aipu_inference.h"

namespace bev {
class FeatureResize {
public:
    void Process(const std::vector<std::int8_t>& input, const AipuTensorInfo& source,
                 const AipuTensorInfo& target, std::vector<std::int8_t>& output,
                 bool packed_backbone = false) const;
    Tensor Process(const Tensor& input, int height, int width) const;
};
} // namespace bev
