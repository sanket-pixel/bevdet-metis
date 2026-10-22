// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include "common/types.h"
#include "inference/aipu_inference.h"

namespace bev {
class ViewTransform {
public:
    void Process(const std::array<std::vector<std::int8_t>, 6>& features,
                 const AipuTensorInfo& source, const PreparedInput& input,
                 const AipuTensorInfo& target, std::vector<std::int8_t>& output) const;
    Tensor Process(const std::array<Tensor, 6>& features, const PreparedInput& input) const;
};
} // namespace bev
