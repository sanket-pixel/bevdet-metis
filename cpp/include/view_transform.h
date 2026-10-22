// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include "common/types.h"

namespace bev {
class ViewTransform {
public:
    Tensor Process(const std::array<Tensor, 6>& features, const PreparedInput& input) const;
};
} // namespace bev
