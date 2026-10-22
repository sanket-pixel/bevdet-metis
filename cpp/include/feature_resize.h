// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include "common/types.h"

namespace bev {
class FeatureResize {
public:
    Tensor Process(const Tensor& input, int height, int width) const;
};
} // namespace bev
