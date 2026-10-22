// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include "common/types.h"

namespace bev {
class Preprocessing {
public:
    Preprocessing();
    PreparedInput Process(const Frame& frame) const;

private:
    static Tensor ReadCamera(const Camera& camera, cv::Vec3f& adjustment);
};
} // namespace bev
