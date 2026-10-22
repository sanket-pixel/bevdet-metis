// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include "common/config.h"
#include "common/types.h"

namespace bev {
class Postprocessing {
public:
    explicit Postprocessing(const PostprocessingConfig& config);
    std::vector<Detection> Process(const Heads& heads) const;

private:
    PostprocessingConfig config_;
};
} // namespace bev
