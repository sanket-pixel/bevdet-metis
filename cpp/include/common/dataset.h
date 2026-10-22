// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include "common/types.h"

namespace bev {
class Dataset {
public:
    explicit Dataset(const std::filesystem::path& directory);
    std::vector<Frame> frames(const std::string& scene, int start, int count) const;

private:
    std::map<std::string, std::vector<Frame>> scenes_;
};
} // namespace bev
