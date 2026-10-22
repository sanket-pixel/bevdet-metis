// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

struct axrContext;

namespace bev {
class AipuContext {
public:
    AipuContext();
    ~AipuContext();
    AipuContext(const AipuContext&) = delete;
    AipuContext& operator=(const AipuContext&) = delete;
    axrContext* get() const;

private:
    axrContext* context_ = nullptr;
};
} // namespace bev
