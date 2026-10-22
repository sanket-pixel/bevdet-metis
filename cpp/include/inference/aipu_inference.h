// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include "common/config.h"
#include "common/types.h"
#include "inference/aipu_context.h"
#include <axruntime/axruntime.h>

namespace bev {
struct AipuTensorInfo {
    std::string name;
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> padded_shape;
    std::vector<std::pair<std::int64_t, std::int64_t>> padding;
    double scale = 1;
    std::int32_t zero_point = 0;
    std::size_t size_bytes = 0;
};

class AipuInference {
public:
    AipuInference(AipuContext& context, const InferenceConfig& config, const std::string& stage);
    ~AipuInference();
    AipuInference(const AipuInference&) = delete;
    AipuInference& operator=(const AipuInference&) = delete;
    // Backend-owned outputs remain valid until the next Process().
    const std::vector<Tensor>& Process(const Tensor& input);
    const AipuTensorInfo& input_info() const;
    const AipuTensorInfo& output_info(std::size_t index) const;
    const Timings& timings() const;

private:
    void Release();
    void PrepareInput(const Tensor& input);
    void DecodeOutputs();
    AipuContext& context_;
    axrConnection* connection_ = nullptr;
    axrModel* model_ = nullptr;
    axrModelInstance* instance_ = nullptr;
    std::vector<AipuTensorInfo> input_infos_;
    std::vector<AipuTensorInfo> output_infos_;
    std::vector<std::vector<std::int8_t>> input_memory_;
    std::vector<std::vector<std::int8_t>> output_memory_;
    std::vector<axrArgument> input_args_;
    std::vector<axrArgument> output_args_;
    std::vector<Tensor> outputs_;
    Timings timings_;
};
} // namespace bev
