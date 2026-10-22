// SPDX-License-Identifier: AGPL-3.0-or-later
#include "inference/aipu_inference.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace bev {
namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
double Milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
Json ReadJson(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("Cannot read " + path.string());
    return Json::parse(input);
}
std::runtime_error RuntimeError(AipuContext& context, const std::string& operation) {
    const char* detail = axr_last_error_string(reinterpret_cast<const axrObject*>(context.get()));
    return std::runtime_error(operation + (detail ? ": " + std::string(detail) : ""));
}
AipuTensorInfo TensorInfo(const axrTensorInfo& raw) {
    if ((raw.ndims != 1 && raw.ndims != 4) || raw.bits != 8 || raw.type != AXR_SIGNED ||
        !std::isfinite(raw.scale) || raw.scale <= 0)
        throw std::runtime_error("Expected an int8 runtime tensor with a valid scale");
    AipuTensorInfo info;
    info.name.assign(raw.name, std::find(raw.name, raw.name + sizeof(raw.name), '\0'));
    info.scale = raw.scale;
    info.zero_point = raw.zero_point;
    std::size_t elements = 1;
    for (std::size_t d = 0; d < raw.ndims; ++d) {
        if (raw.dims[d] == 0 || raw.padding[d][0] >= raw.dims[d] ||
            raw.padding[d][1] >= raw.dims[d] - raw.padding[d][0])
            throw std::runtime_error("Invalid runtime tensor padding");
        info.padded_shape.push_back(raw.dims[d]);
        info.shape.push_back(raw.dims[d] - raw.padding[d][0] - raw.padding[d][1]);
        info.padding.emplace_back(raw.padding[d][0], raw.padding[d][1]);
        elements *= raw.dims[d];
    }
    info.size_bytes = axr_tensor_size(&raw);
    if (info.size_bytes != elements)
        throw std::runtime_error("Unexpected runtime tensor byte count");
    return info;
}

void ValidateTensor(const AipuTensorInfo& info, const Json& description, const Json& shape,
                    const Json& padding, const Json& parameters) {
    std::vector<std::int64_t> flattened;
    for (const auto& pair : info.padding) {
        flattened.push_back(pair.first);
        flattened.push_back(pair.second);
    }
    if (shape != info.padded_shape || padding != flattened || parameters.at(0) != info.scale ||
        parameters.at(1) != info.zero_point || description.at("name") != info.name ||
        description.at("shape") != info.padded_shape || description.at("size") != info.size_bytes ||
        description.at("dtype") != "int8")
        throw std::runtime_error("Runtime tensor differs from compiled metadata");
}
} // namespace

AipuInference::AipuInference(AipuContext& context, const InferenceConfig& config,
                             const std::string& stage)
    : context_(context) {
    try {
        const auto directory = config.compiled_model_directory / stage;
        const auto manifest = ReadJson(directory / "manifest.json");
        const auto description = ReadJson(directory / "model.json");
        const auto provenance = ReadJson(directory / "provenance.json");
        if (provenance.at("onnx_sha256") != Sha256(config.model_directory / (stage + ".onnx")))
            throw std::runtime_error("Compiled model differs from source ONNX: " + stage);
        if (manifest.at("input_tensor_layout") != "NHWC" ||
            !manifest.at("preprocess_graph").empty() || !manifest.at("postprocess_graph").empty())
            throw std::runtime_error("Expected NHWC CNN without implicit host graphs");
        model_ = axr_load_model(context_.get(), (directory / "model.json").string().c_str());
        if (!model_)
            throw RuntimeError(context_, "Cannot load " + stage);
        const auto inputs = axr_num_model_inputs(model_), outputs = axr_num_model_outputs(model_);
        const auto bindings = provenance.at("input_bindings").get<std::vector<int>>();
        if (manifest.at("input_shapes_original").size() != 1 || bindings.size() != inputs ||
            bindings.empty() || bindings.front() != 0 ||
            outputs != manifest.at("output_shapes").size() ||
            (stage == "detection_head" ? outputs != 6 : outputs != 1))
            throw std::runtime_error("Unexpected compiled IO counts for " + stage);
        for (std::size_t i = 0; i < inputs; ++i)
            input_infos_.push_back(TensorInfo(axr_get_model_input(model_, i)));
        for (std::size_t i = 0; i < outputs; ++i)
            output_infos_.push_back(TensorInfo(axr_get_model_output(model_, i)));
        if (inputs != 1) {
            const auto& primary = input_infos_[0];
            const auto& replica = input_infos_[1];
            if (!(stage == "bev_backbone" && inputs == 2 && bindings == std::vector<int>{0, 0} &&
                  replica.padded_shape ==
                      std::vector<std::int64_t>{std::int64_t(primary.size_bytes)} &&
                  replica.size_bytes == primary.size_bytes && replica.scale == primary.scale &&
                  replica.zero_point == primary.zero_point))
                throw std::runtime_error("Unrecognized native runtime input replica");
        }
        for (bool input : {true, false}) {
            const auto& infos = input ? input_infos_ : output_infos_;
            const auto& shapes = manifest.at(input ? "input_shapes" : "output_shapes");
            const auto& original =
                manifest.at(input ? "input_shapes_original" : "output_shapes_original");
            const auto& padding = manifest.at(input ? "n_padded_ch_inputs" : "n_padded_ch_outputs");
            const auto& params = manifest.at(input ? "quantize_params" : "dequantize_params");
            const auto& native = description.at(input ? "inputs" : "outputs");
            if (params.size() != infos.size() || native.size() != infos.size())
                throw std::runtime_error("Compiled quantization count differs from runtime");
            for (std::size_t i = 0; i < (input ? 1u : infos.size()); ++i) {
                ValidateTensor(infos[i], native.at(i), shapes.at(i), padding.at(i), params.at(i));
                const auto nchw = original.at(i).get<std::array<int, 4>>();
                if (nchw[0] != 1 || infos[i].padded_shape.front() != 1 ||
                    infos[i].shape != std::vector<std::int64_t>{nchw[0], nchw[2], nchw[3], nchw[1]})
                    throw std::runtime_error("Logical runtime tensor differs from ONNX shape");
                if (!input)
                    outputs_.emplace_back(nchw);
            }
        }
        for (const auto& info : input_infos_) {
            input_memory_.emplace_back(info.size_bytes,
                                       std::int8_t(std::clamp(info.zero_point, -128, 127)));
            input_args_.push_back({input_memory_.back().data(), -1, 0, info.size_bytes});
        }
        for (const auto& info : output_infos_) {
            output_memory_.emplace_back(info.size_bytes);
            output_args_.push_back({output_memory_.back().data(), -1, 0, info.size_bytes});
        }
        connection_ = axr_device_connect(context_.get(), nullptr, 1, nullptr);
        if (!connection_)
            throw RuntimeError(context_, "Cannot reserve one Metis core for " + stage);
        auto* properties = axr_create_properties(
            context_.get(),
            "num_sub_devices=1;aipu_cores=1;double_buffer=0;input_dmabuf=0;output_dmabuf=0");
        if (!properties)
            throw RuntimeError(context_, "Cannot create runtime properties");
        instance_ = axr_load_model_instance(connection_, model_, properties);
        axr_destroy(reinterpret_cast<const axrObject*>(properties));
        if (!instance_)
            throw RuntimeError(context_, "Cannot load model instance " + stage);
    } catch (...) {
        Release();
        throw;
    }
}

AipuInference::~AipuInference() { Release(); }
void AipuInference::Release() {
    if (instance_)
        axr_destroy(reinterpret_cast<const axrObject*>(instance_));
    if (model_)
        axr_destroy(reinterpret_cast<const axrObject*>(model_));
    if (connection_)
        axr_destroy(reinterpret_cast<const axrObject*>(connection_));
    instance_ = nullptr;
    model_ = nullptr;
    connection_ = nullptr;
}

void AipuInference::PrepareInput(const Tensor& input) {
    const auto& info = input_infos_.front();
    const int channels = int(info.shape[3]), height = int(info.shape[1]),
              width = int(info.shape[2]);
    if (input.shape != std::array<int, 4>{1, channels, height, width})
        throw std::invalid_argument("Input tensor shape differs from runtime metadata");
    std::vector<std::int8_t> quantized(input.data.size());
    cv::parallel_for_(cv::Range(0, int(input.data.size())), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            const float affine = input.data[i] / float(info.scale) + float(info.zero_point);
            if (!std::isfinite(affine))
                throw std::invalid_argument("Nonfinite model input");
            quantized[i] = std::int8_t(cvRound(std::clamp(affine, -128.0f, 127.0f)));
        }
    });
    cv::Mat reordered;
    cv::transpose(cv::Mat(channels, height * width, CV_8SC1, quantized.data()), reordered);
    const int padded_width = int(info.padded_shape[2]), padded_channels = int(info.padded_shape[3]);
    const int top = int(info.padding[1].first), left = int(info.padding[2].first),
              first_channel = int(info.padding[3].first);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            const auto offset =
                ((y + top) * padded_width + x + left) * padded_channels + first_channel;
            std::copy_n(reordered.ptr<std::int8_t>(y * width + x), channels,
                        input_memory_[0].data() + offset);
        }
    for (std::size_t i = 1; i < input_memory_.size(); ++i)
        std::copy(input_memory_[0].begin(), input_memory_[0].end(), input_memory_[i].begin());
}

void AipuInference::DecodeOutputs() {
    for (std::size_t output = 0; output < outputs_.size(); ++output) {
        const auto& info = output_infos_[output];
        auto& tensor = outputs_[output];
        const int channels = tensor.shape[1], height = tensor.shape[2], width = tensor.shape[3];
        const int padded_width = int(info.padded_shape[2]),
                  padded_channels = int(info.padded_shape[3]);
        const int top = int(info.padding[1].first), left = int(info.padding[2].first),
                  first_channel = int(info.padding[3].first);
        std::vector<std::int8_t> cropped(tensor.data.size());
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                const auto offset =
                    ((y + top) * padded_width + x + left) * padded_channels + first_channel;
                std::copy_n(output_memory_[output].data() + offset, channels,
                            cropped.data() + (y * width + x) * channels);
            }
        cv::Mat reordered;
        cv::transpose(cv::Mat(height * width, channels, CV_8SC1, cropped.data()), reordered);
        const auto* values = reordered.ptr<std::int8_t>();
        for (std::size_t i = 0; i < tensor.data.size(); ++i) {
            float value = float(values[i]);
            value -= float(info.zero_point);
            tensor.data[i] = value * float(info.scale);
        }
    }
}

const std::vector<Tensor>& AipuInference::Process(const Tensor& input) {
    const auto start = Clock::now();
    PrepareInput(input);
    const auto prepared = Clock::now();
    const auto result = axr_run_model_instance(instance_, input_args_.data(), input_args_.size(),
                                               output_args_.data(), output_args_.size());
    if (result != AXR_SUCCESS)
        throw RuntimeError(context_, "AIPU model execution failed");
    const auto executed = Clock::now();
    DecodeOutputs();
    const auto end = Clock::now();
    timings_ = {{"quantization", Milliseconds(start, prepared)},
                {"runtime_execution", Milliseconds(prepared, executed)},
                {"dequantization", Milliseconds(executed, end)}};
    return outputs_;
}

const AipuTensorInfo& AipuInference::input_info() const { return input_infos_.front(); }
const AipuTensorInfo& AipuInference::output_info(std::size_t index) const {
    return output_infos_.at(index);
}
const Timings& AipuInference::timings() const { return timings_; }
} // namespace bev
