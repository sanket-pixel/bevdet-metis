// SPDX-License-Identifier: AGPL-3.0-or-later
#include "common/dataset.h"
#include "pipeline.h"

#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace {
using Json = nlohmann::json;
volatile std::sig_atomic_t stop_requested = 0;
void RequestStop(int) { stop_requested = 1; }

struct Arguments {
    std::filesystem::path config;
    std::filesystem::path output = "results/cpp/profile.jsonl";
    std::filesystem::path dump_directory;
    int frames = 20;
    int start = 0;
    std::string scene;
    bool profile = false;
};

std::optional<Arguments> ParseArguments(int argc, char** argv) {
    constexpr const char* usage =
        "Usage: bevdet_metis --config <yaml> [--frames 20] [--start 0]\n"
        "  [--output results/cpp/profile.jsonl] [--profile] [--dump-directory <path>]\n"
        "  [--scene scene-name]\n";
    if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
        std::cout << usage;
        return std::nullopt;
    }
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--profile") {
            args.profile = true;
            continue;
        }
        if (++i >= argc)
            throw std::invalid_argument("Missing value for " + option);
        if (option == "--config")
            args.config = argv[i];
        else if (option == "--output")
            args.output = argv[i];
        else if (option == "--dump-directory")
            args.dump_directory = argv[i];
        else if (option == "--frames")
            args.frames = std::stoi(argv[i]);
        else if (option == "--start")
            args.start = std::stoi(argv[i]);
        else if (option == "--scene")
            args.scene = argv[i];
        else
            throw std::invalid_argument("Unknown argument: " + option);
    }
    if (args.config.empty() || args.frames < 1 || args.start < 0 ||
        args.output.extension() != ".jsonl")
        throw std::invalid_argument(usage);
    return args;
}

Json Detections(const std::vector<bev::Detection>& detections) {
    auto result = Json::array();
    for (const auto& box : detections)
        result.push_back({{"class", bev::class_names.at(box.class_id)},
                          {"class_id", box.class_id},
                          {"score", box.score},
                          {"center", box.center},
                          {"dimensions", box.dimensions},
                          {"yaw", box.yaw},
                          {"velocity", box.velocity}});
    return result;
}

void PrintProfile(const bev::Timings& means, std::size_t frames) {
    std::cout << std::fixed << std::setprecision(3) << "Profile: average ms/frame over " << frames
              << " six-camera sample(s), including first frame\n";
    for (const auto& name : {"preprocessing", "inference", "postprocessing", "pipeline"})
        std::cout << "  " << std::setw(17) << std::left << name << means.at(name) << " ms\n";
    std::cout << "  AxRuntime calls:  " << means.at("inference_runtime_execution") << " ms\n"
              << "  Host total:       "
              << means.at("pipeline") - means.at("inference_runtime_execution") << " ms\n";
    std::cout
        << "  Processing FPS: " << 1000.0 / means.at("pipeline")
        << "\n  Includes JPEG reads; excludes model loading, JSON/tensor export and rendering\n";
}

void Run(const Arguments& args) {
    auto config = bev::LoadConfig(args.config);
    if (!args.scene.empty())
        config.data.scene = args.scene;
    const bev::Dataset dataset(config.data.directory);
    const auto frames = dataset.frames(config.data.scene, args.start, args.frames);
    bev::Pipeline pipeline(config, !args.dump_directory.empty());
    if (args.output.has_parent_path())
        std::filesystem::create_directories(args.output.parent_path());
    if (std::filesystem::weakly_canonical(args.output) == std::filesystem::canonical(args.config))
        throw std::invalid_argument("Output must differ from config");
    std::ofstream stream(args.output);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream
        << Json({{"type", "metadata"},
                 {"backend", "cpp"},
                 {"scene", config.data.scene},
                 {"camera_order", bev::camera_names},
                 {"coordinate_frame", "ego at CAM_FRONT_LEFT timestamp; x forward, y left, z up"},
                 {"box_format", "center xyz, dimensions length width height, yaw about z, velocity "
                                "xy; metres and radians"},
                 {"timings",
                  "Includes JPEG reads; excludes model loading, JSON/tensor export and rendering"}})
               .dump()
        << '\n';
    bev::Timings totals;
    std::size_t count = 0;
    for (const auto& frame : frames) {
        if (stop_requested)
            break;
        bev::Timings timings;
        const auto detections = pipeline.Process(frame, &timings);
        stream << Json({{"type", "frame"},
                        {"index", args.start + count},
                        {"token", frame.token},
                        {"timestamp", frame.timestamp},
                        {"boxes", Detections(detections)},
                        {"timings_ms", timings}})
                      .dump()
               << '\n';
        if (!args.dump_directory.empty())
            pipeline.SaveTensors(args.dump_directory / frame.token);
        for (const auto& [key, value] : timings)
            totals[key] += value;
        std::cout << ++count << '/' << frames.size() << ": " << detections.size() << " boxes, "
                  << timings.at("pipeline") << " ms\n";
    }
    stream << Json({{"type", "summary"}, {"frames", count}, {"completed", !stop_requested}}).dump()
           << '\n';
    if (count == 0)
        throw std::runtime_error("No samples processed");
    for (auto& [key, value] : totals)
        value /= count;
    auto summary = args.output;
    summary.replace_extension(".json");
    std::ofstream report(summary);
    report.exceptions(std::ios::failbit | std::ios::badbit);
    report
        << Json({{"frames", count},
                 {"backend", "cpp"},
                 {"mean_timings_ms", totals},
                 {"processing_fps", 1000.0 / totals.at("pipeline")},
                 {"timing",
                  "Includes JPEG reads; excludes model loading, JSON/tensor export and rendering"}})
               .dump(2)
        << '\n';
    if (args.profile)
        PrintProfile(totals, count);
    std::cout << "Saved detections: " << args.output.string() << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto args = ParseArguments(argc, argv);
        if (!args)
            return 0;
        std::signal(SIGINT, RequestStop);
        Run(*args);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
