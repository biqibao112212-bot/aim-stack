#include "yolo11_buff.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace
{
using Detector = auto_buff::YOLO11_BUFF;
using Object = Detector::Object;
using Profile = Detector::FrameProfile;

std::string argValue(int argc, char** argv, const std::string& key, std::string fallback)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == key) {
            return argv[i + 1];
        }
    }
    return fallback;
}

int intArgValue(int argc, char** argv, const std::string& key, int fallback)
{
    const std::string value = argValue(argc, argv, key, {});
    if (value.empty()) {
        return fallback;
    }
    return std::atoi(value.c_str());
}

Detector::OutputHostMemoryMode outputHostMode(const std::string& value)
{
    if (value == "pageable") {
        return Detector::OutputHostMemoryMode::PageableOnly;
    }
    if (value == "inject-failure") {
        return Detector::OutputHostMemoryMode::InjectPinnedAllocationFailure;
    }
    return Detector::OutputHostMemoryMode::PinnedPreferred;
}

const char* outputHostModeName(Detector::OutputHostMemoryMode mode)
{
    switch (mode) {
        case Detector::OutputHostMemoryMode::PinnedPreferred: return "pinned-preferred";
        case Detector::OutputHostMemoryMode::PageableOnly: return "pageable";
        case Detector::OutputHostMemoryMode::InjectPinnedAllocationFailure:
            return "inject-failure";
    }
    return "unknown";
}

std::vector<Object> detect(Detector& detector, const cv::Mat& source, const std::string& mode)
{
    cv::Mat image = source;
    if (mode == "two") {
        return detector.get_twocandidatebox(image, false);
    }
    return detector.get_onecandidatebox(image, false);
}

bool sameFloatBits(float lhs, float rhs)
{
    std::uint32_t lhs_bits = 0;
    std::uint32_t rhs_bits = 0;
    static_assert(sizeof(lhs_bits) == sizeof(lhs), "unexpected float width");
    std::memcpy(&lhs_bits, &lhs, sizeof(lhs));
    std::memcpy(&rhs_bits, &rhs, sizeof(rhs));
    return lhs_bits == rhs_bits;
}

bool strictObjectsEqual(
    const std::vector<Object>& lhs,
    const std::vector<Object>& rhs,
    std::string* mismatch)
{
    if (lhs.size() != rhs.size()) {
        if (mismatch) {
            *mismatch = "detection count " + std::to_string(lhs.size()) + " != " +
                std::to_string(rhs.size());
        }
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        const auto& a = lhs[i];
        const auto& b = rhs[i];
        if (a.label != b.label) {
            if (mismatch) *mismatch = "label mismatch at detection " + std::to_string(i);
            return false;
        }
        if (!sameFloatBits(a.prob, b.prob)) {
            if (mismatch) *mismatch = "probability mismatch at detection " + std::to_string(i);
            return false;
        }
        if (
            !sameFloatBits(a.rect.x, b.rect.x) ||
            !sameFloatBits(a.rect.y, b.rect.y) ||
            !sameFloatBits(a.rect.width, b.rect.width) ||
            !sameFloatBits(a.rect.height, b.rect.height)) {
            if (mismatch) *mismatch = "rectangle mismatch at detection " + std::to_string(i);
            return false;
        }
        if (a.kpt.size() != b.kpt.size()) {
            if (mismatch) *mismatch = "keypoint count mismatch at detection " + std::to_string(i);
            return false;
        }
        for (std::size_t k = 0; k < a.kpt.size(); ++k) {
            if (
                !sameFloatBits(a.kpt[k].x, b.kpt[k].x) ||
                !sameFloatBits(a.kpt[k].y, b.kpt[k].y)) {
                if (mismatch) {
                    *mismatch = "keypoint mismatch at detection " + std::to_string(i) +
                        ", point " + std::to_string(k);
                }
                return false;
            }
        }
    }
    return true;
}

struct Distribution
{
    double mean = std::numeric_limits<double>::quiet_NaN();
    double p50 = std::numeric_limits<double>::quiet_NaN();
    double p95 = std::numeric_limits<double>::quiet_NaN();
    double p99 = std::numeric_limits<double>::quiet_NaN();
    double max = std::numeric_limits<double>::quiet_NaN();
};

Distribution summarize(std::vector<double> values)
{
    Distribution result;
    if (values.empty()) {
        return result;
    }
    std::sort(values.begin(), values.end());
    double sum = 0.0;
    for (double value : values) sum += value;
    const auto percentile = [&values](double fraction) {
        const double position = fraction * static_cast<double>(values.size() - 1);
        const std::size_t index = static_cast<std::size_t>(std::ceil(position));
        return values[std::min(index, values.size() - 1)];
    };
    result.mean = sum / static_cast<double>(values.size());
    result.p50 = percentile(0.50);
    result.p95 = percentile(0.95);
    result.p99 = percentile(0.99);
    result.max = values.back();
    return result;
}

template <typename Getter>
Distribution summarizeProfiles(const std::vector<Profile>& profiles, Getter getter)
{
    std::vector<double> values;
    values.reserve(profiles.size());
    for (const auto& profile : profiles) values.push_back(getter(profile));
    return summarize(std::move(values));
}

void printDistribution(const char* name, const Distribution& value)
{
    std::cout << "  " << name
              << " mean=" << value.mean
              << " p50=" << value.p50
              << " p95=" << value.p95
              << " p99=" << value.p99
              << " max=" << value.max << " ms\n";
}

struct PhaseResult
{
    std::string name;
    bool ok = false;
    bool pinned = false;
    bool fallback = false;
    std::size_t transfer_bytes = 0;
    double wall_seconds = 0.0;
    std::vector<Object> objects;
    std::vector<Profile> profiles;
    std::string error;
};

PhaseResult runPhase(
    const std::string& name,
    const std::string& config_path,
    const cv::Mat& image,
    const std::string& candidate_mode,
    Detector::OutputHostMemoryMode host_mode,
    int warmup,
    int iterations)
{
    PhaseResult result;
    result.name = name;
    Detector::Options options;
    options.output_host_memory_mode = host_mode;
    options.enable_profiling = true;
    Detector detector(config_path, options);
    result.pinned = detector.output_host_memory_pinned();
    result.fallback = detector.output_host_memory_fallback_used();
    result.transfer_bytes = detector.output_transfer_bytes();

    for (int i = 0; i < warmup; ++i) {
        (void)detect(detector, image, candidate_mode);
    }

    result.profiles.reserve(static_cast<std::size_t>(iterations));
    const auto wall_start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        result.objects = detect(detector, image, candidate_mode);
        const Profile profile = detector.last_frame_profile();
        if (!profile.enabled || !profile.completed) {
            result.error = "incomplete frame profile at iteration " + std::to_string(i);
            return result;
        }
        if (!profile.gpu_events_valid) {
            result.error = "invalid CUDA event profile at iteration " + std::to_string(i);
            return result;
        }
        result.profiles.push_back(profile);
    }
    result.wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_start).count();
    result.ok = true;
    return result;
}

void printPhase(const PhaseResult& phase)
{
    const double qps = phase.wall_seconds > 0.0
        ? static_cast<double>(phase.profiles.size()) / phase.wall_seconds
        : 0.0;
    std::cout << std::fixed << std::setprecision(6)
              << "buff_output_host_phase name=" << phase.name
              << " pinned=" << phase.pinned
              << " fallback=" << phase.fallback
              << " transfer_bytes=" << phase.transfer_bytes
              << " frames=" << phase.profiles.size()
              << " wall_s=" << phase.wall_seconds
              << " qps=" << qps << '\n';
    printDistribution(
        "total", summarizeProfiles(phase.profiles, [](const Profile& p) { return p.total_ms; }));
    printDistribution(
        "raw_h2d_gpu",
        summarizeProfiles(phase.profiles, [](const Profile& p) { return p.raw_h2d_gpu_ms; }));
    printDistribution(
        "preprocess_kernel_gpu",
        summarizeProfiles(
            phase.profiles, [](const Profile& p) { return p.preprocess_kernel_gpu_ms; }));
    printDistribution(
        "h2d_api_cpu",
        summarizeProfiles(phase.profiles, [](const Profile& p) { return p.h2d_api_ms; }));
    printDistribution(
        "preprocess_launch_api_cpu",
        summarizeProfiles(
            phase.profiles, [](const Profile& p) { return p.preprocess_launch_api_ms; }));
    printDistribution(
        "trt_gpu", summarizeProfiles(phase.profiles, [](const Profile& p) { return p.trt_gpu_ms; }));
    printDistribution(
        "enqueue_api_cpu",
        summarizeProfiles(phase.profiles, [](const Profile& p) { return p.enqueue_api_ms; }));
    printDistribution(
        "d2h_gpu", summarizeProfiles(phase.profiles, [](const Profile& p) { return p.d2h_gpu_ms; }));
    printDistribution(
        "d2h_api_cpu",
        summarizeProfiles(phase.profiles, [](const Profile& p) { return p.d2h_api_ms; }));
    printDistribution(
        "sync_wait_cpu",
        summarizeProfiles(phase.profiles, [](const Profile& p) { return p.sync_wait_ms; }));
    printDistribution(
        "d2h_api_plus_sync_cpu",
        summarizeProfiles(
            phase.profiles, [](const Profile& p) { return p.d2h_api_ms + p.sync_wait_ms; }));
    printDistribution(
        "output_convert_cpu",
        summarizeProfiles(phase.profiles, [](const Profile& p) { return p.output_convert_ms; }));
    printDistribution(
        "decode_cpu", summarizeProfiles(phase.profiles, [](const Profile& p) { return p.decode_ms; }));
    printDistribution(
        "nms_cpu", summarizeProfiles(phase.profiles, [](const Profile& p) { return p.nms_ms; }));
    printDistribution(
        "result_build_cpu",
        summarizeProfiles(phase.profiles, [](const Profile& p) { return p.result_build_ms; }));
    printDistribution(
        "restore_cpu", summarizeProfiles(phase.profiles, [](const Profile& p) { return p.restore_ms; }));
    printDistribution(
        "filter_cpu", summarizeProfiles(phase.profiles, [](const Profile& p) { return p.filter_ms; }));
}

int runOutputHostComparison(
    const std::string& config_path,
    const cv::Mat& image,
    const std::string& candidate_mode,
    int warmup,
    int iterations)
{
    const PhaseResult pageable = runPhase(
        "pageable-before",
        config_path,
        image,
        candidate_mode,
        Detector::OutputHostMemoryMode::PageableOnly,
        warmup,
        iterations);
    if (!pageable.ok) {
        std::cerr << "Pageable phase failed: " << pageable.error << '\n';
        return 20;
    }
    const PhaseResult pinned = runPhase(
        "pinned-after",
        config_path,
        image,
        candidate_mode,
        Detector::OutputHostMemoryMode::PinnedPreferred,
        warmup,
        iterations);
    if (!pinned.ok) {
        std::cerr << "Pinned phase failed: " << pinned.error << '\n';
        return 21;
    }
    const PhaseResult injected = runPhase(
        "injected-fallback",
        config_path,
        image,
        candidate_mode,
        Detector::OutputHostMemoryMode::InjectPinnedAllocationFailure,
        warmup,
        std::max(1, std::min(iterations, 20)));
    if (!injected.ok) {
        std::cerr << "Injected fallback phase failed: " << injected.error << '\n';
        return 22;
    }

    if (pageable.pinned || pageable.fallback) {
        std::cerr << "Pageable control did not use the exact pageable-only path\n";
        return 23;
    }
    if (!pinned.pinned || pinned.fallback) {
        std::cerr << "Pinned-preferred phase did not acquire pinned host memory\n";
        return 24;
    }
    if (injected.pinned || !injected.fallback) {
        std::cerr << "Injected pinned-allocation failure did not take pageable fallback\n";
        return 25;
    }
    if (
        pageable.transfer_bytes != pinned.transfer_bytes ||
        pageable.transfer_bytes != injected.transfer_bytes) {
        std::cerr << "Output transfer byte count changed across allocation modes\n";
        return 26;
    }

    std::string mismatch;
    if (!strictObjectsEqual(pageable.objects, pinned.objects, &mismatch)) {
        std::cerr << "Pinned output strict detection/keypoint mismatch: " << mismatch << '\n';
        return 27;
    }
    if (!strictObjectsEqual(pageable.objects, injected.objects, &mismatch)) {
        std::cerr << "Fallback strict detection/keypoint mismatch: " << mismatch << '\n';
        return 28;
    }

    printPhase(pageable);
    printPhase(pinned);
    printPhase(injected);
    std::cout << "buff_output_host_compare strict_detection_keypoint_equal=1"
              << " transfer_bytes=" << pageable.transfer_bytes << '\n';
    return 0;
}

void printDetections(const std::vector<Object>& detections)
{
    for (std::size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        std::cout << "  det[" << i << "] label=" << det.label
                  << " prob=" << det.prob
                  << " rect=(" << det.rect.x << "," << det.rect.y
                  << "," << det.rect.width << "," << det.rect.height
                  << ") kpts=" << det.kpt.size() << " [";
        for (std::size_t k = 0; k < det.kpt.size(); ++k) {
            if (k > 0) std::cout << ',';
            std::cout << '(' << det.kpt[k].x << ',' << det.kpt[k].y << ')';
        }
        std::cout << "]\n";
    }
}
}

int main(int argc, char** argv)
{
    const std::string config_path =
        argValue(argc, argv, "--config", "config/buff_config.sim.yaml");
    const int expected_classes = intArgValue(argc, argv, "--expect-classes", 0);
    const std::string image_path = argValue(argc, argv, "--image", {});
    const std::string candidate_mode = argValue(argc, argv, "--candidate-mode", "one");
    const int expected_min_detections =
        intArgValue(argc, argv, "--expect-min-detections", 0);
    const int expected_max_detections =
        intArgValue(argc, argv, "--expect-max-detections", -1);
    const int expected_label = intArgValue(argc, argv, "--expect-label", -1);
    const int compare_output_host =
        intArgValue(argc, argv, "--compare-output-host-memory", 0);
    const int warmup = std::max(0, intArgValue(argc, argv, "--warmup", 20));
    const int iterations = std::max(1, intArgValue(argc, argv, "--iterations", 200));
    const auto host_mode = outputHostMode(
        argValue(argc, argv, "--output-host-memory", "pinned"));
    const int expected_pinned = intArgValue(argc, argv, "--expect-output-pinned", -1);
    const int expected_fallback = intArgValue(argc, argv, "--expect-output-fallback", -1);

    cv::Mat image;
    if (!image_path.empty()) {
        image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << image_path << std::endl;
            return 3;
        }
    }

    if (compare_output_host != 0) {
        if (image.empty()) {
            std::cerr << "--compare-output-host-memory requires --image\n";
            return 19;
        }
        return runOutputHostComparison(
            config_path, image, candidate_mode, warmup, iterations);
    }

    Detector::Options options;
    options.output_host_memory_mode = host_mode;
    options.enable_profiling = intArgValue(argc, argv, "--profile", 0) != 0;
    Detector detector(config_path, options);
    const int classes = detector.num_classes();
    std::cout << "buff_engine_smoke config=" << config_path
              << " num_classes=" << classes
              << " output_host_mode=" << outputHostModeName(host_mode)
              << " output_pinned=" << detector.output_host_memory_pinned()
              << " output_fallback=" << detector.output_host_memory_fallback_used()
              << " output_bytes=" << detector.output_transfer_bytes() << std::endl;

    if (expected_classes > 0 && classes != expected_classes) {
        std::cerr << "Expected " << expected_classes << ", got " << classes << std::endl;
        return 2;
    }
    if (
        expected_pinned >= 0 &&
        detector.output_host_memory_pinned() != (expected_pinned != 0)) {
        std::cerr << "Unexpected output pinned state\n";
        return 7;
    }
    if (
        expected_fallback >= 0 &&
        detector.output_host_memory_fallback_used() != (expected_fallback != 0)) {
        std::cerr << "Unexpected output fallback state\n";
        return 8;
    }

    if (!image.empty()) {
        const std::vector<Object> detections = detect(detector, image, candidate_mode);
        std::cout << "buff_engine_smoke image=" << image_path
                  << " candidate_mode=" << candidate_mode
                  << " detections=" << detections.size() << std::endl;
        printDetections(detections);

        bool saw_expected_label = expected_label < 0;
        for (const auto& detection : detections) {
            if (detection.label == expected_label) saw_expected_label = true;
        }
        if (detections.size() < static_cast<std::size_t>(expected_min_detections)) {
            std::cerr << "Expected at least " << expected_min_detections
                      << " detections, got " << detections.size() << std::endl;
            return 4;
        }
        if (
            expected_max_detections >= 0 &&
            detections.size() > static_cast<std::size_t>(expected_max_detections)) {
            std::cerr << "Expected at most " << expected_max_detections
                      << " detections, got " << detections.size() << std::endl;
            return 6;
        }
        if (!saw_expected_label) {
            std::cerr << "Expected label " << expected_label << " was not detected" << std::endl;
            return 5;
        }
    }

    return 0;
}
