#pragma once

#include "aim_sim_bridge/aim_types.hpp"

#include <filesystem>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>

namespace aim_sim_bridge
{

class Stage3ObservationJsonlSink final : public IPreTrackerObservationSink
{
public:
    Stage3ObservationJsonlSink(std::filesystem::path path, std::string session_id);
    ~Stage3ObservationJsonlSink() override;

    bool submit(PreTrackerObservationFrame frame) override;
    bool healthy() const override;
    std::uint64_t submitted() const override;
    std::uint64_t failed() const override;

private:
    void workerLoop();
    bool writeLine(const std::string& line);

    std::filesystem::path path_;
    std::string session_id_;
    mutable std::mutex mutex_;
    std::ofstream stream_;
    std::condition_variable ready_;
    std::deque<std::string> pending_;
    bool stopping_ = false;
    std::thread worker_;
    std::uint64_t submitted_ = 0;
    std::uint64_t failed_ = 0;
    bool healthy_ = true;
};

std::shared_ptr<IPreTrackerObservationSink> createStage3ObservationSinkFromEnv();

}  // namespace aim_sim_bridge
