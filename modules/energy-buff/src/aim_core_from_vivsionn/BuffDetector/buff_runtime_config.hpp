#ifndef AUTO_BUFF__BUFF_RUNTIME_CONFIG_HPP
#define AUTO_BUFF__BUFF_RUNTIME_CONFIG_HPP

#include "runtime_paths.h"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <string>

namespace auto_buff
{

struct BuffRuntimeConfig
{
    int exposure_time_ms = 5;
    double gain = 10.0;
    bool correction_enabled = false;
    double correction_cx = 0.0;
    double correction_cy = 0.0;
};

inline std::string default_buff_config_path()
{
    return rm::runtime_paths::repoPath("src/BuffDetector/buff_config.yaml").string();
}

inline std::string resolve_buff_config_path(const std::string& config_path = {})
{
    namespace fs = std::filesystem;

    const fs::path raw = config_path.empty() ? fs::path(default_buff_config_path())
                                             : fs::path(config_path);
    if (raw.is_absolute() && fs::exists(raw)) {
        return raw.string();
    }

    const fs::path repo_path = rm::runtime_paths::repoPath(raw);
    if (fs::exists(repo_path)) {
        return repo_path.string();
    }

    const fs::path cwd_path = fs::current_path() / raw;
    if (fs::exists(cwd_path)) {
        return cwd_path.string();
    }

    return raw.string();
}

template <typename T>
inline void read_optional_yaml_scalar(const YAML::Node& node, const char* key, T& value)
{
    if (node && node[key]) {
        value = node[key].as<T>();
    }
}

inline BuffRuntimeConfig load_buff_runtime_config(const std::string& config_path = {})
{
    BuffRuntimeConfig config;
    const YAML::Node yaml = YAML::LoadFile(resolve_buff_config_path(config_path));

    const YAML::Node camera_yaml = yaml["buff_camera"];
    read_optional_yaml_scalar(camera_yaml, "exposure_time_ms", config.exposure_time_ms);
    read_optional_yaml_scalar(camera_yaml, "gain", config.gain);

    const YAML::Node correction_yaml = yaml["buff_correction"];
    read_optional_yaml_scalar(correction_yaml, "enabled", config.correction_enabled);
    read_optional_yaml_scalar(correction_yaml, "cx", config.correction_cx);
    read_optional_yaml_scalar(correction_yaml, "cy", config.correction_cy);

    if (config.exposure_time_ms < 1) {
        config.exposure_time_ms = 1;
    }
    if (config.gain < 0.0) {
        config.gain = 0.0;
    }
    return config;
}

}  // namespace auto_buff

#endif  // AUTO_BUFF__BUFF_RUNTIME_CONFIG_HPP
