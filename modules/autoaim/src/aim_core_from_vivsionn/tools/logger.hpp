#ifndef TOOLS_LOGGER_HPP
#define TOOLS_LOGGER_HPP

#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace tools
{

class Logger
{
public:
    template <typename... Args>
    void info(const std::string& message, Args&&... args)
    {
        log("INFO", message, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void warn(const std::string& message, Args&&... args)
    {
        log("WARN", message, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void error(const std::string& message, Args&&... args)
    {
        log("ERROR", message, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void debug(const std::string& message, Args&&... args)
    {
        if (!debug_enabled()) return;
        log("DEBUG", message, std::forward<Args>(args)...);
    }

private:
    template <typename T>
    std::string to_string_any(T&& value)
    {
        std::ostringstream oss;
        oss << std::forward<T>(value);
        return oss.str();
    }

    std::string format_message(const std::string& message, const std::vector<std::string>& values)
    {
        std::string output;
        output.reserve(message.size() + values.size() * 8);

        size_t value_index = 0;
        for (size_t i = 0; i < message.size();) {
            if (message[i] == '{') {
                const size_t end = message.find('}', i + 1);
                if (end != std::string::npos && value_index < values.size()) {
                    output += values[value_index++];
                    i = end + 1;
                    continue;
                }
            }
            output.push_back(message[i++]);
        }
        return output;
    }

    bool debug_enabled() const
    {
        static const bool enabled = [] {
            const char* env = std::getenv("BUFF_VERBOSE_LOG");
            return env != nullptr && std::string(env) != "0";
        }();
        return enabled;
    }

    template <typename... Args>
    void log(const char* level, const std::string& message, Args&&... args)
    {
        std::vector<std::string> values;
        values.reserve(sizeof...(args));
        (values.push_back(to_string_any(std::forward<Args>(args))), ...);
        std::cerr << "[auto_buff][" << level << "] "
                  << format_message(message, values) << std::endl;
    }
};

inline std::shared_ptr<Logger> logger()
{
    static auto instance = std::make_shared<Logger>();
    return instance;
}

}  // namespace tools

#endif  // TOOLS_LOGGER_HPP
