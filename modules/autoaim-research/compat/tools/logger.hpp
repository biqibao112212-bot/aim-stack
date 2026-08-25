#pragma once

#include <memory>

namespace tools {

class ResearchNullLogger {
 public:
  template <typename... Args>
  void debug(const char*, Args&&...) const noexcept {}

  template <typename... Args>
  void warn(const char*, Args&&...) const noexcept {}
};

inline std::shared_ptr<ResearchNullLogger> logger() {
  static const auto instance = std::make_shared<ResearchNullLogger>();
  return instance;
}

}  // namespace tools
