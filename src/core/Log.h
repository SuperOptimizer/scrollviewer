#pragma once

#include <format>
#include <string_view>

namespace sv {

enum class LogLevel { Debug, Info, Warn, Error };

void logMessage(LogLevel level, std::string_view msg);

template <class... Args>
void logDebug(std::format_string<Args...> f, Args&&... args) {
  logMessage(LogLevel::Debug, std::format(f, std::forward<Args>(args)...));
}
template <class... Args>
void logInfo(std::format_string<Args...> f, Args&&... args) {
  logMessage(LogLevel::Info, std::format(f, std::forward<Args>(args)...));
}
template <class... Args>
void logWarn(std::format_string<Args...> f, Args&&... args) {
  logMessage(LogLevel::Warn, std::format(f, std::forward<Args>(args)...));
}
template <class... Args>
void logError(std::format_string<Args...> f, Args&&... args) {
  logMessage(LogLevel::Error, std::format(f, std::forward<Args>(args)...));
}

}  // namespace sv
