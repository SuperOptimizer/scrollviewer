#include "core/Log.h"

#include <chrono>
#include <cstdio>
#include <mutex>

namespace sv {

namespace {
std::mutex g_logMutex;

const char* levelTag(LogLevel level) {
  switch (level) {
    case LogLevel::Debug: return "DBG";
    case LogLevel::Info: return "INF";
    case LogLevel::Warn: return "WRN";
    case LogLevel::Error: return "ERR";
  }
  return "???";
}
}  // namespace

void logMessage(LogLevel level, std::string_view msg) {
  const auto now = std::chrono::system_clock::now();
  std::lock_guard lock(g_logMutex);
  std::fprintf(stderr, "[%s] %s %.*s\n",
               std::format("{:%H:%M:%S}", std::chrono::floor<std::chrono::milliseconds>(now)).c_str(),
               levelTag(level), static_cast<int>(msg.size()), msg.data());
}

}  // namespace sv
