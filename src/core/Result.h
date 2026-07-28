#pragma once

#include <expected>
#include <format>
#include <string>

namespace sv {

// Generic error for parse/validation/decode failures. Store IO has its own
// richer StoreError; this is for everything else.
struct Error {
  std::string message;

  template <class... Args>
  static Error fmt(std::format_string<Args...> f, Args&&... args) {
    return Error{std::format(f, std::forward<Args>(args)...)};
  }
};

template <class T>
using Result = std::expected<T, Error>;

}  // namespace sv
