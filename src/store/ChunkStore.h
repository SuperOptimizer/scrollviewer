#pragma once

#include <cstdint>
#include <expected>
#include <functional>
#include <stop_token>
#include <string>

#include "core/ByteBuffer.h"

namespace sv::store {

struct StoreError {
  enum class Kind { NotFound, Io, Http, Cancelled, Auth };
  Kind kind = Kind::Io;
  int code = 0;  // errno or HTTP status, informational
  std::string message;

  bool isNotFound() const { return kind == Kind::NotFound; }
};

using StoreResult = std::expected<ByteBuffer, StoreError>;
using StoreCallback = std::move_only_function<void(StoreResult)>;

// Async key/value byte store. Keys are '/'-separated relative paths
// (zarr layout). Callbacks are invoked on an internal store thread — callers
// must trampoline to their own thread if needed. NotFound is a normal outcome
// for zarr (missing chunk => fill_value).
class ChunkStore {
 public:
  virtual ~ChunkStore() = default;

  virtual void read(std::string key, std::stop_token st, StoreCallback cb) = 0;

  // Byte-range read; needed later for zarr v3 shard indexes. Default
  // implementation reads the whole object and slices (correct, wasteful).
  virtual void readRange(std::string key, std::uint64_t offset,
                         std::uint64_t length, std::stop_token st,
                         StoreCallback cb);

  virtual bool isRemote() const = 0;
};

}  // namespace sv::store
