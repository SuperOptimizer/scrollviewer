#pragma once

#include <atomic>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>

#include "core/ThreadPool.h"
#include "store/ChunkStore.h"

namespace sv::store {

// Optional recompression applied to cached entries: fetched bytes are encoded
// once before hitting disk and decoded back on every cache hit. Entries are
// stored under `<key><suffix()>` so transcoded and verbatim files coexist in
// one cache tree. Implementations must be thread-safe (called concurrently
// from IO pool threads).
class CacheTranscoder {
 public:
  virtual ~CacheTranscoder() = default;
  // Encoded bytes to store, or nullopt to store `raw` verbatim (payload not
  // eligible — wrong size, metadata file, ...).
  virtual std::optional<ByteBuffer> encode(std::span<const std::byte> raw) = 0;
  // Original bytes from a stored entry; nullopt means corrupt (the entry is
  // deleted and the read falls through to the remote).
  virtual std::optional<ByteBuffer> decode(std::span<const std::byte> stored) = 0;
  virtual const char* suffix() const = 0;  // e.g. ".gdct"
};

// Write-through disk cache decorator for a remote store. Caches raw
// (still-compressed) object bytes; the cache directory mirrors the remote key
// layout, so a cached .zarr is itself a valid local zarr tree.
//
// Eviction is deliberately boring: on start() and periodically thereafter the
// cache is scanned and, if over budget, oldest files (by last write time) are
// deleted down to 90% of budget. Corrupt entries are handled by consumers
// failing decode -> invalidate() -> refetch.
class DiskCacheStore final : public ChunkStore {
 public:
  DiskCacheStore(std::shared_ptr<ChunkStore> inner,
                 std::filesystem::path cacheRoot, std::uint64_t byteBudget,
                 std::shared_ptr<ThreadPool> ioPool);

  void read(std::string key, std::stop_token st, StoreCallback cb) override;
  bool isRemote() const override { return true; }

  // Deletes a cached entry (e.g. after a decode failure) so the next read
  // refetches from the remote.
  void invalidate(const std::string& key);

  // Installs (or clears) the recompression codec. Callers set this once after
  // reading volume metadata, before chunk traffic starts; existing verbatim
  // entries keep being served as-is.
  void setTranscoder(std::shared_ptr<CacheTranscoder> t) {
    transcoder_.store(std::move(t), std::memory_order_release);
  }

  // Scan and evict if over budget. Called from a maintenance thread/timer.
  void runEviction();

 private:
  std::filesystem::path pathFor(const std::string& key) const;

  std::shared_ptr<ChunkStore> inner_;
  std::filesystem::path root_;
  std::uint64_t budget_;
  std::shared_ptr<ThreadPool> ioPool_;
  std::atomic<std::shared_ptr<CacheTranscoder>> transcoder_;
};

}  // namespace sv::store
