#pragma once

#include <filesystem>
#include <memory>

#include "core/ThreadPool.h"
#include "store/ChunkStore.h"

namespace sv::store {

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

  // Scan and evict if over budget. Called from a maintenance thread/timer.
  void runEviction();

 private:
  std::filesystem::path pathFor(const std::string& key) const;

  std::shared_ptr<ChunkStore> inner_;
  std::filesystem::path root_;
  std::uint64_t budget_;
  std::shared_ptr<ThreadPool> ioPool_;
};

}  // namespace sv::store
