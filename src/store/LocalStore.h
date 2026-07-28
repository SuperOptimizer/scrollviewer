#pragma once

#include <filesystem>
#include <memory>

#include "core/ThreadPool.h"
#include "store/ChunkStore.h"

namespace sv::store {

// Reads keys as files relative to a root directory, on a shared IO pool.
class LocalStore final : public ChunkStore {
 public:
  LocalStore(std::filesystem::path root, std::shared_ptr<ThreadPool> ioPool);

  void read(std::string key, std::stop_token st, StoreCallback cb) override;
  bool isRemote() const override { return false; }

  const std::filesystem::path& root() const { return root_; }

 private:
  std::filesystem::path root_;
  std::shared_ptr<ThreadPool> ioPool_;
};

}  // namespace sv::store
