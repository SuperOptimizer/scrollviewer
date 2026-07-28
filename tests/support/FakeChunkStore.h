#pragma once

#include <chrono>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "store/ChunkStore.h"

namespace sv::test {

// Deterministic in-memory store for pipeline tests. Requests are held until
// releaseAll()/releaseOne() so tests control completion order and can observe
// intermediate states.
class FakeChunkStore final : public sv::store::ChunkStore {
 public:
  // Keys present in this map succeed with the given bytes; everything else
  // reports NotFound.
  void put(std::string key, std::vector<std::byte> bytes) {
    std::lock_guard lock(mutex_);
    objects_[std::move(key)] = std::move(bytes);
  }

  void read(std::string key, std::stop_token st,
            sv::store::StoreCallback cb) override {
    std::lock_guard lock(mutex_);
    pending_.push_back(Pending{std::move(key), std::move(st), std::move(cb)});
  }

  bool isRemote() const override { return true; }

  std::size_t pendingCount() const {
    std::lock_guard lock(mutex_);
    return pending_.size();
  }

  std::vector<std::string> pendingKeys() const {
    std::lock_guard lock(mutex_);
    std::vector<std::string> keys;
    for (const auto& p : pending_) keys.push_back(p.key);
    return keys;
  }

  // Completes the oldest pending request (respecting its stop_token).
  bool releaseOne() {
    Pending p;
    {
      std::lock_guard lock(mutex_);
      if (pending_.empty()) return false;
      p = std::move(pending_.front());
      pending_.pop_front();
    }
    complete(std::move(p));
    return true;
  }

  void releaseAll() {
    while (releaseOne()) {
    }
  }

 private:
  struct Pending {
    std::string key;
    std::stop_token stop;
    sv::store::StoreCallback cb;
  };

  void complete(Pending p) {
    using sv::store::StoreError;
    if (p.stop.stop_requested()) {
      p.cb(std::unexpected(StoreError{StoreError::Kind::Cancelled, 0, {}}));
      return;
    }
    std::vector<std::byte> bytes;
    {
      std::lock_guard lock(mutex_);
      auto it = objects_.find(p.key);
      if (it == objects_.end()) {
        p.cb(std::unexpected(StoreError{StoreError::Kind::NotFound, 404, {}}));
        return;
      }
      bytes = it->second;
    }
    p.cb(sv::ByteBuffer::copyOf(bytes));
  }

  mutable std::mutex mutex_;
  std::deque<Pending> pending_;
  std::unordered_map<std::string, std::vector<std::byte>> objects_;
};

}  // namespace sv::test
