#pragma once

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace sv {

// Byte-budgeted LRU cache with pinning. Values are held as
// shared_ptr<const V>, so a consumer that grabbed a value stays safe even if
// the entry is evicted afterwards; pinning additionally blocks eviction while
// a value is queued for further processing (e.g. GPU upload).
//
// Eviction happens on insert when over budget, skipping pinned entries. If
// everything is pinned we go temporarily over budget rather than blocking or
// failing: producers must never stall on the cache.
template <class K, class V, class Hash = std::hash<K>>
class LruCache {
 public:
  explicit LruCache(std::uint64_t byteBudget) : budget_(byteBudget) {}

  std::shared_ptr<const V> get(const K& key) {
    std::lock_guard lock(mutex_);
    auto it = map_.find(key);
    if (it == map_.end()) return nullptr;
    lru_.splice(lru_.begin(), lru_, it->second);  // move to front
    return it->second->value;
  }

  bool contains(const K& key) const {
    std::lock_guard lock(mutex_);
    return map_.contains(key);
  }

  // Inserts (or replaces) and returns the stored value with pinCount = pins.
  std::shared_ptr<const V> insert(const K& key, std::shared_ptr<const V> value,
                                  std::uint64_t bytes, int pins = 0) {
    std::lock_guard lock(mutex_);
    if (auto it = map_.find(key); it != map_.end()) {
      bytesUsed_ -= it->second->bytes;
      lru_.erase(it->second);
      map_.erase(it);
    }
    lru_.push_front(Entry{key, std::move(value), bytes, pins});
    map_[key] = lru_.begin();
    bytesUsed_ += bytes;
    evictLocked();
    return lru_.front().value;
  }

  void pin(const K& key) {
    std::lock_guard lock(mutex_);
    if (auto it = map_.find(key); it != map_.end()) ++it->second->pinCount;
  }

  void unpin(const K& key) {
    std::lock_guard lock(mutex_);
    if (auto it = map_.find(key); it != map_.end() && it->second->pinCount > 0)
      --it->second->pinCount;
  }

  std::uint64_t bytesUsed() const {
    std::lock_guard lock(mutex_);
    return bytesUsed_;
  }
  std::uint64_t byteBudget() const { return budget_; }

  std::size_t entryCount() const {
    std::lock_guard lock(mutex_);
    return map_.size();
  }

 private:
  struct Entry {
    K key;
    std::shared_ptr<const V> value;
    std::uint64_t bytes;
    int pinCount;
  };

  void evictLocked() {
    // Walk from LRU tail; skip pinned. Stop when under budget or nothing
    // evictable remains.
    auto it = lru_.end();
    while (bytesUsed_ > budget_ && it != lru_.begin()) {
      --it;
      if (it->pinCount > 0) continue;
      bytesUsed_ -= it->bytes;
      map_.erase(it->key);
      it = lru_.erase(it);
    }
  }

  const std::uint64_t budget_;
  mutable std::mutex mutex_;
  std::list<Entry> lru_;  // front = most recent
  std::unordered_map<K, typename std::list<Entry>::iterator, Hash> map_;
  std::uint64_t bytesUsed_ = 0;
};

}  // namespace sv
