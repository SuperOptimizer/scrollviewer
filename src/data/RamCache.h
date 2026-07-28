#pragma once

#include "core/ByteBuffer.h"
#include "core/LruCache.h"
#include "data/BrickKey.h"

namespace sv::data {

// A decoded chunk (native chunk size, no border) resident in RAM.
struct Brick {
  BrickKey key;
  ByteBuffer voxels;  // chunkVoxels * dtypeSize bytes, z-major (C order)
  bool isFillValue = false;  // chunk was absent or uniform fill
  // Density range (8-bit domain) for occupancy culling in the renderer.
  std::uint8_t minVal = 0;
  std::uint8_t maxVal = 255;
};

// RAM cache of decoded bricks shared by all views. Values are
// shared_ptr<const Brick>: consumers holding a reference stay safe across
// eviction; pinning blocks eviction while a brick awaits GPU upload.
class RamCache {
 public:
  explicit RamCache(std::uint64_t byteBudget) : cache_(byteBudget) {}

  std::shared_ptr<const Brick> get(const BrickKey& key) {
    return cache_.get(key);
  }
  bool contains(const BrickKey& key) const { return cache_.contains(key); }

  std::shared_ptr<const Brick> insertPinned(std::shared_ptr<const Brick> b) {
    const auto bytes = static_cast<std::uint64_t>(b->voxels.size());
    const BrickKey key = b->key;
    return cache_.insert(key, std::move(b), bytes, /*pins=*/1);
  }

  void pin(const BrickKey& key) { cache_.pin(key); }
  void unpin(const BrickKey& key) { cache_.unpin(key); }

  std::uint64_t bytesUsed() const { return cache_.bytesUsed(); }
  std::uint64_t byteBudget() const { return cache_.byteBudget(); }

 private:
  LruCache<BrickKey, Brick, BrickKeyHash> cache_;
};

}  // namespace sv::data
