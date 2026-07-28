#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "core/ByteBuffer.h"
#include "core/Result.h"

namespace sv::data {

// Precomputed per-brick statistics sidecar ("SVMF" format). For every LOD
// level it stores (min,max) density per brick in the 8-bit domain; absent
// chunks are encoded as min == max == fillValue, which renders identically
// to a uniform fill brick. Enables:
//   - instant EMPTY page-table seeding at volume open (no fetches, no 404s)
//   - occupancy culling BEFORE fetch (invisible bricks are never requested)
//   - planner descent pruning through known-empty subtrees
//
// Layout (little-endian):
//   "SVMF" | u32 version=1 | u8 fillValue | u8 levelCount | u16 reserved
//   per level: u32 gridZ, gridY, gridX, then gridZ*gridY*gridX * 2 bytes
//   of (min,max), x fastest (same order as the GPU page tables).
class VolumeManifest {
 public:
  struct MinMax {
    std::uint8_t min = 0;
    std::uint8_t max = 255;
  };

  static Result<std::shared_ptr<VolumeManifest>> parse(
      std::span<const std::byte> bytes);

  int levelCount() const { return static_cast<int>(levels_.size()); }
  std::uint8_t fillValue() const { return fill_; }

  bool hasLevel(int level) const {
    return level >= 0 && level < levelCount() &&
           !levels_[level].data.empty();
  }

  MinMax at(int level, std::uint32_t z, std::uint32_t y,
            std::uint32_t x) const {
    const Level& l = levels_[level];
    if (z >= l.grid[0] || y >= l.grid[1] || x >= l.grid[2]) return {};
    const std::size_t i =
        ((std::size_t{z} * l.grid[1] + y) * l.grid[2] + x) * 2;
    return {l.data[i], l.data[i + 1]};
  }

  // True when the brick is absent or uniform fill: nothing to fetch/render.
  bool empty(int level, std::uint32_t z, std::uint32_t y,
             std::uint32_t x) const {
    const MinMax mm = at(level, z, y, x);
    return mm.min == mm.max && mm.min == fill_;
  }

  // True when the brick cannot produce visible output under a density
  // window whose zero-opacity cutoff is `threshold` (8-bit domain).
  bool invisible(int level, std::uint32_t z, std::uint32_t y, std::uint32_t x,
                 std::uint8_t threshold) const {
    return at(level, z, y, x).max <= threshold;
  }

 private:
  struct Level {
    std::array<std::uint32_t, 3> grid{};  // z,y,x bricks
    std::vector<std::uint8_t> data;       // 2 bytes per brick
  };
  std::vector<Level> levels_;
  std::uint8_t fill_ = 0;
};

}  // namespace sv::data
