#pragma once

#include <cstdint>
#include <functional>

namespace sv::data {

struct ChunkCoord {
  std::uint8_t level = 0;
  std::uint32_t z = 0, y = 0, x = 0;

  bool operator==(const ChunkCoord&) const = default;
};

struct BrickKey {
  std::uint32_t volumeId = 0;
  ChunkCoord coord;

  bool operator==(const BrickKey&) const = default;

  // Packed for hashing; volumeId:8 | level:4 | z:17 | y:17 | x:17 covers
  // Vesuvius chunk grids (level-0 grids are a few hundred per axis) with
  // headroom.
  std::uint64_t packed() const {
    return (std::uint64_t{volumeId & 0xffu} << 55) |
           (std::uint64_t{coord.level & 0xfu} << 51) |
           (std::uint64_t{coord.z & 0x1ffffu} << 34) |
           (std::uint64_t{coord.y & 0x1ffffu} << 17) |
           std::uint64_t{coord.x & 0x1ffffu};
  }
};

struct BrickKeyHash {
  std::size_t operator()(const BrickKey& k) const {
    return std::hash<std::uint64_t>{}(k.packed());
  }
};

}  // namespace sv::data
