#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace sv::test {

// Writes a tiny zarr v2 multiscale tree for tests. Voxel values are a
// deterministic function of (level, z, y, x) so decoded output is verifiable
// per voxel without golden files.
struct ZarrFixtureOptions {
  bool blosc = true;          // blosc1-format frames (as numcodecs writes)
  bool nestedSeparator = true;  // "/" vs "."
  bool uint16 = false;
  bool omitChunk000 = false;  // simulate a missing chunk
};

// Shape 20x17x13, chunks 8^3, two levels ("0" full res, "1" halved).
void writeZarrFixture(const std::filesystem::path& root,
                      const ZarrFixtureOptions& opts);

// The deterministic voxel value at (z,y,x) for a given level.
std::uint16_t fixtureVoxel(int level, std::uint64_t z, std::uint64_t y,
                           std::uint64_t x);

inline constexpr std::uint64_t kFixtureShape[3] = {20, 17, 13};
inline constexpr std::uint32_t kFixtureChunk = 8;

}  // namespace sv::test
