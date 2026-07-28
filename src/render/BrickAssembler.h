#pragma once

#include <cstdint>
#include <vector>

#include "data/BrickKey.h"
#include "data/RamCache.h"
#include "zarr/OmeZarrVolume.h"

namespace sv::render {

// Builds bordered upload-ready bricks: chunk payload (e.g. 128^3) plus a
// 1-voxel border on every side (130^3) so hardware trilinear filtering is
// seamless across brick boundaries. Border voxels come from neighbor chunks
// when they are resident in the RamCache; otherwise the edge voxel is
// duplicated (clamp) — visually fine, patched when neighbors arrive.
//
// Output is always 8-bit: uint16 sources are windowed to 8 bits here.
class BrickAssembler {
 public:
  BrickAssembler(std::shared_ptr<zarr::OmeZarrVolume> volume,
                 std::uint32_t volumeId, data::RamCache& ram);

  std::uint32_t borderedDim() const { return chunkDim_ + 2; }
  std::size_t borderedBytes() const {
    const std::size_t d = borderedDim();
    return d * d * d;
  }

  // Fills out (borderedBytes() bytes, z-major). brick is the payload chunk.
  void assemble(const data::Brick& brick, std::uint8_t* out) const;

 private:
  std::uint8_t voxelFrom(const data::Brick& b, std::uint64_t z, std::uint64_t y,
                         std::uint64_t x) const;  // local chunk coords

  std::shared_ptr<zarr::OmeZarrVolume> volume_;
  std::uint32_t volumeId_;
  data::RamCache& ram_;
  std::uint32_t chunkDim_;
};

}  // namespace sv::render
