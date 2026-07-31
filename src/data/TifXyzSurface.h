#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/Result.h"

namespace sv::data {

// Parametric surface in the "tifxyz" segment format used by the Vesuvius
// pipeline: a directory (or URL prefix) holding x.tif, y.tif, z.tif —
// float32 single-channel TIFFs of identical size — where texel (u,v) gives
// the volume-space voxel coordinate to sample. Invalid grid points are
// marked with -1. meta.json carries the grid scale (volume voxels per grid
// step) but the coordinates themselves are absolute, so rendering does not
// need it.
class TifXyzSurface {
 public:
  // Loads from three in-memory TIFF files (fetched by the caller through
  // any ChunkStore, so local paths and URLs both work).
  static Result<std::shared_ptr<TifXyzSurface>> load(
      std::span<const std::byte> xTif, std::span<const std::byte> yTif,
      std::span<const std::byte> zTif);

  std::uint32_t width() const { return w_; }
  std::uint32_t height() const { return h_; }

  // Row-major (v * width + u), xyz in volume voxel coordinates; any
  // component < 0 marks an invalid point.
  const std::vector<std::array<float, 3>>& points() const { return pts_; }

  bool valid(std::uint32_t u, std::uint32_t v) const {
    const auto& p = pts_[std::size_t{v} * w_ + u];
    return p[0] >= 0.f && p[1] >= 0.f && p[2] >= 0.f;
  }

  // Maps every valid point through a 3x4 affine (row-major, x/y/z order):
  // p' = M * (x, y, z, 1). Used to register a segment traced on one volume
  // into another volume's voxel space (see transform.json sidecars).
  void applyAffine(const std::array<std::array<double, 4>, 3>& m);

 private:
  std::uint32_t w_ = 0, h_ = 0;
  std::vector<std::array<float, 3>> pts_;
};

}  // namespace sv::data
