#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/ThreadPool.h"
#include "data/TifXyzSurface.h"

namespace sv::render {

// Sparse binary shell mask around a parametric surface, in native (level-0)
// voxel space: one 16^3-cell block (8-voxel cells) per 128^3 brick the
// shell touches. Built by sweeping the surface along its normals from
// -front to +behind voxels; a voxel is "in" when a swept sample lands in
// its cell. GPU side: a dense R32UI directory over the level-0 brick grid
// (0 = outside, else blockSlot + 1) plus an R8 block pool.
class SurfaceMask {
 public:
  static constexpr std::uint32_t kCellVox = 8;    // voxels per cell axis
  static constexpr std::uint32_t kBlockDim = 16;  // cells per brick axis

  // Threaded CPU build. Surface points must already be in this volume's
  // voxel space. sampleSpacing: target spacing (voxels) of swept samples.
  // brickGrids: the render cache's per-level brick-grid dims (z,y,x); a
  // per-level presence mip is derived so the raymarcher can skip bricks
  // (at any LOD) the shell does not touch.
  void build(const data::TifXyzSurface& surface,
             const std::array<std::uint64_t, 3>& shapeZyx, float front,
             float behind, float sampleSpacing, ThreadPool& pool,
             const std::vector<std::array<std::uint32_t, 3>>& brickGrids);

  // Creates/updates the GL textures (GL context must be current).
  void upload();
  bool uploaded() const { return dirTex_ != 0; }
  void releaseGraphicsResources();

  unsigned int directoryTexture() const { return dirTex_; }
  unsigned int blockPoolTexture() const { return poolTex_; }
  int presenceLevelCount() const { return int(presence_.size()); }
  unsigned int presenceTexture(int level) const { return presTex_[level]; }
  std::array<std::uint32_t, 3> poolDims() const { return poolDims_; }
  std::array<std::uint32_t, 3> brickGrid() const { return grid_; }
  std::size_t blockCount() const { return blocks_.size(); }
  bool built() const { return !blocks_.empty(); }

 private:
  std::array<std::uint32_t, 3> grid_{};  // level-0 brick grid (z,y,x)
  // brick index (z*gy + y)*gx + x -> 16^3 cell occupancy block
  std::unordered_map<std::uint32_t, std::vector<std::uint8_t>> blocks_;

  // Per render-LOD brick presence (1 = shell intersects this brick).
  std::vector<std::array<std::uint32_t, 3>> presGrids_;
  std::vector<std::vector<std::uint8_t>> presence_;
  std::vector<unsigned int> presTex_;

  unsigned int dirTex_ = 0, poolTex_ = 0;
  std::array<std::uint32_t, 3> poolDims_{};  // blocks per axis (z,y,x)
  bool dirty_ = false;
};

}  // namespace sv::render
