#include "render/BrickAssembler.h"

#include <algorithm>

namespace sv::render {

using data::Brick;
using data::BrickKey;
using data::ChunkCoord;

BrickAssembler::BrickAssembler(std::shared_ptr<zarr::OmeZarrVolume> volume,
                               std::uint32_t volumeId, data::RamCache& ram)
    : volume_(std::move(volume)), volumeId_(volumeId), ram_(ram) {
  chunkDim_ = volume_->level(0).chunks[0];
}

std::uint8_t BrickAssembler::voxelFrom(const Brick& b, std::uint64_t z,
                                       std::uint64_t y, std::uint64_t x) const {
  const auto& meta = volume_->level(b.key.coord.level);
  if (b.isFillValue || b.voxels.empty())
    return static_cast<std::uint8_t>(meta.fillValue);
  const std::size_t idx = (z * meta.chunks[1] + y) * meta.chunks[2] + x;
  if (meta.dtype == zarr::Dtype::U8)
    return static_cast<std::uint8_t>(b.voxels.span()[idx]);
  // uint16 -> 8-bit window (full range for v1).
  const auto* p16 = reinterpret_cast<const std::uint16_t*>(b.voxels.data());
  return static_cast<std::uint8_t>(p16[idx] >> 8);
}

void BrickAssembler::assemble(const Brick& brick, std::uint8_t* out) const {
  const std::int64_t dim = chunkDim_;
  const std::int64_t bd = dim + 2;
  const auto& c = brick.key.coord;
  const auto& meta = volume_->level(c.level);
  const auto grid = meta.chunkGridDims();
  const bool isU8 = meta.dtype == zarr::Dtype::U8;

  // Prefetch the 3x3x3 chunk neighborhood once: border voxels then read
  // neighbors directly instead of hammering the RamCache mutex per voxel.
  // nb[dz+1][dy+1][dx+1]; null => neighbor absent or out of grid (border
  // falls back to clamping into the payload chunk).
  std::shared_ptr<const Brick> nbHold[3][3][3];
  const Brick* nb[3][3][3] = {};
  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        if (dz == 0 && dy == 0 && dx == 0) {
          nb[1][1][1] = &brick;
          continue;
        }
        const std::int64_t nz = std::int64_t(c.z) + dz;
        const std::int64_t ny = std::int64_t(c.y) + dy;
        const std::int64_t nx = std::int64_t(c.x) + dx;
        if (nz < 0 || ny < 0 || nx < 0 || nz >= grid[0] || ny >= grid[1] ||
            nx >= grid[2])
          continue;
        auto& hold = nbHold[dz + 1][dy + 1][dx + 1];
        hold = ram_.get(BrickKey{
            volumeId_,
            ChunkCoord{c.level, static_cast<std::uint32_t>(nz),
                       static_cast<std::uint32_t>(ny),
                       static_cast<std::uint32_t>(nx)}});
        if (hold) nb[dz + 1][dy + 1][dx + 1] = hold.get();
      }

  // Reads voxel at chunk-local coords possibly outside [0,dim): resolves the
  // owning neighbor or clamps into the payload when it is absent.
  auto sample = [&](std::int64_t z, std::int64_t y, std::int64_t x)
      -> std::uint8_t {
    const int cz = z < 0 ? 0 : (z >= dim ? 2 : 1);
    const int cy = y < 0 ? 0 : (y >= dim ? 2 : 1);
    const int cx = x < 0 ? 0 : (x >= dim ? 2 : 1);
    const Brick* src = nb[cz][cy][cx];
    if (!src) {
      // Clamp into the payload chunk (correct at volume edges; a transient
      // approximation next to not-yet-loaded neighbors).
      return voxelFrom(brick, std::clamp<std::int64_t>(z, 0, dim - 1),
                       std::clamp<std::int64_t>(y, 0, dim - 1),
                       std::clamp<std::int64_t>(x, 0, dim - 1));
    }
    return voxelFrom(*src, z - (std::int64_t(cz) - 1) * dim,
                     y - (std::int64_t(cy) - 1) * dim,
                     x - (std::int64_t(cx) - 1) * dim);
  };

  for (std::int64_t z = -1; z <= dim; ++z)
    for (std::int64_t y = -1; y <= dim; ++y) {
      std::uint8_t* row = out + ((z + 1) * bd + (y + 1)) * bd;
      const bool interiorZY = (z >= 0 && z < dim && y >= 0 && y < dim);
      if (interiorZY && isU8) {
        // Interior row: straight copy; only the two border texels resolve
        // through the neighborhood.
        const auto* src =
            reinterpret_cast<const std::uint8_t*>(brick.voxels.data());
        std::copy_n(src + (z * dim + y) * dim, dim, row + 1);
        row[0] = sample(z, y, -1);
        row[bd - 1] = sample(z, y, dim);
      } else {
        for (std::int64_t x = -1; x <= dim; ++x) row[x + 1] = sample(z, y, x);
      }
    }
}

}  // namespace sv::render
