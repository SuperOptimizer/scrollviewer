#include "render/WorkingSetPlanner.h"

#include <vtkCamera.h>
#include <vtkMath.h>
#include <vtkMatrix4x4.h>
#include <vtkRenderer.h>

#include <algorithm>
#include <cmath>

namespace sv::render {

using data::BrickKey;
using data::BrickRequest;
using data::ChunkCoord;

WorkingSetPlanner::WorkingSetPlanner(
    std::array<std::uint64_t, 3> shapeZyx, double spacing,
    std::uint32_t chunkDim, int levelCount,
    std::shared_ptr<const data::VolumeManifest> manifest)
    : shape_(shapeZyx),
      spacing_(spacing),
      chunkDim_(chunkDim),
      levelCount_(levelCount),
      manifest_(std::move(manifest)) {}

std::array<std::uint32_t, 3> WorkingSetPlanner::gridDims(int level) const {
  std::array<std::uint32_t, 3> g;
  for (int a = 0; a < 3; ++a) {
    const std::uint64_t dim =
        std::max<std::uint64_t>(1, (shape_[a] + ((1ull << level) - 1)) >> level);
    g[a] = static_cast<std::uint32_t>((dim + chunkDim_ - 1) / chunkDim_);
  }
  return g;
}

WorkingSetPlanner::Frustum WorkingSetPlanner::extractFrustum(
    vtkRenderer* ren) {
  double aspect[2];
  ren->GetAspect(aspect);
  vtkMatrix4x4* m = ren->GetActiveCamera()->GetCompositeProjectionTransformMatrix(
      aspect[0] / aspect[1], -1, 1);

  // Gribb/Hartmann plane extraction from a world->clip matrix (rows of m).
  Frustum f;
  auto row = [&](int r) {
    return std::array<double, 4>{m->GetElement(r, 0), m->GetElement(r, 1),
                                 m->GetElement(r, 2), m->GetElement(r, 3)};
  };
  const auto r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
  auto combine = [](const std::array<double, 4>& a,
                    const std::array<double, 4>& b, double sign) {
    return std::array<double, 4>{b[0] + sign * a[0], b[1] + sign * a[1],
                                 b[2] + sign * a[2], b[3] + sign * a[3]};
  };
  f.planes[0] = combine(r0, r3, +1);  // left
  f.planes[1] = combine(r0, r3, -1);  // right
  f.planes[2] = combine(r1, r3, +1);  // bottom
  f.planes[3] = combine(r1, r3, -1);  // top
  f.planes[4] = combine(r2, r3, +1);  // near
  f.planes[5] = combine(r2, r3, -1);  // far
  return f;
}

bool WorkingSetPlanner::brickIntersectsFrustum(const Frustum& f, int level,
                                               std::uint32_t z,
                                               std::uint32_t y,
                                               std::uint32_t x) const {
  // Brick AABB in world units.
  const double brickWorld = double(chunkDim_) * spacing_ * double(1ull << level);
  const double bmin[3] = {double(x) * brickWorld, double(y) * brickWorld,
                          double(z) * brickWorld};
  const double bmax[3] = {
      std::min(bmin[0] + brickWorld, double(shape_[2]) * spacing_),
      std::min(bmin[1] + brickWorld, double(shape_[1]) * spacing_),
      std::min(bmin[2] + brickWorld, double(shape_[0]) * spacing_)};

  for (const auto& p : f.planes) {
    // Positive vertex test: if the farthest corner along the plane normal is
    // outside, the whole box is outside.
    const double px = p[0] >= 0 ? bmax[0] : bmin[0];
    const double py = p[1] >= 0 ? bmax[1] : bmin[1];
    const double pz = p[2] >= 0 ? bmax[2] : bmin[2];
    if (p[0] * px + p[1] * py + p[2] * pz + p[3] < 0) return false;
  }
  return true;
}

WorkingSetPlanner::Plan WorkingSetPlanner::plan(vtkRenderer* ren,
                                                std::uint32_t volumeId,
                                                int viewportHeightPx,
                                                const GpuBrickCache& cache,
                                                std::size_t maxRequests) const {
  Plan out;

  // Desired level: voxel size projected at the camera focal distance vs
  // pixel size. parallelScale is half the visible world height (ortho); for
  // perspective use distance * tan(viewAngle/2).
  vtkCamera* cam = ren->GetActiveCamera();
  double halfHeightWorld;
  if (cam->GetParallelProjection()) {
    halfHeightWorld = cam->GetParallelScale();
  } else {
    const double dist = cam->GetDistance();
    halfHeightWorld =
        dist * std::tan(vtkMath::RadiansFromDegrees(cam->GetViewAngle()) / 2);
  }
  const double worldPerPixel =
      2.0 * halfHeightWorld / std::max(1, viewportHeightPx);
  const double voxelsPerPixel = worldPerPixel / spacing_;
  // Level tracks zoom so we never fetch finer than the screen resolves:
  // >=1 px per voxel -> level 0, 0.5-1 px -> level 1, 0.25-0.5 px -> 2, ...
  int desired =
      static_cast<int>(std::ceil(std::log2(std::max(1.0, voxelsPerPixel))));
  desired = std::clamp(desired, 0, levelCount_ - 1);
  out.desiredLevel = desired;
  out.voxelsPerPixel = voxelsPerPixel;

  const Frustum frustum = extractFrustum(ren);

  // Hierarchical descent: start at the coarsest level, recurse into children
  // of visible bricks until the desired level. Coarser levels get strictly
  // higher priority so the fallback chain fills top-down.
  struct Node {
    int level;
    std::uint32_t z, y, x;
  };
  std::vector<Node> stack;
  {
    const auto g = gridDims(levelCount_ - 1);
    for (std::uint32_t z = 0; z < g[0]; ++z)
      for (std::uint32_t y = 0; y < g[1]; ++y)
        for (std::uint32_t x = 0; x < g[2]; ++x)
          stack.push_back({levelCount_ - 1, z, y, x});
  }

  const double camPos[3] = {cam->GetPosition()[0], cam->GetPosition()[1],
                            cam->GetPosition()[2]};

  while (!stack.empty()) {
    const Node n = stack.back();
    stack.pop_back();
    if (!brickIntersectsFrustum(frustum, n.level, n.z, n.y, n.x)) continue;
    // Manifest pruning: known-empty/invisible subtrees are never requested
    // and never recursed into (children of an empty region are empty too).
    if (manifest_ && manifest_->hasLevel(n.level) &&
        manifest_->invisible(n.level, n.z, n.y, n.x, visThreshold_))
      continue;

    const BrickKey key{volumeId,
                       ChunkCoord{static_cast<std::uint8_t>(n.level), n.z, n.y,
                                  n.x}};
    if (cache.isResident(key)) {
      out.visible.push_back(key);
    } else if (out.missing.size() < maxRequests) {
      // Priority: coarser level dominates; nearer bricks first within level.
      const double brickWorld =
          double(chunkDim_) * spacing_ * double(1ull << n.level);
      const double cx = (double(n.x) + 0.5) * brickWorld - camPos[0];
      const double cy = (double(n.y) + 0.5) * brickWorld - camPos[1];
      const double cz = (double(n.z) + 0.5) * brickWorld - camPos[2];
      const double dist = std::sqrt(cx * cx + cy * cy + cz * cz);
      const float priority =
          1000.f * float(n.level) + 1000.f / (1.f + float(dist));
      out.missing.push_back(BrickRequest{key, priority});
    }

    if (n.level > desired) {
      const auto cg = gridDims(n.level - 1);
      for (std::uint32_t dz = 0; dz < 2; ++dz)
        for (std::uint32_t dy = 0; dy < 2; ++dy)
          for (std::uint32_t dx = 0; dx < 2; ++dx) {
            const std::uint32_t cz2 = n.z * 2 + dz, cy2 = n.y * 2 + dy,
                                cx2 = n.x * 2 + dx;
            if (cz2 < cg[0] && cy2 < cg[1] && cx2 < cg[2])
              stack.push_back({n.level - 1, cz2, cy2, cx2});
          }
    }
  }

  return out;
}

}  // namespace sv::render
