#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "data/ChunkFetchPipeline.h"
#include "data/VolumeManifest.h"
#include "render/GpuBrickCache.h"

class vtkRenderer;

namespace sv::render {

// Decides which bricks each frame wants: hierarchical descent of the LOD
// brick octree over the view frustum, stopping at the level the current zoom
// warrants. Emits prioritized requests for bricks not yet GPU-resident
// (coarse levels first so the fallback chain always fills top-down).
class WorkingSetPlanner {
 public:
  WorkingSetPlanner(std::array<std::uint64_t, 3> shapeZyx, double spacing,
                    std::uint32_t chunkDim, int levelCount,
                    std::shared_ptr<const data::VolumeManifest> manifest = {});

  // Bricks whose manifest max density is <= this are skipped entirely
  // (never requested, never recursed into). Track the visibility window.
  void setVisibilityThreshold(std::uint8_t t) { visThreshold_ = t; }

  struct Plan {
    std::vector<data::BrickRequest> missing;   // to submit to the pipeline
    std::vector<data::BrickKey> visible;       // resident: LRU-touch these
    int desiredLevel = 0;
    double voxelsPerPixel = 0.0;  // diagnostic: what desiredLevel came from
  };

  // viewportHeightPx: pixels; used to pick the level where one voxel is
  // roughly >= one pixel.
  Plan plan(vtkRenderer* ren, std::uint32_t volumeId, int viewportHeightPx,
            const GpuBrickCache& cache, std::size_t maxRequests = 4096) const;

 private:
  struct Frustum {
    std::array<std::array<double, 4>, 6> planes;
  };
  static Frustum extractFrustum(vtkRenderer* ren);
  bool brickIntersectsFrustum(const Frustum& f, int level, std::uint32_t z,
                              std::uint32_t y, std::uint32_t x) const;
  std::array<std::uint32_t, 3> gridDims(int level) const;

  std::array<std::uint64_t, 3> shape_;  // z,y,x full-res voxels
  double spacing_;
  std::uint32_t chunkDim_;
  int levelCount_;
  std::shared_ptr<const data::VolumeManifest> manifest_;
  std::uint8_t visThreshold_ = 0;
};

}  // namespace sv::render
