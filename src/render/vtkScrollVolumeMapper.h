#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>

#include <vtkVolumeMapper.h>

#include "data/TifXyzSurface.h"
#include "render/GpuBrickCache.h"
#include "render/SurfaceMask.h"

class vtkOpenGLQuadHelper;
class vtkOpenGLRenderWindow;
class vtkShaderProgram;
class vtkMatrix4x4;

namespace sv::render {

// Out-of-core virtual-texture volume raycaster. Renders a full-screen quad;
// the fragment shader intersects the volume AABB analytically and marches it,
// resolving each sample through the per-level page tables with fallback to
// coarser levels (progressive refinement). No vtkImageData input: the
// dataset is the brick cache + volume extents.
class vtkScrollVolumeMapper : public vtkVolumeMapper {
 public:
  static vtkScrollVolumeMapper* New();
  vtkTypeMacro(vtkScrollVolumeMapper, vtkVolumeMapper);

  void Render(vtkRenderer* ren, vtkVolume* vol) override;
  double* GetBounds() override;
  void ReleaseGraphicsResources(vtkWindow* window) override;

 protected:
  int FillInputPortInformation(int port, vtkInformation* info) override;

 public:

  // Wiring (not VTK-idiomatic setters; this mapper is internal to the app).
  void SetBrickCache(GpuBrickCache* cache) { Cache = cache; }
  // Full-resolution volume extent in voxels (z,y,x) and voxel spacing.
  void SetVolumeExtent(const std::array<std::uint64_t, 3>& shapeZyx,
                       double spacing);
  void SetChunkDim(std::uint32_t dim) { ChunkDim = dim; }

  // Invoked at the start of every Render with the GL context current —
  // the app drains the fetch pipeline and uploads bricks here.
  void SetPreRenderCallback(std::function<void()> cb) {
    PreRender = std::move(cb);
  }

  // Ray-guided streaming (GigaVoxels-style): bricks the shader wanted but
  // had to substitute with a coarser level last frame. Drained per frame by
  // the app and merged into the fetch queue at high priority.
  std::vector<data::ChunkCoord> TakeFeedbackRequests() {
    return std::move(FeedbackRequests);
  }

  // Rendering quality/LOD controls.
  void SetDesiredLevel(float level) { DesiredLevel = level; }
  void SetSampleStepScale(float s) { SampleStepScale = s; }
  // < 1: raymarch into a reduced-res offscreen target, composited upscaled
  // (interaction quality mode).
  void SetRenderScale(float s) { RenderScale = s; }
  void SetWindowLevel(float window, float level) {
    Window = window;
    Level = level;
  }
  void SetOpacityScale(float s) { OpacityScale = s; }
  // 0 = x-ray (emission/absorption), 1 = shaded DVR (gradient-lit, warm
  // ramp), 2 = opaque isosurface at winCenter (Cook-Torrance shading).
  // A uniform switch — no shader rebuild.
  void SetRenderMode(int m) { RenderMode = m; }
  // GPU per-sample 3D processing: op 0 = none, 1 = smooth (3D box),
  // 2 = sharpen (3D unsharp mask), 3 = edge field (gradient magnitude).
  // floorV is a high-pass cutoff: filtered density below it reads as zero
  // (and feeds the brick/cell occupancy culling, so it prunes work too).
  // Directional key light (world-space, toward the light), ambient/fill
  // scale, and surface-mode volumetric shadow rays.
  void SetLighting(const std::array<float, 3>& dir, float ambient,
                   bool shadows) {
    const float len = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] +
                                dir[2] * dir[2]);
    for (int i = 0; i < 3; ++i) LightDir[i] = dir[i] / std::max(len, 1e-6f);
    LightAmbient = ambient;
    ShadowsOn = shadows;
  }
  // Nearest-neighbor pool taps instead of trilinear: 1 texel per sample
  // instead of 8, a real win when rays are bandwidth-bound (deep zoom at
  // fine LODs). Blocky up close, by design.
  void SetNearestSampling(bool nn) { NearestTaps = nn; }
  void SetVoxelFilter(int op, float amount, float floorV) {
    FilterOp = op;
    FilterAmount = amount;
    FilterFloor = floorV;
  }

  // Parametric surface (tifxyz segment): rendered as a mesh in the same
  // scene, textured by sampling the shared brick pool in a slab of
  // [-front, +behind] voxels along the surface normal. Missing bricks are
  // requested through the same feedback stream as the raymarcher.
  void SetSurface(std::shared_ptr<const data::TifXyzSurface> s) {
    Surface = std::move(s);
    SurfaceDirty = true;
  }
  void SetSurfaceSlab(float frontVoxels, float behindVoxels) {
    SlabFront = frontVoxels;
    SlabBehind = behindVoxels;
  }
  void SetSurfaceMean(bool mean) { SlabMean = mean; }
  // Multiplier from tifxyz coordinate space to this volume's voxel space
  // (segments are often traced on an older/coarser volume of the scroll).
  void SetSurfaceScale(float s) {
    if (s > 0.f && s != SurfScale) {
      SurfScale = s;
      SurfaceDirty = true;
    }
  }
  void SetShowVolume(bool b) { ShowVolume = b; }
  void SetShowSurface(bool b) { ShowSurface = b; }

  // Shell mask: density outside the mask renders as zero (the "only voxels
  // within X of the surface" view). Passing null textures disables. The
  // shader is a compile-time variant, so toggling rebuilds the program.
  void SetDensityMask(const SurfaceMask* mask);  // null disables
  void SetMaskStyle(float strength) { MaskStrength = strength; }
  void SetMaskIsolate(bool isolate) { MaskIsolate = isolate; }
  // Secondary instances (extra windows) skip cache upkeep and callbacks the
  // primary already performs each frame.
  void SetPrimary(bool p) { Primary = p; }

  // Prediction overlay (e.g. 3D ink detection): a second, co-registered
  // volume with its own brick cache/page tables, sampled at every visible
  // sample and blended in as a hot tint above `threshold`. The overlay
  // volume shares the physical extent of the main volume (it is typically
  // a downsampled prediction pyramid).
  void SetOverlay(GpuBrickCache* cache,
                  const std::array<std::uint64_t, 3>& shapeZyx);
  void SetOverlayStyle(float strength, float threshold) {
    OverlayStrength = strength;
    OverlayThreshold = threshold;
  }

  // Supervoxel clusters (3D SNIC), rendered as lit sphere impostors in
  // render mode 3. World-space centers/radii; value drives the color ramp.
  struct ClusterSphere {
    float x, y, z, radius, value;
  };
  void SetClusters(std::vector<ClusterSphere> cs) {
    Clusters = std::move(cs);
    ClustersDirty = true;
  }

 protected:
  vtkScrollVolumeMapper();
  ~vtkScrollVolumeMapper() override;

 private:
  vtkScrollVolumeMapper(const vtkScrollVolumeMapper&) = delete;
  void operator=(const vtkScrollVolumeMapper&) = delete;

  std::string BuildCommonGlsl() const;
  std::string BuildFragmentShader() const;
  std::string BuildSurfaceVertexShader() const;
  std::string BuildSurfaceFragmentShader() const;
  void EnsureSurfaceMesh();
  void RenderSurface(vtkOpenGLRenderWindow* renWin, vtkRenderer* ren);
  void RenderClusters(vtkOpenGLRenderWindow* renWin, vtkRenderer* ren);

  GpuBrickCache* Cache = nullptr;
  std::function<void()> PreRender;
  std::array<std::uint64_t, 3> ShapeZyx{1, 1, 1};
  double Spacing = 1.0;
  std::uint32_t ChunkDim = 128;

  float DesiredLevel = 0.f;
  float SampleStepScale = 1.f;
  float RenderScale = 1.f;

  // Low-res offscreen target for interaction rendering.
  unsigned int LowResFbo = 0;
  unsigned int LowResTex = 0;
  int LowResW = 0, LowResH = 0;
  std::unique_ptr<vtkOpenGLQuadHelper> UpscaleQuad;
  void EnsureLowResTarget(int w, int h);
  float Window = 1.f;
  float Level = 0.5f;
  float OpacityScale = 0.05f;
  int RenderMode = 0;
  int FilterOp = 0;
  float FilterAmount = 1.f;
  float FilterFloor = 0.f;
  float LightDir[3] = {0.42f, 0.31f, 0.85f};
  float LightAmbient = 1.f;
  bool ShadowsOn = true;
  bool NearestTaps = false;
  bool AppliedNearest = false;  // sampler state currently on the pool

  // Surface pass state. The shader program is owned by VTK's shader cache.
  std::shared_ptr<const data::TifXyzSurface> Surface;
  bool SurfaceDirty = false;
  unsigned int SurfVao = 0, SurfVbo = 0, SurfIbo = 0;
  int SurfIndexCount = 0;
  std::string SurfVsSrc, SurfFsSrc;
  float SlabFront = 8.f, SlabBehind = 8.f;
  float SurfScale = 1.f;
  bool SlabMean = false;
  bool ShowVolume = true, ShowSurface = true;
  bool Primary = true;
  const SurfaceMask* Mask = nullptr;
  float MaskStrength = 0.6f;
  bool MaskIsolate = false;

  std::vector<ClusterSphere> Clusters;
  std::vector<ClusterSphere> ClusterSorted;  // back-to-front order
  double LastSortDop[3] = {0, 0, 0};
  bool ClustersDirty = false;
  unsigned int ClusterVao = 0, ClusterVbo = 0;
  std::string ClusterVsSrc, ClusterFsSrc;

  GpuBrickCache* Overlay = nullptr;
  std::array<std::uint64_t, 3> OverlayShapeZyx{1, 1, 1};
  float OverlayStrength = 0.7f;
  float OverlayThreshold = 0.4f;

  double Bounds[6] = {0, 1, 0, 1, 0, 1};
  std::unique_ptr<vtkOpenGLQuadHelper> Quad;
  std::uint64_t FrameIndex = 0;

  // GL timer query around the raymarch draw (non-blocking, skip-if-busy).
  unsigned int GpuTimerQuery = 0;
  bool GpuTimerInFlight = false;

  // Double-buffered request SSBOs, persistently mapped for zero-stall
  // readback: frame N writes buffer A while the CPU reads buffer B behind
  // its fence.
  static constexpr std::uint32_t kMaxFeedback = 16384;
  unsigned int FeedbackSsbo[2] = {0, 0};
  std::uint32_t* FeedbackMapped[2] = {nullptr, nullptr};
  void* FeedbackFence[2] = {nullptr, nullptr};  // GLsync
  int FeedbackWriteIndex = 0;
  std::vector<data::ChunkCoord> FeedbackRequests;
  void InitFeedbackBuffers();
  void ReadFeedback(int index);
};

}  // namespace sv::render
