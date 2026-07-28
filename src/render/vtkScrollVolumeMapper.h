#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>

#include <vtkVolumeMapper.h>

#include "render/GpuBrickCache.h"

class vtkOpenGLQuadHelper;

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

 protected:
  vtkScrollVolumeMapper();
  ~vtkScrollVolumeMapper() override;

 private:
  vtkScrollVolumeMapper(const vtkScrollVolumeMapper&) = delete;
  void operator=(const vtkScrollVolumeMapper&) = delete;

  std::string BuildFragmentShader() const;

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
