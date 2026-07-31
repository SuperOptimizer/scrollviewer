#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include <vtkVolumeMapper.h>

#include "data/TifXyzSurface.h"
#include "render/GpuBrickCache.h"

class vtkOpenGLQuadHelper;

namespace sv::render {

// Flattened-surface volume raycaster: renders a tifxyz segment as a box of
// (W*gridStep) x (H*gridStep) x (front+behind) native voxels. Each sample at
// flat coords (u, v, w) warps to volume space P(u,v) + w*N(u,v) via small
// position/normal textures, then samples the SHARED brick pool through the
// sparse page tables with coarse fallback — so the flattened view streams
// and refines exactly like the main view. Runs in its own window/context;
// textures are shared through the Qt share group.
class vtkScrollSurfaceMapper : public vtkVolumeMapper {
 public:
  static vtkScrollSurfaceMapper* New();
  vtkTypeMacro(vtkScrollSurfaceMapper, vtkVolumeMapper);

  void Render(vtkRenderer* ren, vtkVolume* vol) override;
  double* GetBounds() override;
  void ReleaseGraphicsResources(vtkWindow* window) override;

 protected:
  int FillInputPortInformation(int port, vtkInformation* info) override;

 public:
  void SetBrickCache(GpuBrickCache* cache) { Cache = cache; }
  void SetVolumeExtent(const std::array<std::uint64_t, 3>& shapeZyx,
                       double spacing);
  void SetChunkDim(std::uint32_t dim) { ChunkDim = dim; }

  // gridStep: native voxels per tifxyz grid texel (1 / meta.json scale).
  // surfScale: tifxyz coordinate space -> this volume's voxel space.
  void SetSurface(std::shared_ptr<const data::TifXyzSurface> s,
                  float gridStep, float surfScale);
  void SetSlab(float frontVoxels, float behindVoxels);
  void SetWindowLevel(float window, float level) {
    Window = window;
    Level = level;
  }
  void SetOpacityScale(float s) { OpacityScale = s; }
  void SetDesiredLevel(float lod) { DesiredLevel = lod; }
  void SetSampleStepScale(float s) { SampleStepScale = s; }

 protected:
  vtkScrollSurfaceMapper();
  ~vtkScrollSurfaceMapper() override;

 private:
  vtkScrollSurfaceMapper(const vtkScrollSurfaceMapper&) = delete;
  void operator=(const vtkScrollSurfaceMapper&) = delete;

  std::string BuildFragmentShader() const;
  void EnsureSurfaceTextures();
  void UpdateBounds();

  GpuBrickCache* Cache = nullptr;
  std::array<std::uint64_t, 3> ShapeZyx{1, 1, 1};
  double Spacing = 1.0;
  std::uint32_t ChunkDim = 128;

  std::shared_ptr<const data::TifXyzSurface> Surface;
  bool SurfaceDirty = false;
  float GridStep = 20.f;
  float SurfScale = 1.f;
  float SlabFront = 8.f, SlabBehind = 8.f;
  float Window = 1.f, Level = 0.5f;
  float OpacityScale = 0.05f;
  float DesiredLevel = 0.f;
  float SampleStepScale = 1.f;

  unsigned int PosTex = 0, NrmTex = 0;  // RGBA32F / RGB32F, W x H
  double Bounds[6] = {0, 1, 0, 1, 0, 1};
  std::unique_ptr<vtkOpenGLQuadHelper> Quad;
  std::uint64_t FrameIndex = 0;
};

}  // namespace sv::render
