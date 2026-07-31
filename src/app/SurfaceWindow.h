#pragma once

#include <QMainWindow>

#include <functional>

#include <vtkNew.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkVolume.h>

#include "render/vtkScrollVolumeMapper.h"

class QVTKOpenGLNativeWidget;
class QSlider;

namespace sv::app {

// Companion window: the same out-of-core volume raycaster as the main view,
// but with a shell mask applied — every voxel further than the configured
// front/behind distance from the tifxyz surface renders as zero. Shares the
// GPU brick cache with the main window through the Qt GL share group.
class SurfaceWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit SurfaceWindow(QWidget* parent = nullptr);

  render::vtkScrollVolumeMapper* mapper() { return mapper_; }
  vtkRenderer* renderer() { return renderer_; }
  void resetCamera();
  void requestRender();

  float slabFront() const;
  float slabBehind() const;

  // Fired after the slab sliders settle so the owner can rebuild the mask.
  std::function<void()> onSlabChanged;

 private:
  QVTKOpenGLNativeWidget* widget_ = nullptr;
  vtkNew<vtkRenderer> renderer_;
  vtkNew<vtkVolume> actor_;
  vtkSmartPointer<render::vtkScrollVolumeMapper> mapper_;
  QSlider* frontSlider_ = nullptr;
  QSlider* behindSlider_ = nullptr;
};

}  // namespace sv::app
