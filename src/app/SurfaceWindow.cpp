#include "app/SurfaceWindow.h"

#include <QLabel>
#include <QSlider>
#include <QToolBar>
#include <QVTKOpenGLNativeWidget.h>

#include <vtkCamera.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkRenderWindowInteractor.h>

namespace sv::app {

SurfaceWindow::SurfaceWindow(QWidget* parent) : QMainWindow(parent) {
  widget_ = new QVTKOpenGLNativeWidget(this);
  setCentralWidget(widget_);
  resize(1100, 800);
  setWindowTitle("scrollviewer — surface shell");

  vtkNew<vtkGenericOpenGLRenderWindow> renWin;
  widget_->setRenderWindow(renWin);
  renWin->AddRenderer(renderer_);
  renderer_->SetBackground(0.05, 0.05, 0.08);
  vtkNew<vtkInteractorStyleTrackballCamera> style;
  renWin->GetInteractor()->SetInteractorStyle(style);

  mapper_ = vtkSmartPointer<render::vtkScrollVolumeMapper>::New();
  mapper_->SetPrimary(false);
  actor_->SetMapper(mapper_);
  renderer_->AddVolume(actor_);

  auto* tb = addToolBar("Slab");
  tb->setMovable(false);
  auto addSlider = [&](const QString& name, int min, int max, int value) {
    tb->addWidget(new QLabel(QString("  %1 ").arg(name)));
    auto* s = new QSlider(Qt::Horizontal);
    s->setRange(min, max);
    s->setValue(value);
    s->setFixedWidth(150);
    tb->addWidget(s);
    // Mask rebuilds cost ~a second: fire on release, not every tick.
    connect(s, &QSlider::sliderReleased, this, [this] {
      if (onSlabChanged) onSlabChanged();
    });
    connect(s, &QSlider::valueChanged, this, [this](int) {
      if (!frontSlider_ || !behindSlider_) return;
    });
    return s;
  };
  // Slab in native voxels. At ~2.4 um/voxel the papyrus sheet itself is
  // ~80+ voxels thick: the defaults must contain it, or the shell view
  // degenerates into a paper-thin lamina that reads as a textured sheet.
  frontSlider_ = addSlider("front", 0, 256, 64);
  behindSlider_ = addSlider("behind", 0, 256, 64);
}

float SurfaceWindow::slabFront() const { return float(frontSlider_->value()); }
float SurfaceWindow::slabBehind() const {
  return float(behindSlider_->value());
}

void SurfaceWindow::resetCamera() {
  renderer_->ResetCamera();
  requestRender();
}

void SurfaceWindow::requestRender() { widget_->renderWindow()->Render(); }

}  // namespace sv::app
