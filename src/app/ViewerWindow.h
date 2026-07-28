#pragma once

#include <QMainWindow>
#include <QTimer>

#include <chrono>
#include <deque>
#include <memory>
#include <string>
#include <unordered_set>

#include <vtkNew.h>
#include <vtkRenderer.h>
#include <vtkVolume.h>

#include "data/ChunkFetchPipeline.h"
#include "data/RamCache.h"
#include "data/VolumeManifest.h"
#include "render/BrickAssembler.h"
#include "render/GpuBrickCache.h"
#include "render/GpuUploader.h"
#include "render/WorkingSetPlanner.h"
#include "render/vtkScrollVolumeMapper.h"
#include "store/ChunkStore.h"
#include "zarr/OmeZarrVolume.h"

class QVTKOpenGLNativeWidget;
class QSlider;

namespace sv::app {

// Main window: one 3D progressive volume view (milestone 5/6). Owns the
// whole data stack for the opened volume.
class ViewerWindow : public QMainWindow {
  Q_OBJECT

 public:
  // source: local zarr directory or http(s) URL of a zarr root.
  explicit ViewerWindow(const std::string& source, QWidget* parent = nullptr);
  ~ViewerWindow() override;

 protected:
  bool eventFilter(QObject* watched, QEvent* event) override;

 private:
  bool openVolume(const std::string& source);
  void buildDisplayToolbar();
  void applyDisplaySettings();
  void onCameraChanged();
  void pumpStreaming();   // timer tick: replan + schedule renders while loading
  void preRenderUpload(); // mapper callback, GL context current

  QVTKOpenGLNativeWidget* vtkWidget_ = nullptr;
  QTimer streamTimer_;
  QSlider* levelSlider_ = nullptr;
  QSlider* windowSlider_ = nullptr;
  QSlider* opacitySlider_ = nullptr;

  vtkNew<vtkRenderer> renderer_;
  vtkNew<vtkVolume> volumeActor_;
  vtkSmartPointer<render::vtkScrollVolumeMapper> mapper_;

  std::shared_ptr<ThreadPool> ioPool_;
  std::shared_ptr<store::ChunkStore> store_;
  std::shared_ptr<zarr::OmeZarrVolume> volume_;
  std::shared_ptr<data::VolumeManifest> manifest_;
  std::unique_ptr<data::RamCache> ramCache_;
  std::unique_ptr<data::ChunkFetchPipeline> pipeline_;
  std::unique_ptr<render::BrickAssembler> assembler_;
  std::unique_ptr<render::GpuUploader> uploader_;
  std::unique_ptr<render::GpuBrickCache> gpuCache_;
  std::unique_ptr<render::WorkingSetPlanner> planner_;

  std::vector<std::uint8_t> borderedScratch_;
  // Decoded bricks awaiting GPU upload (budgeted per frame). Keys mirror the
  // deque for dedup.
  std::deque<std::shared_ptr<const data::Brick>> uploadQueue_;
  std::unordered_set<std::uint64_t> queuedKeys_;
  std::uint64_t uploadQueueBytes_ = 0;
  std::uint32_t volumeId_ = 1;
  std::uint64_t pumpCounter_ = 0;
  bool streamingActive_ = false;
  bool interacting_ = false;
  std::chrono::steady_clock::time_point lastMouseMove_{};
  std::chrono::steady_clock::time_point lastWheel_{};
  std::uint64_t lastCamMTime_ = 0;
};

}  // namespace sv::app
