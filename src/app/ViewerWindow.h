#pragma once

#include <QMainWindow>
#include <QTimer>

#include <atomic>
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
#include "render/SurfaceMask.h"
#include "render/GpuUploader.h"
#include "render/WorkingSetPlanner.h"
#include "render/vtkScrollVolumeMapper.h"
#include "store/ChunkStore.h"
#include "zarr/OmeZarrVolume.h"

class QVTKOpenGLNativeWidget;
class QSlider;
class QCheckBox;
class QComboBox;
class QPushButton;

namespace sv::app {

class SurfaceWindow;

// Main window: one 3D progressive volume view (milestone 5/6). Owns the
// whole data stack for the opened volume.
class ViewerWindow : public QMainWindow {
  Q_OBJECT

 public:
  // source: local zarr directory or http(s) URL of a zarr root.
  // surfaceSource: optional tifxyz directory or URL prefix (x/y/z.tif).
  // overlaySource: optional co-registered prediction zarr (ink detection
  // etc.), typically a downsampled pyramid sharing the volume's space.
  explicit ViewerWindow(const std::string& source,
                        const std::string& surfaceSource = {},
                        double surfaceScale = 1.0,
                        const std::string& overlaySource = {},
                        QWidget* parent = nullptr);
  ~ViewerWindow() override;

 protected:
  bool eventFilter(QObject* watched, QEvent* event) override;

 private:
  bool openVolume(const std::string& source);
  bool loadOverlay(const std::string& overlaySource);
  void loadSurface(const std::string& surfaceSource, double surfaceScale);
  void rebuildSurfaceMask();
  void updateSurfaceStreaming();
  void buildDisplayToolbar();
  void applyDisplaySettings();
  void runSegmentation();
  void injectTestClusters();
  void captureShot(const QString& path);
  void startBenchmark(const QString& csvPath);
  void benchTick();
  void benchFinishSegment();
  void benchWriteReport();
  void onCameraChanged();
  void pumpStreaming();   // timer tick: replan + schedule renders while loading
  void preRenderUpload(); // mapper callback, GL context current

  QVTKOpenGLNativeWidget* vtkWidget_ = nullptr;
  QTimer streamTimer_;
  QComboBox* modeCombo_ = nullptr;
  QComboBox* filterCombo_ = nullptr;
  QSlider* filterAmtSlider_ = nullptr;
  QSlider* filterFloorSlider_ = nullptr;
  QSlider* lightAzSlider_ = nullptr;
  QSlider* lightElSlider_ = nullptr;
  QSlider* lightAmbSlider_ = nullptr;
  QCheckBox* shadowsCheck_ = nullptr;
  QCheckBox* nearestCheck_ = nullptr;
  QSlider* seedsSlider_ = nullptr;
  QPushButton* segButton_ = nullptr;
  std::atomic<bool> segRunning_{false};
  bool benchActive_ = false;
  QString benchCsv_;
  QTimer* benchTimer_ = nullptr;
  int benchSeg_ = -1;
  int benchFrame_ = 0;
  std::vector<double> benchCpu_, benchGpu_;
  std::vector<std::string> benchRows_, benchSummary_;
  QSlider* levelSlider_ = nullptr;
  QSlider* windowSlider_ = nullptr;
  QSlider* opacitySlider_ = nullptr;
  QSlider* slabFrontSlider_ = nullptr;
  QSlider* slabBehindSlider_ = nullptr;
  QSlider* surfStrengthSlider_ = nullptr;
  QCheckBox* showVolumeCheck_ = nullptr;
  QCheckBox* showSurfaceCheck_ = nullptr;
  bool hasSurface_ = false;

  // Prediction overlay: an independent streaming stack (its own store,
  // pipeline, GPU cache and planner) for a co-registered second volume.
  std::shared_ptr<store::ChunkStore> ovStore_;
  std::shared_ptr<zarr::OmeZarrVolume> ovVolume_;
  std::unique_ptr<data::ChunkFetchPipeline> ovPipeline_;
  std::unique_ptr<render::BrickAssembler> ovAssembler_;
  std::unique_ptr<render::GpuUploader> ovUploader_;
  std::unique_ptr<render::GpuBrickCache> ovGpuCache_;
  std::unique_ptr<render::WorkingSetPlanner> ovPlanner_;
  QSlider* inkStrengthSlider_ = nullptr;
  QSlider* inkThresholdSlider_ = nullptr;
  std::uint32_t ovVolumeId_ = 2;

  std::unique_ptr<SurfaceWindow> surfaceWindow_;
  std::shared_ptr<data::TifXyzSurface> surface_;
  std::vector<data::BrickKey> surfaceBricks_;
  render::SurfaceMask surfaceMask_;
  bool maskNeedsUpload_ = false;

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
