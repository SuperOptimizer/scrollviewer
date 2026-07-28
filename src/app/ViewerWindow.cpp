#include "app/ViewerWindow.h"

#include <QEvent>
#include <QMessageBox>
#include <QVTKOpenGLNativeWidget.h>

#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkRenderWindowInteractor.h>

#include <filesystem>
#include <fstream>
#include <future>

#include "core/Log.h"
#include "core/Profiling.h"
#include "store/DiskCacheStore.h"
#include "store/HttpStore.h"
#include "store/LocalStore.h"

namespace sv::app {

namespace {

// Synchronous metadata read through any store (metadata files are tiny).
Result<ByteBuffer> readMetaSync(store::ChunkStore& cs, const std::string& key) {
  std::promise<store::StoreResult> promise;
  auto future = promise.get_future();
  cs.read(key, {}, [&](store::StoreResult r) { promise.set_value(std::move(r)); });
  auto r = future.get();
  if (!r) return std::unexpected(Error{r.error().message.empty()
                                           ? "read failed"
                                           : r.error().message});
  return std::move(*r);
}

}  // namespace

ViewerWindow::ViewerWindow(const std::string& source, QWidget* parent)
    : QMainWindow(parent) {
  vtkWidget_ = new QVTKOpenGLNativeWidget(this);
  setCentralWidget(vtkWidget_);
  resize(1280, 900);

  vtkNew<vtkGenericOpenGLRenderWindow> renWin;
  vtkWidget_->setRenderWindow(renWin);
  renWin->AddRenderer(renderer_);
  renderer_->SetBackground(0.05, 0.05, 0.08);

  vtkNew<vtkInteractorStyleTrackballCamera> style;
  renWin->GetInteractor()->SetInteractorStyle(style);

  // Drop sampling quality during camera drags; restore + rerender on release.
  vtkNew<vtkCallbackCommand> startCb;
  startCb->SetClientData(this);
  startCb->SetCallback([](vtkObject*, unsigned long, void* cd, void*) {
    auto* self = static_cast<ViewerWindow*>(cd);
    self->interacting_ = true;
    if (self->mapper_) {
      self->mapper_->SetSampleStepScale(2.0f);
      self->mapper_->SetRenderScale(0.6f);
    }
  });
  style->AddObserver(vtkCommand::StartInteractionEvent, startCb);

  vtkNew<vtkCallbackCommand> endCb;
  endCb->SetClientData(this);
  endCb->SetCallback([](vtkObject*, unsigned long, void* cd, void*) {
    auto* self = static_cast<ViewerWindow*>(cd);
    self->interacting_ = false;
    if (self->mapper_) {
      self->mapper_->SetSampleStepScale(1.0f);
      self->mapper_->SetRenderScale(1.0f);
    }
    self->streamingActive_ = true;  // replan + full-quality pass
  });
  style->AddObserver(vtkCommand::EndInteractionEvent, endCb);

  if (!openVolume(source)) {
    QMessageBox::critical(this, "scrollviewer",
                          QString("Failed to open volume: %1")
                              .arg(QString::fromStdString(source)));
    return;
  }

  // Replan on camera motion.
  vtkNew<vtkCallbackCommand> camCb;
  camCb->SetClientData(this);
  camCb->SetCallback([](vtkObject*, unsigned long, void* clientData, void*) {
    static_cast<ViewerWindow*>(clientData)->onCameraChanged();
  });
  renderer_->GetActiveCamera()->AddObserver(vtkCommand::ModifiedEvent, camCb);

  // 60 fps target: pump at ~16 ms and coalesce the mouse-move flood (VTK
  // renders once per move event; drops are lossless for a trackball camera
  // since positions are absolute).
  vtkWidget_->installEventFilter(this);
  streamTimer_.setInterval(16);
  connect(&streamTimer_, &QTimer::timeout, this, &ViewerWindow::pumpStreaming);
  streamTimer_.start();

  renderer_->ResetCamera();
  setWindowTitle(QString("scrollviewer — %1").arg(QString::fromStdString(source)));
}

ViewerWindow::~ViewerWindow() {
  // Pipeline before store/pools (teardown contract).
  pipeline_.reset();
}

bool ViewerWindow::eventFilter(QObject* watched, QEvent* event) {
  if (watched == vtkWidget_ && event->type() == QEvent::MouseMove) {
    const auto now = std::chrono::steady_clock::now();
    if (now - lastMouseMove_ < std::chrono::milliseconds(16)) return true;
    lastMouseMove_ = now;
  }
  if (watched == vtkWidget_ && event->type() == QEvent::Wheel) {
    // VTK wraps each wheel click in its own start/end interaction, so the
    // drag-quality path never engages for zooming; treat a wheel burst as
    // an interaction window instead.
    lastWheel_ = std::chrono::steady_clock::now();
    streamingActive_ = true;
  }
  return QMainWindow::eventFilter(watched, event);
}

bool ViewerWindow::openVolume(const std::string& source) {
  ioPool_ = std::make_shared<ThreadPool>("io", 4);

  if (source.starts_with("http://") || source.starts_with("https://")) {
    auto http = std::make_shared<store::HttpStore>(source);
    const auto cacheDir =
        std::filesystem::path(qgetenv("LOCALAPPDATA").toStdString()) /
        "scrollviewer" / "chunk-cache";
    store_ = std::make_shared<store::DiskCacheStore>(http, cacheDir,
                                                     64ull << 30, ioPool_);
  } else {
    store_ = std::make_shared<store::LocalStore>(source, ioPool_);
  }

  auto vol = zarr::OmeZarrVolume::open([this](const std::string& rel) {
    return readMetaSync(*store_, rel);
  });
  if (!vol) {
    logError("open failed: {}", vol.error().message);
    return false;
  }
  volume_ = *vol;

  const auto& l0 = volume_->level(0);
  const std::uint32_t chunkDim = l0.chunks[0];
  const int levels = volume_->levelCount();

  // Host memory budget: the RamCache is the dominant consumer; the upload
  // queue below adds at most kUploadQueueMaxBytes on top.
  ramCache_ = std::make_unique<data::RamCache>(6ull << 30);
  pipeline_ = std::make_unique<data::ChunkFetchPipeline>(
      store_, volume_, volumeId_, *ramCache_);
  assembler_ =
      std::make_unique<render::BrickAssembler>(volume_, volumeId_, *ramCache_);
  uploader_ = std::make_unique<render::GpuUploader>(*assembler_, 2);

  std::vector<render::GpuBrickCache::LevelDims> dims;
  for (int i = 0; i < levels; ++i) {
    const auto g = volume_->level(i).chunkGridDims();
    dims.push_back({g});
  }
  gpuCache_ = std::make_unique<render::GpuBrickCache>(
      std::move(dims), chunkDim + 2, 4ull << 30);

  // Optional manifest sidecar: instant empty-space knowledge for the whole
  // pyramid (no fetches, no 404s) + planner pruning.
  if (auto mfBytes = readMetaSync(*store_, "manifest.svmf")) {
    if (auto mf = data::VolumeManifest::parse(mfBytes->span())) {
      manifest_ = *mf;
      const int mfLevels = std::min(levels, manifest_->levelCount());
      for (int li = 0; li < mfLevels; ++li) {
        gpuCache_->seedEmpties(li, [&](std::uint32_t z, std::uint32_t y,
                                       std::uint32_t x) {
          return manifest_->empty(li, z, y, x);
        });
      }
      logInfo("manifest: seeded {} levels; {} page tiles materialized",
              mfLevels, gpuCache_->tilesAllocated());
    } else {
      logWarn("manifest.svmf present but unreadable: {}", mf.error().message);
    }
  }

  planner_ = std::make_unique<render::WorkingSetPlanner>(
      l0.shape, /*spacing=*/1.0, chunkDim, levels, manifest_);

  mapper_ = vtkSmartPointer<render::vtkScrollVolumeMapper>::New();
  mapper_->SetBrickCache(gpuCache_.get());
  mapper_->SetVolumeExtent(l0.shape, 1.0);
  mapper_->SetChunkDim(chunkDim);
  mapper_->SetPreRenderCallback([this] { preRenderUpload(); });
  volumeActor_->SetMapper(mapper_);
  renderer_->AddVolume(volumeActor_);

  borderedScratch_.resize(assembler_->borderedBytes());
  streamingActive_ = true;
  return true;
}

void ViewerWindow::onCameraChanged() {
  streamingActive_ = true;  // pumpStreaming replans on the next tick
}

void ViewerWindow::pumpStreaming() {
  if (!streamingActive_ || !mapper_ || !gpuCache_) return;
  ++pumpCounter_;

  // Replanning is only needed when the camera moved; with a static camera a
  // slower cadence is plenty (the traversal costs milliseconds at deep
  // zoom). Off-ticks still render to drain decoded bricks to the GPU.
  vtkCamera* cam = renderer_->GetActiveCamera();
  const bool camChanged = cam->GetMTime() != lastCamMTime_;
  lastCamMTime_ = cam->GetMTime();
  const bool wheelBurst = std::chrono::steady_clock::now() - lastWheel_ <
                          std::chrono::milliseconds(300);
  const bool motion = interacting_ || wheelBurst;
  // Static camera: replan rarely. Moving camera at deep zoom: the traversal
  // costs multiple ms, so even then cap it at every 4th tick (ray feedback
  // still streams exact requests every frame in between).
  const int replanStride = camChanged ? (motion ? 4 : 1) : 8;
  if ((pumpCounter_ % replanStride) != 0) {
    const auto s = pipeline_->stats();
    if (s.ready > 0 || uploader_->pendingCount() > 0 ||
        !uploadQueue_.empty())
      vtkWidget_->renderWindow()->Render();
    return;
  }

  // Planning needs residency info only; rendering (and uploads) happen in
  // the mapper's PreRender callback with the GL context current.
  const int vh = std::max(1, vtkWidget_->height());
  render::WorkingSetPlanner::Plan plan;
  {
    ProfScope timePlan(profStages().plan);
    plan = planner_->plan(renderer_, volumeId_, vh, *gpuCache_);
  }

  for (const auto& key : plan.visible) gpuCache_->touch(key);

  // Ray-guided streaming owns the finest level: planner guesses at
  // plan.desiredLevel are speculative (no occlusion knowledge), so drop them
  // and let the rays request exactly what they touch. Coarser levels keep
  // planner prefetch for guaranteed coverage.
  if (plan.desiredLevel < volume_->levelCount() - 1) {
    std::erase_if(plan.missing, [&](const data::BrickRequest& r) {
      return r.key.coord.level == plan.desiredLevel;
    });
  }

  // Ray-guided requests: bricks actual rays needed last frame. They are
  // exact (visible, unoccluded), so they outrank same-level planner guesses.
  // Manifest-known empties are filtered (belt and braces; they are seeded
  // EMPTY at open so rays normally never ask).
  for (const auto& coord : mapper_->TakeFeedbackRequests()) {
    if (manifest_ && manifest_->hasLevel(coord.level) &&
        manifest_->empty(coord.level, coord.z, coord.y, coord.x))
      continue;
    plan.missing.push_back(data::BrickRequest{
        data::BrickKey{volumeId_, coord},
        1000.f * float(coord.level) + 500.f});
  }

  // Requests already decoded in RAM re-feed the upload path; the rest go to
  // the fetch pipeline. The re-feed queue holds shared_ptrs only briefly
  // (drained every frame into the bounded PBO ring).
  std::vector<data::BrickRequest> wanted;
  wanted.reserve(plan.missing.size());
  for (const auto& r : plan.missing) {
    if (auto brick = ramCache_->get(r.key)) {
      if (queuedKeys_.insert(r.key.packed()).second)
        uploadQueue_.push_back(std::move(brick));
    } else {
      wanted.push_back(r);
    }
  }

  // One level coarser while dragging: ~8x fewer voxels to march.
  mapper_->SetDesiredLevel(float(plan.desiredLevel + (interacting_ ? 1 : 0)));
  pipeline_->submit(wanted);

  const auto stats = pipeline_->stats();
  const bool anythingPending = !wanted.empty() || stats.inflightFetches > 0 ||
                               stats.queuedDecodes > 0 || stats.ready > 0 ||
                               !uploadQueue_.empty() ||
                               uploader_->pendingCount() > 0;

  // Progressive quality: reduced resolution during camera motion (drag or
  // wheel burst) and while bricks stream; one full-resolution frame lands
  // at convergence below.
  mapper_->SetRenderScale(motion ? 0.6f : (anythingPending ? 0.75f : 1.0f));

  // Periodic streaming + profiling diagnostics (~every 2 s while active).
  if (pumpCounter_ % 120 == 0) {
    logInfo(
        "stream: desiredL={} visible={} missing={} wanted={} queued={} "
        "inflight={} decoding={} ready={} uploadQ={} gpuResident={} ramMB={}",
        plan.desiredLevel, plan.visible.size(), plan.missing.size(),
        wanted.size(), stats.queued, stats.inflightFetches,
        stats.queuedDecodes, stats.ready, uploadQueue_.size(),
        gpuCache_->residentCount(), ramCache_->bytesUsed() >> 20);
    auto& ps = profStages();
    const auto fetch = ps.fetch.take(), decode = ps.decode.take(),
               assemble = ps.assemble.take(), planT = ps.plan.take(),
               upload = ps.upload.take(), frame = ps.frame.take(),
               gpu = ps.gpuRaymarch.take();
    logInfo(
        "prof: gpuRay={:.2f}ms frame={:.2f}ms plan={:.2f}ms "
        "upload={:.2f}ms/frame | fetch={:.1f}ms decode={:.2f}ms "
        "assemble={:.2f}ms per-brick (n={}/{}/{})",
        gpu.avgMs(), frame.avgMs(), planT.avgMs(), upload.avgMs(),
        fetch.avgMs(), decode.avgMs(), assemble.avgMs(), fetch.count,
        decode.count, assemble.count);
  }
  if (anythingPending || !plan.missing.empty()) {
    vtkWidget_->renderWindow()->Render();
  } else {
    // Converged: one final full-resolution frame, then go idle.
    mapper_->SetRenderScale(1.0f);
    vtkWidget_->renderWindow()->Render();
    streamingActive_ = false;  // camera motion reactivates
  }
}

void ViewerWindow::preRenderUpload() {
  // GL context is current (called from mapper Render).
  constexpr int kUploadBudget = 32;  // async PBO path: cheap per brick

  if (!uploader_->initialized()) uploader_->initialize(96);

  // Route decoded bricks: fill bricks become free EMPTY marks immediately;
  // real bricks go to the assembly workers + PBO ring. The ring/queue bound
  // host memory; a rejected brick stays in the RamCache and is re-planned.
  for (auto& rb : pipeline_->drainReady()) {
    if (rb.brick->isFillValue) {
      gpuCache_->markEmpty(rb.brick->key);
    } else if (queuedKeys_.insert(rb.brick->key.packed()).second) {
      const bool pinned =
          rb.brick->key.coord.level >= volume_->levelCount() - 2;
      if (!uploader_->enqueue(rb.brick, pinned))
        queuedKeys_.erase(rb.brick->key.packed());
    }
    pipeline_->unpin(rb.brick->key);
  }

  // Re-feed bricks that are decoded in RAM but not yet GPU-resident (e.g.
  // shed earlier or evicted): pumpStreaming enqueues them via uploadQueue_.
  while (!uploadQueue_.empty()) {
    auto brick = std::move(uploadQueue_.front());
    uploadQueue_.pop_front();
    const auto packed = brick->key.packed();
    if (brick->isFillValue) {
      gpuCache_->markEmpty(brick->key);
      queuedKeys_.erase(packed);
      continue;
    }
    const bool pinned = brick->key.coord.level >= volume_->levelCount() - 2;
    if (!uploader_->enqueue(std::move(brick), pinned)) {
      queuedKeys_.erase(packed);
      break;  // ring saturated; try again next frame
    }
  }

  const int uploaded = uploader_->drainToGpu(*gpuCache_, kUploadBudget);
  (void)uploaded;
  // Uploaded keys leave the dedup set lazily: once resident, the planner
  // stops listing them and stale entries get erased on the re-feed path.
  if (uploader_->pendingCount() == 0) queuedKeys_.clear();
}

}  // namespace sv::app
