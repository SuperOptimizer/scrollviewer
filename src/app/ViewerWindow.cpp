#include "app/ViewerWindow.h"

#include <QCheckBox>
#include <QComboBox>
#include <QEvent>
#include <QLabel>
#include <QMessageBox>
#include <QApplication>
#include <QPushButton>
#include <QSlider>
#include <QToolBar>
#include <QVTKOpenGLNativeWidget.h>

#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkPNGWriter.h>
#include <vtkWindowToImageFilter.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <thread>

#include "data/Snic3D.h"

#include "app/SurfaceWindow.h"

#include "core/Log.h"
#include "core/Profiling.h"
#include "store/DiskCacheStore.h"
#include "store/GdctTranscoder.h"
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

// Per-volume disk-cache directory. Chunk keys are volume-relative
// ("<level>/z/y/x"), so each remote source needs its own namespace or two
// volumes would collide in one tree. FNV-1a of the URL, in hex.
std::filesystem::path cacheDirFor(const char* kind, const std::string& source) {
  std::uint64_t h = 14695981039346656037ull;
  for (const char c : source) h = (h ^ std::uint8_t(c)) * 1099511628211ull;
  return std::filesystem::path(qgetenv("LOCALAPPDATA").toStdString()) /
         "scrollviewer" / kind / std::format("{:016x}", h);
}

}  // namespace

ViewerWindow::ViewerWindow(const std::string& source,
                           const std::string& surfaceSource,
                           double surfaceScale,
                           const std::string& overlaySource, QWidget* parent)
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

  if (!overlaySource.empty()) loadOverlay(overlaySource);
  if (!surfaceSource.empty()) loadSurface(surfaceSource, surfaceScale);
  buildDisplayToolbar();

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

  // Automated rendering-test hooks (env-driven, used by the test harness).
  if (const int m = qEnvironmentVariableIntValue("SCROLLVIEWER_MODE"); m > 0)
    modeCombo_->setCurrentIndex(m);
  if (qgetenv("SCROLLVIEWER_NN") == "1") nearestCheck_->setChecked(true);

  if (qgetenv("SCROLLVIEWER_TEST_CLUSTERS") == "1") injectTestClusters();
  if (qgetenv("SCROLLVIEWER_AUTOSEG") == "1")
    QTimer::singleShot(2500, this, &ViewerWindow::runSegmentation);
  if (const QString bench = qEnvironmentVariable("SCROLLVIEWER_BENCH");
      !bench.isEmpty())
    QTimer::singleShot(8000, this, [this, bench] { startBenchmark(bench); });
  if (const QString shot = qEnvironmentVariable("SCROLLVIEWER_SHOT");
      !shot.isEmpty()) {
    int ms = qEnvironmentVariableIntValue("SCROLLVIEWER_SHOT_MS");
    if (ms <= 0) ms = 6000;
    QTimer::singleShot(ms, this, [this, shot] { captureShot(shot); });
  }
}

// Automated rendering benchmark: scripted camera choreography across render
// modes/features, per-frame CPU (Render wall time) + GPU (raymarch timer
// query) samples, CSV dump + summary table, then exit. Driven by
// SCROLLVIEWER_BENCH=<csv path>.
namespace {
struct BenchSegment {
  const char* name;
  int mode;       // modeCombo index
  int filterOp;   // filterCombo index
  bool shadows;
  int frames;
  enum Motion { Orbit, ZoomIn, DeepOrbit } motion;
};
constexpr BenchSegment kBenchSegments[] = {
    {"orbit-xray", 0, 0, false, 180, BenchSegment::Orbit},
    {"zoom-xray", 0, 0, false, 180, BenchSegment::ZoomIn},
    {"deep-orbit-xray", 0, 0, false, 120, BenchSegment::DeepOrbit},
    {"orbit-shaded", 1, 0, false, 120, BenchSegment::Orbit},
    {"orbit-shaded-sharpen", 1, 2, false, 120, BenchSegment::Orbit},
    {"orbit-shaded-smooth", 1, 1, false, 120, BenchSegment::Orbit},
    {"orbit-surface-shadows", 2, 0, true, 120, BenchSegment::Orbit},
    {"orbit-surface-noshadow", 2, 0, false, 120, BenchSegment::Orbit},
    {"orbit-edges", 1, 3, false, 120, BenchSegment::Orbit},
    {"orbit-clusters", 3, 0, false, 120, BenchSegment::Orbit},
};
}  // namespace

void ViewerWindow::startBenchmark(const QString& csvPath) {
  benchActive_ = true;
  benchCsv_ = csvPath;
  benchSeg_ = -1;
  logInfo("bench: starting ({} segments)", std::size(kBenchSegments));
  runSegmentation();  // clusters ready by the time the last segment runs
  benchTimer_ = new QTimer(this);
  benchTimer_->setInterval(0);  // as fast as frames complete
  connect(benchTimer_, &QTimer::timeout, this, &ViewerWindow::benchTick);
  benchTimer_->start();
}

void ViewerWindow::benchTick() {
  auto* cam = renderer_->GetActiveCamera();
  const bool segDone =
      benchSeg_ < 0 ||
      benchFrame_ >= kBenchSegments[benchSeg_].frames;
  if (segDone) {
    if (benchSeg_ >= 0) benchFinishSegment();
    ++benchSeg_;
    if (benchSeg_ >= int(std::size(kBenchSegments))) {
      benchWriteReport();
      QApplication::quit();
      return;
    }
    const auto& seg = kBenchSegments[benchSeg_];
    benchFrame_ = 0;
    benchCpu_.clear();
    benchGpu_.clear();
    modeCombo_->setCurrentIndex(seg.mode);
    filterCombo_->setCurrentIndex(seg.filterOp);
    shadowsCheck_->setChecked(seg.shadows);
    renderer_->ResetCamera();
    if (seg.motion == BenchSegment::DeepOrbit) {
      cam->Dolly(9.0);
      renderer_->ResetCameraClippingRange();
    }
    vtkWidget_->renderWindow()->Render();  // settle new state
    profStages().gpuRaymarch.take();       // reset accumulators
    profStages().frame.take();
    logInfo("bench: segment {} ({})", benchSeg_, seg.name);
    return;
  }

  const auto& seg = kBenchSegments[benchSeg_];
  switch (seg.motion) {
    case BenchSegment::Orbit:
    case BenchSegment::DeepOrbit:
      cam->Azimuth(360.0 / seg.frames);
      if (benchFrame_ % 3 == 0) cam->Elevation(0.2);
      break;
    case BenchSegment::ZoomIn:
      cam->Dolly(std::pow(14.0, 1.0 / seg.frames));
      break;
  }
  renderer_->ResetCameraClippingRange();

  const auto t0 = std::chrono::steady_clock::now();
  vtkWidget_->renderWindow()->Render();
  const double cpuMs =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0)
          .count();
  benchCpu_.push_back(cpuMs);
  const auto gpu = profStages().gpuRaymarch.take();
  if (gpu.count > 0) benchGpu_.push_back(gpu.avgMs());
  benchRows_.push_back(std::format("{},{},{:.3f},{:.3f}", seg.name,
                                   benchFrame_, cpuMs,
                                   gpu.count ? gpu.avgMs() : -1.0));
  ++benchFrame_;
}

namespace {
struct BenchStats {
  double mean, p50, p95, mx;
};
BenchStats benchStatsOf(std::vector<double> v) {
  if (v.empty()) return {0, 0, 0, 0};
  std::ranges::sort(v);
  double sum = 0;
  for (double x : v) sum += x;
  return {sum / double(v.size()), v[v.size() / 2],
          v[std::min(v.size() - 1, v.size() * 95 / 100)], v.back()};
}
}  // namespace

void ViewerWindow::benchFinishSegment() {
  const auto& seg = kBenchSegments[benchSeg_];
  const auto c = benchStatsOf(benchCpu_);
  const auto g = benchStatsOf(benchGpu_);
  benchSummary_.push_back(std::format(
      "{:24s} cpu mean {:6.2f} p50 {:6.2f} p95 {:6.2f} max {:7.2f} | "
      "gpu mean {:6.2f} p50 {:6.2f} p95 {:6.2f} max {:7.2f} (n={})",
      seg.name, c.mean, c.p50, c.p95, c.mx, g.mean, g.p50, g.p95, g.mx,
      benchCpu_.size()));
  logInfo("bench: {}", benchSummary_.back());
}

void ViewerWindow::benchWriteReport() {
  std::ofstream f(benchCsv_.toStdString());
  f << "segment,frame,cpu_ms,gpu_ms\n";
  for (const auto& r : benchRows_) f << r << "\n";
  logInfo("bench: === SUMMARY ===");
  for (const auto& s : benchSummary_) logInfo("bench: {}", s);
  logInfo("bench: csv written to {}", benchCsv_.toStdString());
  benchActive_ = false;
}

// Synthetic cluster pattern with known world positions: corner spheres, a
// center sphere, and one sphere row along each axis (x bright, y mid,
// z dark). If these render anywhere but the volume's corners/center/axes,
// the sphere pass — not the clustering — is at fault.
void ViewerWindow::injectTestClusters() {
  if (!volume_ || !mapper_) return;
  const auto& sh = volume_->level(0).shape;  // z,y,x
  const float wx = float(sh[2]), wy = float(sh[1]), wz = float(sh[0]);
  const float r = 0.03f * std::min({wx, wy, wz});
  std::vector<render::vtkScrollVolumeMapper::ClusterSphere> cs;
  for (const float fx : {0.f, 1.f})
    for (const float fy : {0.f, 1.f})
      for (const float fz : {0.f, 1.f})
        cs.push_back({fx * wx, fy * wy, fz * wz, r, 1.0f});
  cs.push_back({0.5f * wx, 0.5f * wy, 0.5f * wz, 2.f * r, 0.7f});
  for (int i = 1; i <= 8; ++i) {
    const float t = float(i) / 9.f;
    cs.push_back({t * wx, 0.5f * wy, 0.5f * wz, r * 0.6f, 0.95f});  // x row
    cs.push_back({0.5f * wx, t * wy, 0.5f * wz, r * 0.6f, 0.5f});   // y row
    cs.push_back({0.5f * wx, 0.5f * wy, t * wz, r * 0.6f, 0.12f});  // z row
  }
  mapper_->SetClusters(std::move(cs));
  if (modeCombo_) modeCombo_->setCurrentIndex(3);
  logInfo("test clusters injected: volume world {} x {} x {}", wx, wy, wz);
}

void ViewerWindow::captureShot(const QString& path) {
  auto* rw = vtkWidget_->renderWindow();
  rw->Render();
  vtkNew<vtkWindowToImageFilter> grab;
  grab->SetInput(rw);
  grab->ReadFrontBufferOff();
  grab->Update();
  vtkNew<vtkPNGWriter> png;
  png->SetFileName(path.toStdString().c_str());
  png->SetInputConnection(grab->GetOutputPort());
  png->Write();
  logInfo("screenshot written to {}", path.toStdString());
}

ViewerWindow::~ViewerWindow() {
  // Pipelines before stores/pools (teardown contract).
  ovPipeline_.reset();
  pipeline_.reset();
}

void ViewerWindow::loadSurface(const std::string& surfaceSource,
                               double surfaceScale) {
  // A tifxyz segment is tiny (a few MB): fetch its files through a
  // throwaway store so local paths and URL prefixes both work.
  std::shared_ptr<store::ChunkStore> ss;
  if (surfaceSource.starts_with("http://") ||
      surfaceSource.starts_with("https://"))
    ss = std::make_shared<store::HttpStore>(surfaceSource);
  else
    ss = std::make_shared<store::LocalStore>(surfaceSource, ioPool_);

  auto x = readMetaSync(*ss, "x.tif");
  auto y = readMetaSync(*ss, "y.tif");
  auto z = readMetaSync(*ss, "z.tif");
  if (!x || !y || !z) {
    logError("surface: failed to fetch tifxyz from {}", surfaceSource);
    return;
  }
  auto surf = data::TifXyzSurface::load(x->span(), y->span(), z->span());
  if (!surf) {
    logError("surface: {}", surf.error().message);
    return;
  }
  surface_ = *surf;
  logInfo("surface: loaded {}x{} tifxyz grid from {}", surface_->width(),
          surface_->height(), surfaceSource);

  // Register the segment into this volume's voxel space. Preference order:
  // explicit CLI scale, the volume's transform.json (full affine from this
  // volume to the older canonical volume segments were traced on — we apply
  // its inverse), else a manifest-scored uniform-scale search.
  bool registered = false;
  if (surfaceScale != 1.0) {
    const double s = surfaceScale;
    surface_->applyAffine({{{s, 0, 0, 0}, {0, s, 0, 0}, {0, 0, s, 0}}});
    registered = true;
    logInfo("surface: applied explicit scale {}", s);
  } else if (auto tj = readMetaSync(*store_, "transform.json")) {
    try {
      const auto j = nlohmann::json::parse(
          std::string_view(reinterpret_cast<const char*>(tj->data()),
                           tj->size()));
      const auto& m = j.at("transformation_matrix");
      double A[3][4];
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c) A[r][c] = m[r][c].get<double>();
      // Invert: new = Ainv * (old - t). transform.json maps new -> old.
      const double det =
          A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
          A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
          A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
      if (std::abs(det) > 1e-12) {
        double I[3][3];
        I[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) / det;
        I[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) / det;
        I[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) / det;
        I[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) / det;
        I[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) / det;
        I[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) / det;
        I[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) / det;
        I[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) / det;
        I[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) / det;
        std::array<std::array<double, 4>, 3> inv{};
        for (int r = 0; r < 3; ++r) {
          for (int c = 0; c < 3; ++c) inv[r][c] = I[r][c];
          inv[r][3] = -(I[r][0] * A[0][3] + I[r][1] * A[1][3] +
                        I[r][2] * A[2][3]);
        }
        surface_->applyAffine(inv);
        registered = true;
        logInfo("surface: registered via transform.json (fixed volume {})",
                j.value("fixed_volume", std::string("?")));
      }
    } catch (const std::exception& e) {
      logWarn("surface: transform.json unusable: {}", e.what());
    }
  }
  if (!registered && manifest_) {
    const int Ls = std::min(3, volume_->levelCount() - 1);
    const auto& pts = surface_->points();
    auto score = [&](float cand) {
      int hit = 0, total = 0;
      for (std::size_t i = 0; i < pts.size(); i += 97) {
        const auto& p = pts[i];
        if (p[0] < 0.f) continue;
        ++total;
        const auto vx = std::uint32_t(p[0] * cand) >> Ls;
        const auto vy = std::uint32_t(p[1] * cand) >> Ls;
        const auto vz = std::uint32_t(p[2] * cand) >> Ls;
        if (!manifest_->empty(Ls, vz / 128, vy / 128, vx / 128)) ++hit;
      }
      return total ? float(hit) / float(total) : 0.f;
    };
    float bestScore = -1.f, bestScale = 1.f;
    for (const float cand : {1.0f, 1.3506f, 2.0f, 3.297f, 4.0f, 5.0f}) {
      const float s = score(cand);
      if (s > bestScore) {
        bestScore = s;
        bestScale = cand;
      }
    }
    for (float cand = bestScale * 0.8f; cand <= bestScale * 1.25f;
         cand += bestScale * 0.025f) {
      const float s = score(cand);
      if (s > bestScore) {
        bestScore = s;
        bestScale = cand;
      }
    }
    const double s = bestScale;
    surface_->applyAffine({{{s, 0, 0, 0}, {0, s, 0, 0}, {0, 0, s, 0}}});
    logInfo("surface: scored scale {} (occupancy {:.1f}%)", bestScale,
            bestScore * 100.f);
  }

  // Post-registration sanity: how many sample points land in non-empty
  // bricks of this volume?
  if (manifest_) {
    const int Ls = std::min(3, volume_->levelCount() - 1);
    int hit = 0, total = 0;
    for (std::size_t i = 0; i < surface_->points().size(); i += 97) {
      const auto& p = surface_->points()[i];
      if (p[0] < 0.f) continue;
      ++total;
      if (!manifest_->empty(Ls, std::uint32_t(p[2]) >> (Ls + 7),
                            std::uint32_t(p[1]) >> (Ls + 7),
                            std::uint32_t(p[0]) >> (Ls + 7)))
        ++hit;
    }
    logInfo("surface: {:.1f}% of registered points land on volume content",
            total ? 100.0 * hit / total : 0.0);
  }

  hasSurface_ = true;
  rebuildSurfaceMask();
  updateSurfaceStreaming();
}

bool ViewerWindow::loadOverlay(const std::string& overlaySource) {
  if (overlaySource.starts_with("http://") ||
      overlaySource.starts_with("https://")) {
    auto http = std::make_shared<store::HttpStore>(overlaySource);
    ovStore_ = std::make_shared<store::DiskCacheStore>(
        http, cacheDirFor("overlay-cache", overlaySource), 32ull << 30,
        ioPool_);
  } else {
    ovStore_ = std::make_shared<store::LocalStore>(overlaySource, ioPool_);
  }
  auto vol = zarr::OmeZarrVolume::open([this](const std::string& rel) {
    return readMetaSync(*ovStore_, rel);
  });
  if (!vol) {
    logError("overlay: open failed: {}", vol.error().message);
    return false;
  }
  ovVolume_ = *vol;

  const auto& l0 = ovVolume_->level(0);
  const auto& mainShape = volume_->level(0).shape;
  const std::uint32_t chunkDim = l0.chunks[0];
  const int levels = ovVolume_->levelCount();
  // The overlay must share the main volume's physical space; its voxel size
  // is mainShape/ovShape times coarser.
  const double spacing = double(mainShape[0]) / double(l0.shape[0]);
  logInfo("overlay: {}x{}x{} chunks {} levels {} ({}x downsampled)",
          l0.shape[2], l0.shape[1], l0.shape[0], chunkDim, levels, spacing);

  ovPipeline_ = std::make_unique<data::ChunkFetchPipeline>(
      ovStore_, ovVolume_, ovVolumeId_, *ramCache_);
  ovAssembler_ = std::make_unique<render::BrickAssembler>(ovVolume_,
                                                          ovVolumeId_,
                                                          *ramCache_);
  ovUploader_ = std::make_unique<render::GpuUploader>(*ovAssembler_, 1);
  std::vector<render::GpuBrickCache::LevelDims> dims;
  for (int i = 0; i < levels; ++i)
    dims.push_back({ovVolume_->level(i).chunkGridDims()});
  // 192^3 bricks are ~7 MB apiece: a small pool thrashes (visible as
  // predictions popping in and out), so give the overlay real headroom.
  ovGpuCache_ = std::make_unique<render::GpuBrickCache>(
      std::move(dims), chunkDim + 2, 3ull << 30);
  ovPlanner_ = std::make_unique<render::WorkingSetPlanner>(
      l0.shape, spacing, chunkDim, levels, nullptr);

  // Prefetch the overlay's coarse levels for instant fallback everywhere.
  {
    std::uint64_t budget = 256ull << 20;
    const std::uint64_t brickBytes =
        std::uint64_t{chunkDim} * chunkDim * chunkDim;
    std::vector<data::BrickRequest> prefetch;
    for (int li = levels - 1; li >= 0; --li) {
      const auto g = ovVolume_->level(li).chunkGridDims();
      const std::uint64_t count = std::uint64_t{g[0]} * g[1] * g[2];
      if (count * brickBytes > budget) break;
      budget -= count * brickBytes;
      for (std::uint32_t z = 0; z < g[0]; ++z)
        for (std::uint32_t y = 0; y < g[1]; ++y)
          for (std::uint32_t x = 0; x < g[2]; ++x)
            prefetch.push_back(data::BrickRequest{
                data::BrickKey{ovVolumeId_,
                               data::ChunkCoord{static_cast<std::uint8_t>(li),
                                                z, y, x}},
                1000.f * float(li) + 1.f});
    }
    if (!prefetch.empty()) ovPipeline_->submit(prefetch);
    logInfo("overlay: {} coarse bricks prefetched", prefetch.size());
  }

  mapper_->SetOverlay(ovGpuCache_.get(), l0.shape);
  return true;
}

void ViewerWindow::rebuildSurfaceMask() {
  if (!surface_) return;
  const float front =
      slabFrontSlider_ ? float(slabFrontSlider_->value()) : 64.f;
  const float behind =
      slabBehindSlider_ ? float(slabBehindSlider_->value()) : 64.f;
  std::vector<std::array<std::uint32_t, 3>> grids;
  for (int i = 0; i < gpuCache_->levelCount(); ++i)
    grids.push_back(gpuCache_->levelDims(i).grid);
  surfaceMask_.build(*surface_, volume_->level(0).shape, front, behind,
                     /*sampleSpacing=*/6.f, *ioPool_, grids);
  maskNeedsUpload_ = true;
  streamingActive_ = true;
}

void ViewerWindow::updateSurfaceStreaming() {
  if (!surface_) return;
  int Ls = 0;
  // Partially-downloaded volumes may have no data at fine levels at all
  // (the manifest knows): fall back to the finest level with content.
  if (manifest_) {
    while (Ls < volume_->levelCount() - 1 && manifest_->hasLevel(Ls)) {
      const auto g = volume_->level(Ls).chunkGridDims();
      bool any = false;
      for (std::uint32_t z = 0; z < g[0] && !any; ++z)
        for (std::uint32_t y = 0; y < g[1] && !any; ++y)
          for (std::uint32_t x = 0; x < g[2] && !any; ++x)
            any = !manifest_->empty(Ls, z, y, x);
      if (any) break;
      ++Ls;
    }
  }
  const float front =
      slabFrontSlider_ ? float(slabFrontSlider_->value()) : 64.f;
  const float behind =
      slabBehindSlider_ ? float(slabBehindSlider_->value()) : 64.f;

  // Covering brick set at the chosen level: every valid grid point, plus the
  // slab extremes along a cheap forward-difference normal.
  const auto& pts = surface_->points();
  const std::uint32_t W = surface_->width(), H = surface_->height();
  std::unordered_set<std::uint64_t> keys;
  std::vector<data::BrickKey> bricks;
  auto addPoint = [&](float px, float py, float pz) {
    if (px < 0.f || py < 0.f || pz < 0.f) return;
    const auto g = volume_->level(Ls).chunkGridDims();  // z,y,x
    const std::uint32_t bx = std::uint32_t(px) >> Ls;
    const std::uint32_t by = std::uint32_t(py) >> Ls;
    const std::uint32_t bz = std::uint32_t(pz) >> Ls;
    const std::uint32_t cx = std::min(bx / 128, g[2] - 1);
    const std::uint32_t cy = std::min(by / 128, g[1] - 1);
    const std::uint32_t cz = std::min(bz / 128, g[0] - 1);
    if (manifest_ && manifest_->hasLevel(Ls) &&
        manifest_->empty(Ls, cz, cy, cx))
      return;
    const data::BrickKey key{
        volumeId_,
        data::ChunkCoord{static_cast<std::uint8_t>(Ls), cz, cy, cx}};
    if (keys.insert(key.packed()).second) bricks.push_back(key);
  };
  for (std::uint32_t v = 0; v < H; ++v)
    for (std::uint32_t u = 0; u < W; ++u) {
      const auto& p = pts[std::size_t{v} * W + u];
      if (p[0] < 0.f) continue;
      addPoint(p[0], p[1], p[2]);
      if (front > 0.f || behind > 0.f) {
        // Forward-difference normal is plenty for brick-granular dilation.
        const std::uint32_t u1 = std::min(u + 1, W - 1);
        const std::uint32_t v1 = std::min(v + 1, H - 1);
        const auto& pu = pts[std::size_t{v} * W + u1];
        const auto& pv = pts[std::size_t{v1} * W + u];
        if (pu[0] < 0.f || pv[0] < 0.f) continue;
        const float dux = pu[0] - p[0], duy = pu[1] - p[1], duz = pu[2] - p[2];
        const float dvx = pv[0] - p[0], dvy = pv[1] - p[1], dvz = pv[2] - p[2];
        float nx = duy * dvz - duz * dvy;
        float ny = duz * dvx - dux * dvz;
        float nz = dux * dvy - duy * dvx;
        const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (len <= 0.f) continue;
        nx /= len, ny /= len, nz /= len;
        addPoint(p[0] - nx * front, p[1] - ny * front, p[2] - nz * front);
        addPoint(p[0] + nx * behind, p[1] + ny * behind, p[2] + nz * behind);
      }
    }

  surfaceBricks_ = std::move(bricks);
  const std::uint64_t bytes =
      std::uint64_t{surfaceBricks_.size()} * 128 * 128 * 128;
  logInfo("surface: {} bricks at level {} ({} MB) cover the slab",
          surfaceBricks_.size(), Ls, bytes >> 20);
  if (bytes > (3ull << 30)) {
    logWarn("surface: covering brick set exceeds budget; keeping coarse "
            "fallback only");
    surfaceBricks_.clear();
    return;
  }

  std::vector<data::BrickRequest> wanted;
  for (const auto& key : surfaceBricks_) {
    if (gpuCache_->isResident(key)) continue;
    if (auto brick = ramCache_->get(key)) {
      if (uploadQueue_.size() < 256 &&
          queuedKeys_.insert(key.packed()).second)
        uploadQueue_.push_back(std::move(brick));
    } else {
      wanted.push_back(data::BrickRequest{key, 1000.f * float(Ls) + 900.f});
    }
  }
  if (!wanted.empty()) pipeline_->submit(wanted);
  streamingActive_ = true;
}

void ViewerWindow::buildDisplayToolbar() {
  auto* tb = addToolBar("Display");
  tb->setMovable(false);
  auto addSlider = [&](const QString& name, int min, int max, int value) {
    tb->addWidget(new QLabel(QString("  %1 ").arg(name)));
    auto* s = new QSlider(Qt::Horizontal);
    s->setRange(min, max);
    s->setValue(value);
    s->setFixedWidth(150);
    tb->addWidget(s);
    connect(s, &QSlider::valueChanged, this,
            &ViewerWindow::applyDisplaySettings);
    return s;
  };
  tb->addWidget(new QLabel("  mode "));
  modeCombo_ = new QComboBox;
  // In surface mode the level slider doubles as the iso threshold.
  modeCombo_->addItems({"x-ray", "shaded", "surface", "clusters"});
  tb->addWidget(modeCombo_);
  connect(modeCombo_, &QComboBox::currentIndexChanged, this,
          &ViewerWindow::applyDisplaySettings);

  levelSlider_ = addSlider("level", 0, 255, 128);
  windowSlider_ = addSlider("window", 1, 255, 255);
  opacitySlider_ = addSlider("opacity", 1, 200, 50);

  // Second row: GPU voxel processing (+ surface/ink when present); the
  // first row is full with the display sliders.
  addToolBarBreak();
  tb = addToolBar("Filter/Surface/Ink");
  tb->setMovable(false);
  tb->addWidget(new QLabel("  filter "));
  filterCombo_ = new QComboBox;
  filterCombo_->addItems({"none", "smooth", "sharpen", "edges"});
  tb->addWidget(filterCombo_);
  connect(filterCombo_, &QComboBox::currentIndexChanged, this,
          &ViewerWindow::applyDisplaySettings);
  filterAmtSlider_ = addSlider("f-amt", 0, 200, 100);
  filterFloorSlider_ = addSlider("floor", 0, 255, 0);

  // Key-light direction (azimuth around z, elevation above the xy plane),
  // ambient scale, and surface-mode shadow rays.
  lightAzSlider_ = addSlider("light az", 0, 360, 35);
  lightElSlider_ = addSlider("light el", -90, 90, 55);
  lightAmbSlider_ = addSlider("ambient", 0, 100, 50);
  shadowsCheck_ = new QCheckBox("shadows");
  shadowsCheck_->setChecked(true);
  tb->addWidget(shadowsCheck_);
  connect(shadowsCheck_, &QCheckBox::toggled, this,
          &ViewerWindow::applyDisplaySettings);
  nearestCheck_ = new QCheckBox("fast taps");
  nearestCheck_->setToolTip(
      "Nearest-neighbor sampling: 1 texel per tap instead of 8. "
      "Faster deep zooms, blocky up close.");
  tb->addWidget(nearestCheck_);
  connect(nearestCheck_, &QCheckBox::toggled, this,
          &ViewerWindow::applyDisplaySettings);
  if (hasSurface_) {
    // Slab sliders rebuild the mask (~1 s): apply on release, not per tick.
    slabFrontSlider_ = addSlider("front", 0, 256, 64);
    slabBehindSlider_ = addSlider("behind", 0, 256, 64);
    for (auto* s : {slabFrontSlider_, slabBehindSlider_}) {
      disconnect(s, &QSlider::valueChanged, this,
                 &ViewerWindow::applyDisplaySettings);
      connect(s, &QSlider::sliderReleased, this, [this] {
        rebuildSurfaceMask();
        updateSurfaceStreaming();
      });
    }
    surfStrengthSlider_ = addSlider("surface", 0, 100, 60);
    showVolumeCheck_ = new QCheckBox("whole volume");
    showVolumeCheck_->setChecked(true);
    tb->addWidget(showVolumeCheck_);
    connect(showVolumeCheck_, &QCheckBox::toggled, this,
            &ViewerWindow::applyDisplaySettings);
  }
  if (ovGpuCache_) {
    inkStrengthSlider_ = addSlider("ink", 0, 100, 70);
    inkThresholdSlider_ = addSlider("ink thresh", 0, 255, 100);
  }

  // Third row: supervoxel segmentation (3D SNIC over a coarse level).
  addToolBarBreak();
  tb = addToolBar("Segment");
  tb->setMovable(false);
  seedsSlider_ = addSlider("seeds ‰", 1, 100, 10);  // 0.01%..1% of voxels
  disconnect(seedsSlider_, &QSlider::valueChanged, this,
             &ViewerWindow::applyDisplaySettings);
  segButton_ = new QPushButton("segment");
  tb->addWidget(segButton_);
  connect(segButton_, &QPushButton::clicked, this,
          &ViewerWindow::runSegmentation);
}

void ViewerWindow::runSegmentation() {
  if (!volume_ || !mapper_ || segRunning_.exchange(true)) return;
  // Cluster density: slider is tenths of a permille of the voxel count.
  const double frac = double(seedsSlider_->value()) / 10000.0;

  // Segment the finest pyramid level that still fits comfortably in memory
  // and CPU budget (SNIC is a single global priority-flood).
  int L = volume_->levelCount() - 1;
  for (int i = 0; i < volume_->levelCount(); ++i) {
    const auto& m = volume_->level(i);
    if (double(m.shape[0]) * double(m.shape[1]) * double(m.shape[2]) <=
        24e6) {
      L = i;
      break;
    }
  }
  // Air exclusion: scans normalize background anywhere from 0 to ~103, so
  // "empty" needs a data-derived threshold (Otsu over the level histogram).
  // A nonzero floor slider overrides it.
  const auto sliderThr = std::uint8_t(filterFloorSlider_->value());
  logInfo("segment: SNIC over level {} ({} voxels, seed frac {})", L,
          volume_->level(L).shape[0] * volume_->level(L).shape[1] *
              volume_->level(L).shape[2],
          frac);
  segButton_->setEnabled(false);

  std::thread([this, L, frac, sliderThr] {
    const auto meta = volume_->level(L);
    const std::array<std::uint32_t, 3> dims{std::uint32_t(meta.shape[0]),
                                            std::uint32_t(meta.shape[1]),
                                            std::uint32_t(meta.shape[2])};
    const std::size_t nvox = std::size_t{dims[0]} * dims[1] * dims[2];
    std::vector<std::uint8_t> vol(nvox, 0);

    // Assemble the whole level from the store (gdct/disk cached, so warm
    // runs never touch the network).
    const auto grid = meta.chunkGridDims();
    std::vector<std::byte> chunk(meta.chunkBytes());
    for (std::uint32_t cz = 0; cz < grid[0]; ++cz)
      for (std::uint32_t cy = 0; cy < grid[1]; ++cy)
        for (std::uint32_t cx = 0; cx < grid[2]; ++cx) {
          auto r = readMetaSync(*store_, volume_->chunkStoreKey(L, cz, cy, cx));
          if (!r) continue;  // missing chunk = fill value 0
          if (!volume_->codec(L).decode(r->span(), chunk)) continue;
          const std::uint32_t z0 = cz * meta.chunks[0],
                              y0 = cy * meta.chunks[1],
                              x0 = cx * meta.chunks[2];
          const std::uint32_t zn = std::min(meta.chunks[0], dims[0] - z0),
                              yn = std::min(meta.chunks[1], dims[1] - y0),
                              xn = std::min(meta.chunks[2], dims[2] - x0);
          for (std::uint32_t z = 0; z < zn; ++z)
            for (std::uint32_t y = 0; y < yn; ++y)
              std::memcpy(&vol[(std::size_t{z0 + z} * dims[1] + y0 + y) *
                                   dims[2] +
                               x0],
                          &chunk[(std::size_t{z} * meta.chunks[1] + y) *
                                 meta.chunks[2]],
                          xn);
        }

    const auto t0 = std::chrono::steady_clock::now();
    const std::uint8_t thr =
        sliderThr > 0 ? sliderThr : data::otsuThreshold(vol);
    logInfo("segment: air threshold {} ({})", thr,
            sliderThr > 0 ? "floor slider" : "otsu");
    auto clusters = data::snic3d(vol, dims, frac, 0.12f, thr);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();

    // Voxel coords at level L -> world (world units are level-0 voxels).
    const auto ws =
        data::clustersToWorldSpheres(clusters, std::exp2(float(L)), 2);
    // Cluster means bunch into a narrow band; stretch the 5th-95th
    // percentile across the full colormap so structure is visible.
    std::vector<float> vals;
    vals.reserve(ws.size());
    for (const auto& w : ws) vals.push_back(w.value);
    std::ranges::sort(vals);
    const float lo = vals.empty() ? 0.f : vals[vals.size() * 5 / 100];
    const float hi = vals.empty() ? 1.f : vals[vals.size() * 95 / 100];
    const float span = std::max(hi - lo, 1e-4f);
    std::vector<render::vtkScrollVolumeMapper::ClusterSphere> spheres;
    spheres.reserve(ws.size());
    for (const auto& w : ws)
      spheres.push_back({w.x, w.y, w.z, w.radius,
                         std::clamp((w.value - lo) / span, 0.f, 1.f)});
    logInfo("segment: {} clusters ({} kept) in {} ms", clusters.size(),
            spheres.size(), ms);

    QMetaObject::invokeMethod(
        this,
        [this, spheres = std::move(spheres)]() mutable {
          mapper_->SetClusters(std::move(spheres));
          modeCombo_->setCurrentIndex(3);  // switch to cluster view
          segButton_->setEnabled(true);
          segRunning_ = false;
          streamingActive_ = true;  // trigger re-render
        },
        Qt::QueuedConnection);
  }).detach();
}

void ViewerWindow::applyDisplaySettings() {
  const float level = static_cast<float>(levelSlider_->value()) / 255.f;
  const float window = static_cast<float>(windowSlider_->value()) / 255.f;
  if (mapper_) {
    mapper_->SetWindowLevel(window, level);
    if (modeCombo_) mapper_->SetRenderMode(modeCombo_->currentIndex());
    if (filterCombo_)
      mapper_->SetVoxelFilter(filterCombo_->currentIndex(),
                              float(filterAmtSlider_->value()) / 100.f,
                              float(filterFloorSlider_->value()) / 255.f);
    if (nearestCheck_) mapper_->SetNearestSampling(nearestCheck_->isChecked());
    if (lightAzSlider_) {
      const float az = float(lightAzSlider_->value()) * 3.14159265f / 180.f;
      const float el = float(lightElSlider_->value()) * 3.14159265f / 180.f;
      mapper_->SetLighting({std::cos(el) * std::cos(az),
                            std::cos(el) * std::sin(az), std::sin(el)},
                           float(lightAmbSlider_->value()) / 50.f,
                           shadowsCheck_->isChecked());
    }
    mapper_->SetOpacityScale(static_cast<float>(opacitySlider_->value()) /
                             1000.f);
    if (hasSurface_ && surfStrengthSlider_) {
      mapper_->SetMaskStyle(float(surfStrengthSlider_->value()) / 100.f);
      mapper_->SetMaskIsolate(!showVolumeCheck_->isChecked());
    }
    if (ovGpuCache_ && inkStrengthSlider_) {
      mapper_->SetOverlayStyle(float(inkStrengthSlider_->value()) / 100.f,
                               float(inkThresholdSlider_->value()) / 255.f);
    }
  }
  // Keep the planner's pre-fetch culling in lockstep with the GPU's
  // occupancy skip: bricks whose max density can't clear the window's
  // zero-opacity cutoff are never even requested.
  float cut = std::max(0.f, level - 0.5f * window);
  if (filterCombo_ && filterCombo_->currentIndex() != 3)
    cut = std::max(cut, float(filterFloorSlider_->value()) / 255.f);
  if (planner_)
    planner_->setVisibilityThreshold(
        static_cast<std::uint8_t>(std::min(255.f, cut * 255.f)));
  // Slider drags re-render every tick: reuse the wheel-burst window so they
  // run at interaction quality, with a full-quality pass on settle.
  lastWheel_ = std::chrono::steady_clock::now();
  streamingActive_ = true;  // re-render + replan under the new window
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
    store_ = std::make_shared<store::DiskCacheStore>(
        http, cacheDirFor("chunk-cache", source), 64ull << 30, ioPool_);
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

  // Recompress raw remote chunks with gpudct in the disk cache. Safe only for
  // genuinely uncompressed u8 128^3 chunk layouts (the codec's brick unit):
  // the round trip is lossy, so anything else must stay verbatim.
  if (auto* dc = dynamic_cast<store::DiskCacheStore*>(store_.get());
      dc && qgetenv("SCROLLVIEWER_GDCT") != "0") {
    bool eligible = true;
    for (int i = 0; i < levels; ++i) {
      const auto& m = volume_->level(i);
      eligible &= m.compressor == zarr::CompressorId::None &&
                  m.dtype == zarr::Dtype::U8 && m.chunks[0] == 128 &&
                  m.chunks[1] == 128 && m.chunks[2] == 128;
    }
    if (eligible) {
      bool ok = false;
      float quality =
          qEnvironmentVariable("SCROLLVIEWER_GDCT_QUALITY").toFloat(&ok);
      if (!ok || quality <= 0.f) quality = 1.0f;
      if (auto t = store::makeGdctTranscoder(quality)) {
        dc->setTranscoder(std::move(t));
        logInfo("disk cache: gpudct recompression on (quality {})", quality);
      }
    }
  }

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

  // Prefetch whole coarse levels once at open (coarsest first, as many
  // levels as fit a fixed budget): fallback rendering can then resolve
  // every ray from any camera angle, so orbiting a cold remote volume
  // never pops. Priority sits just below planner/feedback requests of the
  // same level; preRenderUpload drains everything decoded to the GPU.
  {
    const std::uint64_t brickBytes =
        std::uint64_t{chunkDim} * chunkDim * chunkDim;
    std::uint64_t budget = 512ull << 20;
    std::vector<data::BrickRequest> prefetch;
    for (int li = levels - 1; li >= 0; --li) {
      const auto g = volume_->level(li).chunkGridDims();  // z,y,x
      std::vector<data::BrickRequest> lvl;
      bool fits = true;
      for (std::uint32_t z = 0; z < g[0] && fits; ++z)
        for (std::uint32_t y = 0; y < g[1] && fits; ++y)
          for (std::uint32_t x = 0; x < g[2] && fits; ++x) {
            if (manifest_ && manifest_->hasLevel(li) &&
                manifest_->empty(li, z, y, x))
              continue;
            lvl.push_back(data::BrickRequest{
                data::BrickKey{volumeId_,
                               data::ChunkCoord{static_cast<std::uint8_t>(li),
                                                z, y, x}},
                1000.f * float(li) + 1.f});
            fits = std::uint64_t{lvl.size()} * brickBytes <= budget;
          }
      if (!fits) break;
      budget -= std::uint64_t{lvl.size()} * brickBytes;
      prefetch.insert(prefetch.end(), lvl.begin(), lvl.end());
    }
    if (!prefetch.empty()) {
      pipeline_->submit(prefetch);
      logInfo("prefetch: {} coarse bricks submitted at open", prefetch.size());
    }
  }

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
  // Keep the surface window's working set out of the LRU's reach.
  for (const auto& key : surfaceBricks_) gpuCache_->touch(key);

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
  auto feedback = mapper_->TakeFeedbackRequests();
  if (surfaceWindow_) {
    auto sf = surfaceWindow_->mapper()->TakeFeedbackRequests();
    feedback.insert(feedback.end(), sf.begin(), sf.end());
  }
  for (const auto& coord : feedback) {
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
  // Hard bound on the upload re-feed queue: every shared_ptr here pins decoded
  // voxels OUTSIDE the RamCache budget, so an unbounded queue is an OOM
  // (planner re-requests anything dropped once the ring catches up).
  constexpr std::size_t kMaxUploadQueue = 256;
  std::vector<data::BrickRequest> wanted;
  wanted.reserve(plan.missing.size());
  for (const auto& r : plan.missing) {
    if (auto brick = ramCache_->get(r.key)) {
      if (uploadQueue_.size() < kMaxUploadQueue &&
          queuedKeys_.insert(r.key.packed()).second)
        uploadQueue_.push_back(std::move(brick));
    } else {
      wanted.push_back(r);
    }
  }

  // One level coarser while dragging: ~8x fewer voxels to march.
  mapper_->SetDesiredLevel(float(plan.desiredLevel + (interacting_ ? 1 : 0)));
  pipeline_->submit(wanted);

  // Overlay volume: its own plan over the same camera; RAM hits re-feed the
  // upload queue, the rest stream through the overlay pipeline.
  bool ovPending = false;
  if (ovPlanner_) {
    auto ovPlan = ovPlanner_->plan(renderer_, ovVolumeId_, vh, *ovGpuCache_);
    for (const auto& key : ovPlan.visible) ovGpuCache_->touch(key);
    std::vector<data::BrickRequest> ovWanted;
    for (const auto& r : ovPlan.missing) {
      // The overlay is already 4x downsampled; streaming its fine levels
      // buys little visually and the working set outgrows the pool (7 MB
      // bricks -> permanent evict/re-decode churn). Level 2+ always fits.
      if (r.key.coord.level < 2) continue;
      if (auto brick = ramCache_->get(r.key)) {
        if (uploadQueue_.size() < kMaxUploadQueue &&
            queuedKeys_.insert(r.key.packed()).second)
          uploadQueue_.push_back(std::move(brick));
      } else {
        ovWanted.push_back(r);
      }
    }
    ovPipeline_->submit(ovWanted);
    const auto os = ovPipeline_->stats();
    ovPending = !ovWanted.empty() || os.inflightFetches > 0 ||
                os.queuedDecodes > 0 || os.ready > 0 ||
                ovUploader_->pendingCount() > 0;
  }

  const auto stats = pipeline_->stats();
  const bool anythingPending = !wanted.empty() || stats.inflightFetches > 0 ||
                               stats.queuedDecodes > 0 || stats.ready > 0 ||
                               !uploadQueue_.empty() ||
                               uploader_->pendingCount() > 0 || ovPending;

  // Progressive quality: reduced resolution during camera motion (drag or
  // wheel burst) and while bricks stream; one full-resolution frame lands
  // at convergence below.
  mapper_->SetRenderScale(motion ? 0.6f : (anythingPending ? 0.75f : 1.0f));

  // Periodic streaming + profiling diagnostics (~every 2 s while active).
  // Suppressed while the benchmark owns the profiling accumulators.
  if (!benchActive_ && pumpCounter_ % 120 == 0) {
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
  // The surface window shares the brick pool: repaint it while bricks are
  // still landing (its own interactor covers camera-driven repaints).
  if (surfaceWindow_ && (anythingPending || !streamingActive_))
    surfaceWindow_->requestRender();
}

void ViewerWindow::preRenderUpload() {
  // GL context is current (called from mapper Render).
  constexpr int kUploadBudget = 32;  // async PBO path: cheap per brick

  if (!uploader_->initialized()) uploader_->initialize(96);

  // Freshly built surface mask: upload in this (primary) context — the
  // textures are visible to the shell window through the share group.
  if (maskNeedsUpload_) {
    maskNeedsUpload_ = false;
    surfaceMask_.upload();
    mapper_->SetDensityMask(&surfaceMask_);
  }

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
    const bool isOverlay = brick->key.volumeId == ovVolumeId_;
    auto& cache = isOverlay ? *ovGpuCache_ : *gpuCache_;
    auto& uploader = isOverlay ? *ovUploader_ : *uploader_;
    if (brick->isFillValue) {
      cache.markEmpty(brick->key);
      queuedKeys_.erase(packed);
      continue;
    }
    const bool pinned =
        !isOverlay && brick->key.coord.level >= volume_->levelCount() - 2;
    if (!uploader.enqueue(std::move(brick), pinned)) {
      queuedKeys_.erase(packed);
      break;  // ring saturated; try again next frame
    }
  }

  const int uploaded = uploader_->drainToGpu(*gpuCache_, kUploadBudget);
  (void)uploaded;

  // Overlay stack: initialize lazily in this context, drain its pipeline
  // into its own cache, and keep its page tables synced.
  if (ovGpuCache_) {
    if (!ovGpuCache_->initialized()) {
      auto* renWin = static_cast<vtkOpenGLRenderWindow*>(
          vtkWidget_->renderWindow());
      ovGpuCache_->initialize(renWin);
    }
    if (!ovUploader_->initialized()) ovUploader_->initialize(24);
    ovGpuCache_->frameBegin(pumpCounter_);
    for (auto& rb : ovPipeline_->drainReady()) {
      if (rb.brick->isFillValue) {
        ovGpuCache_->markEmpty(rb.brick->key);
      } else if (queuedKeys_.insert(rb.brick->key.packed()).second) {
        if (!ovUploader_->enqueue(rb.brick, false))
          queuedKeys_.erase(rb.brick->key.packed());
      }
      ovPipeline_->unpin(rb.brick->key);
    }
    ovUploader_->drainToGpu(*ovGpuCache_, 12);
    ovGpuCache_->syncPageTables();
  }
  // Uploaded keys leave the dedup set lazily: once resident, the planner
  // stops listing them and stale entries get erased on the re-feed path.
  if (uploader_->pendingCount() == 0) queuedKeys_.clear();
}

}  // namespace sv::app
