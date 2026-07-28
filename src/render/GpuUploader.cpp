#include "render/GpuUploader.h"

#include <vtk_glew.h>

#include "core/Log.h"
#include "core/Profiling.h"

namespace sv::render {

namespace {
constexpr std::size_t kMaxPending = 4096;  // shared_ptrs only; cheap
}

GpuUploader::GpuUploader(BrickAssembler& assembler, unsigned workerThreads)
    : assembler_(assembler),
      slotBytes_(assembler.borderedBytes()),
      workers_("assemble", workerThreads) {}

GpuUploader::~GpuUploader() = default;

void GpuUploader::initialize(std::uint32_t slotCount) {
  glGenBuffers(1, &pbo_);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
  const GLbitfield flags =
      GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
  glBufferStorage(GL_PIXEL_UNPACK_BUFFER, slotBytes_ * slotCount, nullptr,
                  flags);
  mapped_ = static_cast<std::uint8_t*>(glMapBufferRange(
      GL_PIXEL_UNPACK_BUFFER, 0, slotBytes_ * slotCount, flags));
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  slots_.assign(slotCount, Slot{});
  logInfo("upload ring: {} slots x {} KB (persistent-mapped)", slotCount,
          slotBytes_ >> 10);
}

void GpuUploader::release() {
  for (auto& fb : fences_)
    if (fb.sync) glDeleteSync(static_cast<GLsync>(fb.sync));
  fences_.clear();
  if (pbo_) {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glDeleteBuffers(1, &pbo_);
    pbo_ = 0;
    mapped_ = nullptr;
  }
  slots_.clear();
}

bool GpuUploader::enqueue(std::shared_ptr<const data::Brick> brick,
                          bool pinned) {
  {
    std::lock_guard lock(mutex_);
    if (pending_.size() >= kMaxPending) return false;
    pending_.emplace_back(std::move(brick), pinned);
  }
  kickWorkers();
  return true;
}

void GpuUploader::kickWorkers() {
  workers_.post([this](std::stop_token st) { workerAssemble(st); });
}

void GpuUploader::workerAssemble(std::stop_token st) {
  for (;;) {
    if (st.stop_requested() || !mapped_) return;

    std::shared_ptr<const data::Brick> brick;
    bool pinned = false;
    std::uint32_t slotIdx = UINT32_MAX;
    {
      std::lock_guard lock(mutex_);
      if (pending_.empty()) return;
      for (std::uint32_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].state == SlotState::Free) {
          slotIdx = i;
          break;
        }
      }
      if (slotIdx == UINT32_MAX) return;  // ring full; drain re-kicks
      brick = std::move(pending_.front().first);
      pinned = pending_.front().second;
      pending_.pop_front();
      slots_[slotIdx].state = SlotState::Writing;
      slots_[slotIdx].key = brick->key;
      slots_[slotIdx].pinned = pinned;
      slots_[slotIdx].minVal = brick->minVal;
      slots_[slotIdx].maxVal = brick->maxVal;
    }

    {
      ProfScope timeAssemble(profStages().assemble);
      assembler_.assemble(*brick,
                          mapped_ + std::size_t{slotIdx} * slotBytes_);
    }

    std::lock_guard lock(mutex_);
    slots_[slotIdx].state = SlotState::Ready;
  }
}

int GpuUploader::drainToGpu(GpuBrickCache& cache, int budget) {
  if (!pbo_) return 0;
  ProfScope timeUpload(profStages().upload);

  // Recycle slots whose GPU-side copy finished.
  bool freed = false;
  while (!fences_.empty()) {
    FenceBatch& fb = fences_.front();
    const GLenum r = glClientWaitSync(static_cast<GLsync>(fb.sync), 0, 0);
    if (r != GL_ALREADY_SIGNALED && r != GL_CONDITION_SATISFIED) break;
    glDeleteSync(static_cast<GLsync>(fb.sync));
    {
      std::lock_guard lock(mutex_);
      for (auto idx : fb.slots) slots_[idx].state = SlotState::Free;
    }
    fences_.pop_front();
    freed = true;
  }

  // Upload ready slots (budgeted), sourced from the bound PBO.
  std::vector<std::uint32_t> batch;
  {
    std::lock_guard lock(mutex_);
    for (std::uint32_t i = 0; i < slots_.size() &&
                              batch.size() < static_cast<std::size_t>(budget);
         ++i) {
      if (slots_[i].state == SlotState::Ready) batch.push_back(i);
    }
    for (auto idx : batch) slots_[idx].state = SlotState::InFlight;
  }

  const int uploaded = static_cast<int>(batch.size());
  if (!batch.empty()) {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    for (auto idx : batch) {
      cache.uploadFromBoundPbo(slots_[idx].key,
                               std::size_t{idx} * slotBytes_,
                               slots_[idx].pinned, slots_[idx].minVal,
                               slots_[idx].maxVal);
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    FenceBatch fb;
    fb.sync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    fb.slots = std::move(batch);
    fences_.push_back(std::move(fb));
  }

  if (freed || uploaded > 0) kickWorkers();
  return uploaded;
}

std::size_t GpuUploader::pendingCount() const {
  std::lock_guard lock(mutex_);
  std::size_t inRing = 0;
  for (const auto& s : slots_)
    if (s.state != SlotState::Free) ++inRing;
  return pending_.size() + inRing;
}

}  // namespace sv::render
