#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include "core/ThreadPool.h"
#include "data/RamCache.h"
#include "render/BrickAssembler.h"
#include "render/GpuBrickCache.h"

class vtkOpenGLRenderWindow;

namespace sv::render {

// Asynchronous brick upload path:
//   enqueue(brick) -> [worker: assemble borders directly into a
//   persistent-mapped PBO slot] -> drainToGpu() on the render thread issues
//   glTexSubImage3D sourced from the PBO (async DMA, no client-memory
//   stall) and recycles slots once their fence signals.
class GpuUploader {
 public:
  GpuUploader(BrickAssembler& assembler, unsigned workerThreads = 2);
  ~GpuUploader();

  // GL context must be current. slotCount ~ 2-3 frames of upload budget.
  void initialize(std::uint32_t slotCount);
  void release();  // GL context must be current
  bool initialized() const { return pbo_ != 0; }

  // Hands a decoded brick to the assembly workers. Returns false when the
  // pending queue is saturated (caller retries later; nothing is lost).
  bool enqueue(std::shared_ptr<const data::Brick> brick, bool pinned);

  // Render thread: upload up to budget assembled bricks, recycle fenced
  // slots. Returns number uploaded.
  int drainToGpu(GpuBrickCache& cache, int budget);

  std::size_t pendingCount() const;

 private:
  enum class SlotState : std::uint8_t { Free, Writing, Ready, InFlight };

  struct Slot {
    SlotState state = SlotState::Free;
    data::BrickKey key;
    bool pinned = false;
    std::uint8_t minVal = 0;
    std::uint8_t maxVal = 255;
  };

  struct FenceBatch {
    void* sync = nullptr;  // GLsync
    std::vector<std::uint32_t> slots;
  };

  void kickWorkers();
  void workerAssemble(std::stop_token st);

  BrickAssembler& assembler_;
  std::size_t slotBytes_ = 0;

  unsigned int pbo_ = 0;
  std::uint8_t* mapped_ = nullptr;
  std::vector<Slot> slots_;
  std::deque<FenceBatch> fences_;

  mutable std::mutex mutex_;
  std::deque<std::pair<std::shared_ptr<const data::Brick>, bool>> pending_;

  ThreadPool workers_;  // last member: joins before state dies
};

}  // namespace sv::render
