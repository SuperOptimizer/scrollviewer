#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

namespace sv {

// Lock-free accumulator for a profiled stage: total nanoseconds + samples.
// Workers add(); the stats reporter snapshots and resets periodically.
class ProfAccum {
 public:
  void add(std::uint64_t ns) {
    ns_.fetch_add(ns, std::memory_order_relaxed);
    count_.fetch_add(1, std::memory_order_relaxed);
  }

  struct Snapshot {
    double totalMs = 0;
    std::uint64_t count = 0;
    double avgMs() const { return count ? totalMs / double(count) : 0.0; }
  };

  Snapshot take() {
    Snapshot s;
    s.totalMs = double(ns_.exchange(0, std::memory_order_relaxed)) * 1e-6;
    s.count = count_.exchange(0, std::memory_order_relaxed);
    return s;
  }

 private:
  std::atomic<std::uint64_t> ns_{0};
  std::atomic<std::uint64_t> count_{0};
};

// RAII scope timer feeding a ProfAccum.
class ProfScope {
 public:
  explicit ProfScope(ProfAccum& accum)
      : accum_(accum), start_(std::chrono::steady_clock::now()) {}
  ~ProfScope() {
    accum_.add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now() - start_)
                   .count());
  }

 private:
  ProfAccum& accum_;
  std::chrono::steady_clock::time_point start_;
};

// Global stage accumulators (simple by design; one process, known stages).
struct ProfStages {
  ProfAccum fetch;     // store read -> callback (latency, includes queue)
  ProfAccum decode;    // blosc/raw decode per brick
  ProfAccum assemble;  // border assembly per brick (workers)
  ProfAccum plan;      // WorkingSetPlanner::plan per call
  ProfAccum upload;      // render-thread drainToGpu per frame
  ProfAccum frame;       // mapper Render() wall time per frame
  ProfAccum gpuRaymarch; // GL timer query around the raymarch draw
};

ProfStages& profStages();

}  // namespace sv
