#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <span>
#include <stop_token>
#include <unordered_map>
#include <vector>

#include "core/ThreadPool.h"
#include "data/BrickKey.h"
#include "data/RamCache.h"
#include "store/ChunkStore.h"
#include "zarr/OmeZarrVolume.h"

namespace sv::data {

struct BrickRequest {
  BrickKey key;
  float priority = 0.f;  // higher = sooner; opaque to the pipeline
};

struct ReadyBrick {
  std::shared_ptr<const Brick> brick;  // pinned in RamCache until unpin()
};

struct PipelineConfig {
  int maxInflightFetches = 128;
  int maxQueuedDecodes = 0;      // 0 = 2 * decode threads
  std::uint64_t maxReadyPinnedBytes = 1ull << 30;
  unsigned decodeThreads = 0;    // 0 = max(2, hw_concurrency - 4)
};

struct PipelineStats {
  std::size_t queued = 0;
  std::size_t inflightFetches = 0;
  std::size_t queuedDecodes = 0;
  std::size_t ready = 0;
  std::uint64_t readyPinnedBytes = 0;
};

// Prioritized async chunk-fetch pipeline:
//   submit(wanted set) -> [priority heap] -> store fetch -> decode -> RamCache
//   (pinned) -> drainReady() on the render thread -> GPU upload -> unpin.
//
// Re-prioritization is lazy: submit() overwrites priorities in the dedupe map;
// stale heap entries are re-pushed with the fresh priority when popped.
// Requests absent from a newer submit epoch are dropped before starting IO;
// in-flight fetches for unwanted keys get their stop_source triggered.
// Decodes always run to completion (the bytes are already paid for).
class ChunkFetchPipeline {
 public:
  ChunkFetchPipeline(std::shared_ptr<store::ChunkStore> chunkStore,
                     std::shared_ptr<zarr::OmeZarrVolume> volume,
                     std::uint32_t volumeId, RamCache& ramCache,
                     PipelineConfig config = {});
  ~ChunkFetchPipeline();

  // Full desired set for the current view state. Keys already resident in the
  // RamCache are skipped by the caller (planner diffs against residency).
  void submit(std::span<const BrickRequest> wanted);

  // Render thread: collect decoded bricks (each pinned; unpin after upload).
  std::vector<ReadyBrick> drainReady();

  void unpin(const BrickKey& key);

  PipelineStats stats() const;

 private:
  enum class State { Queued, Fetching, Decoding, Done };

  struct Tracked {
    float priority = 0.f;
    std::uint64_t epoch = 0;
    State state = State::Queued;
    std::shared_ptr<std::stop_source> stop;
  };

  struct HeapEntry {
    float priority;
    std::uint64_t seq;  // FIFO tiebreak
    BrickKey key;
    bool operator<(const HeapEntry& o) const {
      if (priority != o.priority) return priority < o.priority;
      return seq > o.seq;
    }
  };

  void pump();  // issue fetches while under the caps (mutex_ held by caller)
  void onFetchDone(BrickKey key, store::StoreResult r);
  void decodeAndPublish(BrickKey key, ByteBuffer compressed, bool isMissing,
                        std::stop_token st);

  std::shared_ptr<store::ChunkStore> store_;
  std::shared_ptr<zarr::OmeZarrVolume> volume_;
  const std::uint32_t volumeId_;
  RamCache& ramCache_;
  PipelineConfig config_;

  std::unique_ptr<ThreadPool> decodePool_;

  mutable std::mutex mutex_;
  std::condition_variable callbacksDoneCv_;
  int pendingStoreCallbacks_ = 0;
  std::priority_queue<HeapEntry> heap_;
  std::unordered_map<BrickKey, Tracked, BrickKeyHash> tracked_;
  std::uint64_t currentEpoch_ = 0;
  std::uint64_t seqCounter_ = 0;
  int inflightFetches_ = 0;
  int queuedDecodes_ = 0;

  std::vector<ReadyBrick> ready_;
  std::uint64_t readyPinnedBytes_ = 0;
};

}  // namespace sv::data
