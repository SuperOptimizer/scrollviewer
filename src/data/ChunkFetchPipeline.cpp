#include "data/ChunkFetchPipeline.h"

#include <algorithm>
#include <cstring>
#include <unordered_set>

#include "core/Log.h"
#include "core/Profiling.h"

namespace sv::data {

namespace {
// Items older than the previous submit epoch are dropped before IO starts.
constexpr std::uint64_t kStaleWindow = 1;
}  // namespace

ChunkFetchPipeline::ChunkFetchPipeline(
    std::shared_ptr<store::ChunkStore> chunkStore,
    std::shared_ptr<zarr::OmeZarrVolume> volume, std::uint32_t volumeId,
    RamCache& ramCache, PipelineConfig config)
    : store_(std::move(chunkStore)),
      volume_(std::move(volume)),
      volumeId_(volumeId),
      ramCache_(ramCache),
      config_(config) {
  unsigned decodeThreads = config_.decodeThreads;
  if (decodeThreads == 0) {
    const unsigned hw = std::max(2u, std::thread::hardware_concurrency());
    decodeThreads = std::max(2u, hw > 4 ? hw - 4 : 2u);
  }
  if (config_.maxQueuedDecodes == 0)
    config_.maxQueuedDecodes = static_cast<int>(decodeThreads) * 2;
  decodePool_ = std::make_unique<ThreadPool>("decode", decodeThreads);
}

ChunkFetchPipeline::~ChunkFetchPipeline() {
  // Teardown contract: the pipeline must be destroyed while its store (and
  // the store's IO pool) are still alive, because every issued read's
  // callback references this object and must be allowed to finish.
  {
    std::unique_lock lock(mutex_);
    for (auto& [key, t] : tracked_) {
      if (t.stop) t.stop->request_stop();
    }
    callbacksDoneCv_.wait(lock, [this] { return pendingStoreCallbacks_ == 0; });
  }
  // Joins running decode jobs (they use mutex_/ramCache_, both still alive);
  // queued jobs are discarded.
  decodePool_.reset();
}

void ChunkFetchPipeline::submit(std::span<const BrickRequest> wanted) {
  std::lock_guard lock(mutex_);
  ++currentEpoch_;

  std::unordered_set<std::uint64_t> wantedKeys;
  wantedKeys.reserve(wanted.size());

  for (const BrickRequest& r : wanted) {
    wantedKeys.insert(r.key.packed());
    auto [it, inserted] = tracked_.try_emplace(r.key);
    Tracked& t = it->second;
    t.epoch = currentEpoch_;
    if (inserted) {
      t.priority = r.priority;
      t.state = State::Queued;
      heap_.push(HeapEntry{r.priority, seqCounter_++, r.key});
    } else if (t.state == State::Queued && t.priority != r.priority) {
      // Lazy decrease/increase-key: the fresh entry supersedes stale ones.
      t.priority = r.priority;
      heap_.push(HeapEntry{r.priority, seqCounter_++, r.key});
    }
  }

  // Cancel in-flight fetches that are no longer wanted by anyone.
  for (auto& [key, t] : tracked_) {
    if (t.state == State::Fetching && t.stop &&
        !wantedKeys.contains(key.packed())) {
      t.stop->request_stop();
    }
  }

  pump();
}

void ChunkFetchPipeline::pump() {
  // Caller holds mutex_. Issue fetches while under all caps.
  while (inflightFetches_ < config_.maxInflightFetches &&
         queuedDecodes_ < config_.maxQueuedDecodes &&
         readyPinnedBytes_ < config_.maxReadyPinnedBytes && !heap_.empty()) {
    const HeapEntry e = heap_.top();
    heap_.pop();

    auto it = tracked_.find(e.key);
    if (it == tracked_.end()) continue;          // dropped earlier
    Tracked& t = it->second;
    if (t.state != State::Queued) continue;      // duplicate heap entry
    if (e.priority != t.priority) {
      // Stale priority: superseding entry is (or will be) in the heap.
      continue;
    }
    if (t.epoch + kStaleWindow < currentEpoch_) {
      tracked_.erase(it);                        // no longer wanted
      continue;
    }

    t.state = State::Fetching;
    t.stop = std::make_shared<std::stop_source>();
    ++inflightFetches_;
    ++pendingStoreCallbacks_;

    const BrickKey key = e.key;
    const auto& c = key.coord;
    std::string storeKey =
        volume_->chunkStoreKey(c.level, c.z, c.y, c.x);
    const auto issued = std::chrono::steady_clock::now();
    store_->read(std::move(storeKey), t.stop->get_token(),
                 [this, key, issued](store::StoreResult r) {
                   profStages().fetch.add(
                       std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now() - issued)
                           .count());
                   onFetchDone(key, std::move(r));
                 });
  }
}

void ChunkFetchPipeline::onFetchDone(BrickKey key, store::StoreResult r) {
  std::unique_lock lock(mutex_);
  --inflightFetches_;
  // Signal potential destructor waiters once we're done touching state; the
  // decrement itself must happen under the same lock as the wait predicate.
  struct CallbackGuard {
    ChunkFetchPipeline& p;
    ~CallbackGuard() {
      --p.pendingStoreCallbacks_;
      p.callbacksDoneCv_.notify_all();
    }
  } guard{*this};

  auto it = tracked_.find(key);
  if (it == tracked_.end()) {
    pump();
    return;
  }

  if (!r && r.error().kind == store::StoreError::Kind::Cancelled) {
    tracked_.erase(it);
    pump();
    return;
  }
  if (!r && !r.error().isNotFound()) {
    logWarn("fetch failed for level {} chunk ({},{},{}): {}", key.coord.level,
            key.coord.z, key.coord.y, key.coord.x, r.error().message);
    tracked_.erase(it);  // renderer re-requests on a future submit
    pump();
    return;
  }

  it->second.state = State::Decoding;
  it->second.stop.reset();
  ++queuedDecodes_;

  const bool isMissing = !r;
  ByteBuffer compressed = isMissing ? ByteBuffer{} : std::move(*r);

  decodePool_->post([this, key, compressed = std::move(compressed),
                     isMissing](std::stop_token st) mutable {
    decodeAndPublish(key, std::move(compressed), isMissing, st);
  });
}

void ChunkFetchPipeline::decodeAndPublish(BrickKey key, ByteBuffer compressed,
                                          bool isMissing, std::stop_token) {
  const auto& meta = volume_->level(key.coord.level);
  const std::size_t decodedBytes = meta.chunkBytes();

  auto brick = std::make_shared<Brick>();
  brick->key = key;
  brick->isFillValue = isMissing;

  // Fill-value bricks (absent chunks, or present-but-uniform chunks in
  // masked volumes) carry no voxel payload: the renderer marks them EMPTY in
  // the page table and consumers synthesize the fill value on read.
  if (!isMissing) {
    ProfScope timeDecode(profStages().decode);
    brick->voxels = ByteBuffer::uninitialized(decodedBytes);
    auto ok = volume_->codec(key.coord.level)
                  .decode(compressed.span(), brick->voxels.span());
    if (ok) {
      // One pass: min/max (8-bit domain) for occupancy culling; uniform
      // fill-value chunks become weightless.
      std::uint8_t lo = 255, hi = 0;
      if (meta.dtype == zarr::Dtype::U8) {
        for (const std::byte b : brick->voxels.span()) {
          const auto v = static_cast<std::uint8_t>(b);
          lo = std::min(lo, v);
          hi = std::max(hi, v);
        }
      } else {
        const auto* p16 =
            reinterpret_cast<const std::uint16_t*>(brick->voxels.data());
        for (std::size_t i = 0; i < decodedBytes / 2; ++i) {
          const auto v = static_cast<std::uint8_t>(p16[i] >> 8);
          lo = std::min(lo, v);
          hi = std::max(hi, v);
        }
      }
      brick->minVal = lo;
      brick->maxVal = hi;
      if (lo == hi && lo == static_cast<std::uint8_t>(meta.fillValue)) {
        brick->isFillValue = true;
        brick->voxels = ByteBuffer{};  // drop the uniform payload
      }
    }
    if (!ok) {
      logWarn("decode failed for level {} chunk ({},{},{}): {}",
              key.coord.level, key.coord.z, key.coord.y, key.coord.x,
              ok.error().message);
      std::lock_guard lock(mutex_);
      --queuedDecodes_;
      tracked_.erase(key);
      pump();
      return;
    }
  }

  auto stored = ramCache_.insertPinned(std::move(brick));

  std::lock_guard lock(mutex_);
  --queuedDecodes_;
  tracked_.erase(key);
  readyPinnedBytes_ += decodedBytes;
  ready_.push_back(ReadyBrick{std::move(stored)});
  pump();
}

std::vector<ReadyBrick> ChunkFetchPipeline::drainReady() {
  std::lock_guard lock(mutex_);
  return std::exchange(ready_, {});
}

void ChunkFetchPipeline::unpin(const BrickKey& key) {
  ramCache_.unpin(key);
  std::lock_guard lock(mutex_);
  const auto bytes = volume_->level(key.coord.level).chunkBytes();
  readyPinnedBytes_ = readyPinnedBytes_ > bytes ? readyPinnedBytes_ - bytes : 0;
  pump();
}

PipelineStats ChunkFetchPipeline::stats() const {
  std::lock_guard lock(mutex_);
  PipelineStats s;
  s.queued = heap_.size();
  s.inflightFetches = static_cast<std::size_t>(inflightFetches_);
  s.queuedDecodes = static_cast<std::size_t>(queuedDecodes_);
  s.ready = ready_.size();
  s.readyPinnedBytes = readyPinnedBytes_;
  return s;
}

}  // namespace sv::data
