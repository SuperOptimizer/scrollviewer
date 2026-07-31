#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

#include "data/ChunkFetchPipeline.h"
#include "support/FakeChunkStore.h"
#include "support/ZarrFixture.h"

using namespace sv;
using namespace sv::data;
using namespace std::chrono_literals;

namespace {

// Opens the fixture volume backed by the filesystem for metadata, then wires
// a FakeChunkStore for chunk traffic.
struct PipelineHarness {
  std::shared_ptr<test::FakeChunkStore> fake =
      std::make_shared<test::FakeChunkStore>();
  std::shared_ptr<zarr::OmeZarrVolume> volume;
  RamCache ram{64ull << 20};
  std::unique_ptr<ChunkFetchPipeline> pipeline;

  explicit PipelineHarness(PipelineConfig config = {}) {
    const auto root =
        std::filesystem::temp_directory_path() / "sv_test_pipeline_meta";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    test::ZarrFixtureOptions opts;
    opts.blosc = false;  // raw chunks: fake store payloads are trivial
    test::writeZarrFixture(root, opts);

    auto vol = zarr::OmeZarrVolume::open([&](const std::string& rel)
                                             -> Result<ByteBuffer> {
      std::ifstream f(root / rel, std::ios::binary | std::ios::ate);
      if (!f) return std::unexpected(Error{"not found"});
      const auto size = static_cast<std::size_t>(f.tellg());
      f.seekg(0);
      auto buf = ByteBuffer::uninitialized(size);
      f.read(reinterpret_cast<char*>(buf.data()),
             static_cast<std::streamsize>(size));
      return buf;
    });
    REQUIRE(vol.has_value());
    volume = *vol;

    pipeline = std::make_unique<ChunkFetchPipeline>(fake, volume, 1, ram,
                                                    config);
  }

  BrickKey key(std::uint8_t level, std::uint32_t z, std::uint32_t y,
               std::uint32_t x) const {
    return BrickKey{1, ChunkCoord{level, z, y, x}};
  }

  // Raw chunk payload of the right size for the fixture level.
  std::vector<std::byte> payload(int level, std::byte fill) const {
    return std::vector<std::byte>(volume->level(level).chunkBytes(), fill);
  }

  std::vector<ReadyBrick> waitReady(std::size_t n,
                                    std::chrono::milliseconds timeout = 2s) {
    std::vector<ReadyBrick> got;
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (got.size() < n && std::chrono::steady_clock::now() < deadline) {
      auto batch = pipeline->drainReady();
      got.insert(got.end(), std::make_move_iterator(batch.begin()),
                 std::make_move_iterator(batch.end()));
      if (got.size() < n) std::this_thread::sleep_for(1ms);
    }
    return got;
  }
};

}  // namespace

TEST_CASE("pipeline fetches, decodes, and pins ready bricks") {
  PipelineHarness h;
  h.fake->put("0/0/0/0", h.payload(0, std::byte{7}));

  h.pipeline->submit(std::array{BrickRequest{h.key(0, 0, 0, 0), 1.0f}});
  while (h.fake->pendingCount() == 0) std::this_thread::yield();
  h.fake->releaseAll();

  auto ready = h.waitReady(1);
  REQUIRE(ready.size() == 1);
  CHECK(ready[0].brick->key == h.key(0, 0, 0, 0));
  CHECK_FALSE(ready[0].brick->isFillValue);
  CHECK(ready[0].brick->voxels.span()[0] == std::byte{7});
  CHECK(h.ram.contains(h.key(0, 0, 0, 0)));
  h.pipeline->unpin(ready[0].brick->key);
}

TEST_CASE("missing chunks become fill-value bricks") {
  PipelineHarness h;
  h.pipeline->submit(std::array{BrickRequest{h.key(0, 1, 1, 1), 1.0f}});
  while (h.fake->pendingCount() == 0) std::this_thread::yield();
  h.fake->releaseAll();

  auto ready = h.waitReady(1);
  REQUIRE(ready.size() == 1);
  CHECK(ready[0].brick->isFillValue);
  // Fill bricks carry no payload; consumers synthesize the fill value.
  CHECK(ready[0].brick->voxels.empty());
  h.pipeline->unpin(ready[0].brick->key);
}

TEST_CASE("higher priority requests are issued first under a fetch cap") {
  PipelineConfig config;
  config.maxInflightFetches = 1;
  PipelineHarness h(config);
  h.fake->put("0/0/0/0", h.payload(0, std::byte{1}));
  h.fake->put("0/0/0/1", h.payload(0, std::byte{2}));
  h.fake->put("0/0/1/0", h.payload(0, std::byte{3}));

  h.pipeline->submit(std::array{
      BrickRequest{h.key(0, 0, 0, 0), 0.1f},
      BrickRequest{h.key(0, 0, 0, 1), 0.9f},
      BrickRequest{h.key(0, 0, 1, 0), 0.5f},
  });

  // Only one in flight; it must be the highest-priority key.
  while (h.fake->pendingCount() == 0) std::this_thread::yield();
  CHECK(h.fake->pendingKeys() == std::vector<std::string>{"0/0/0/1"});
  h.fake->releaseOne();

  while (h.fake->pendingCount() == 0) std::this_thread::yield();
  CHECK(h.fake->pendingKeys() == std::vector<std::string>{"0/0/1/0"});
  h.fake->releaseAll();
  while (h.fake->pendingCount() == 0) std::this_thread::yield();
  h.fake->releaseAll();

  auto ready = h.waitReady(3);
  CHECK(ready.size() == 3);
  for (auto& r : ready) h.pipeline->unpin(r.brick->key);
}

TEST_CASE("resubmit reprioritizes queued requests") {
  PipelineConfig config;
  config.maxInflightFetches = 1;
  PipelineHarness h(config);
  h.fake->put("0/0/0/0", h.payload(0, std::byte{1}));
  h.fake->put("0/0/0/1", h.payload(0, std::byte{2}));
  h.fake->put("0/0/1/0", h.payload(0, std::byte{3}));

  // First submit: a occupies the single fetch slot; b and c stay queued with
  // b ahead of c.
  h.pipeline->submit(std::array{
      BrickRequest{h.key(0, 0, 0, 0), 1.0f},
      BrickRequest{h.key(0, 0, 0, 1), 0.8f},
      BrickRequest{h.key(0, 0, 1, 0), 0.2f},
  });
  while (h.fake->pendingCount() == 0) std::this_thread::yield();

  // Camera moved: c now beats b.
  h.pipeline->submit(std::array{
      BrickRequest{h.key(0, 0, 0, 0), 1.0f},
      BrickRequest{h.key(0, 0, 0, 1), 0.3f},
      BrickRequest{h.key(0, 0, 1, 0), 0.9f},
  });

  h.fake->releaseOne();  // completes a; frees the slot
  while (h.fake->pendingCount() == 0) std::this_thread::yield();
  CHECK(h.fake->pendingKeys() == std::vector<std::string>{"0/0/1/0"});

  h.fake->releaseAll();
  while (h.fake->pendingCount() == 0) std::this_thread::yield();
  h.fake->releaseAll();
  auto ready = h.waitReady(3);
  CHECK(ready.size() == 3);
  for (auto& r : ready) h.pipeline->unpin(r.brick->key);
}

TEST_CASE("requests absent from newer submits age out before IO") {
  PipelineConfig config;
  config.maxInflightFetches = 1;
  PipelineHarness h(config);
  h.fake->put("0/0/0/0", h.payload(0, std::byte{1}));
  h.fake->put("0/0/0/1", h.payload(0, std::byte{2}));

  h.pipeline->submit(std::array{
      BrickRequest{h.key(0, 0, 0, 0), 1.0f},
      BrickRequest{h.key(0, 0, 0, 1), 0.5f},  // will be abandoned
  });
  while (h.fake->pendingCount() == 0) std::this_thread::yield();

  // Two newer submits without the second key: it exceeds the stale window.
  h.pipeline->submit(std::array{BrickRequest{h.key(0, 0, 0, 0), 1.0f}});
  h.pipeline->submit(std::array{BrickRequest{h.key(0, 0, 0, 0), 1.0f}});

  h.fake->releaseOne();  // completes the in-flight first key
  auto ready = h.waitReady(1);
  REQUIRE(ready.size() == 1);
  CHECK(ready[0].brick->key == h.key(0, 0, 0, 0));
  h.pipeline->unpin(ready[0].brick->key);

  // The abandoned key must never reach the store.
  std::this_thread::sleep_for(50ms);
  CHECK(h.fake->pendingCount() == 0);
  CHECK(h.pipeline->stats().queued == 0);
}

TEST_CASE("in-flight fetches for unwanted keys are stop-requested") {
  PipelineHarness h;
  h.fake->put("0/0/0/0", h.payload(0, std::byte{1}));

  h.pipeline->submit(std::array{BrickRequest{h.key(0, 0, 0, 0), 1.0f}});
  while (h.fake->pendingCount() == 0) std::this_thread::yield();

  // New submit no longer wants it; the pending request's token flips.
  h.pipeline->submit(std::array{BrickRequest{h.key(0, 1, 0, 0), 1.0f}});
  h.fake->releaseAll();  // FakeChunkStore honors stop -> Cancelled

  std::this_thread::sleep_for(50ms);
  CHECK_FALSE(h.ram.contains(h.key(0, 0, 0, 0)));
}

#include "data/Snic3D.h"

TEST_CASE("snic3d clusters a two-phase volume cleanly") {
  // 32^3: left half dim (60), right half bright (200), z<4 masked to 0.
  const std::array<std::uint32_t, 3> dims{32, 32, 32};
  std::vector<std::uint8_t> vol(32u * 32u * 32u);
  for (std::uint32_t z = 0; z < 32; ++z)
    for (std::uint32_t y = 0; y < 32; ++y)
      for (std::uint32_t x = 0; x < 32; ++x)
        vol[(std::size_t{z} * 32 + y) * 32 + x] =
            (z < 4) ? 0 : (x < 16 ? 60 : 200);

  auto cs = sv::data::snic3d(vol, dims, 1.0 / 512.0, 0.12f);  // ~8^3 cells
  REQUIRE_FALSE(cs.empty());

  std::uint64_t total = 0;
  for (const auto& c : cs) {
    total += c.count;
    // No cluster may straddle the intensity boundary: mean stays near one
    // of the two phases.
    const bool dim = std::abs(c.mean - 60.f / 255.f) < 0.08f;
    const bool bright = std::abs(c.mean - 200.f / 255.f) < 0.08f;
    CHECK((dim || bright));
    // Centroids stay on their side.
    if (dim) CHECK(c.x < 16.5f);
    if (bright) CHECK(c.x > 15.5f);
  }
  // Every non-zero voxel is claimed by exactly one cluster.
  CHECK(total == std::uint64_t(32 - 4) * 32 * 32);
}

#include <cstdio>
#include <future>

#include "store/DiskCacheStore.h"
#include "store/HttpStore.h"
#include "zarr/OmeZarrVolume.h"

// Manual diagnostic: run 3D SNIC over the real PHerc0500P2 level 5 (through
// the shared disk cache) and print the cluster size distribution.
TEST_CASE("snic3d distribution on real scroll level", "[.seg]") {
  const std::string url =
      "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0500P2/"
      "volumes/20250821110041-0.500um-masked.zarr";
  auto pool = std::make_shared<sv::ThreadPool>("io", 4);
  auto store = std::make_shared<sv::store::HttpStore>(url);

  auto readSyncK = [&](const std::string& key) -> sv::store::StoreResult {
    std::promise<sv::store::StoreResult> pr;
    auto fu = pr.get_future();
    store->read(key, {}, [&](sv::store::StoreResult r) { pr.set_value(std::move(r)); });
    return fu.get();
  };
  auto vol = sv::zarr::OmeZarrVolume::open([&](const std::string& rel)
                                               -> sv::Result<sv::ByteBuffer> {
    auto r = readSyncK(rel);
    if (!r) return std::unexpected(sv::Error{"read failed"});
    return std::move(*r);
  });
  REQUIRE(vol.has_value());
  const int L = 5;
  const auto meta = (*vol)->level(L);
  const std::array<std::uint32_t, 3> dims{std::uint32_t(meta.shape[0]),
                                          std::uint32_t(meta.shape[1]),
                                          std::uint32_t(meta.shape[2])};
  std::vector<std::uint8_t> v(std::size_t{dims[0]} * dims[1] * dims[2], 0);
  const auto grid = meta.chunkGridDims();
  std::vector<std::byte> chunk(meta.chunkBytes());
  for (std::uint32_t cz = 0; cz < grid[0]; ++cz)
    for (std::uint32_t cy = 0; cy < grid[1]; ++cy)
      for (std::uint32_t cx = 0; cx < grid[2]; ++cx) {
        auto r = readSyncK((*vol)->chunkStoreKey(L, cz, cy, cx));
        if (!r) continue;
        if (!(*vol)->codec(L).decode(r->span(), chunk)) continue;
        const std::uint32_t z0 = cz * meta.chunks[0], y0 = cy * meta.chunks[1],
                            x0 = cx * meta.chunks[2];
        const std::uint32_t zn = std::min(meta.chunks[0], dims[0] - z0),
                            yn = std::min(meta.chunks[1], dims[1] - y0),
                            xn = std::min(meta.chunks[2], dims[2] - x0);
        for (std::uint32_t z = 0; z < zn; ++z)
          for (std::uint32_t y = 0; y < yn; ++y)
            std::memcpy(
                &v[(std::size_t{z0 + z} * dims[1] + y0 + y) * dims[2] + x0],
                &chunk[(std::size_t{z} * meta.chunks[1] + y) * meta.chunks[2]],
                xn);
      }
  const std::uint8_t thr = sv::data::otsuThreshold(v);
  std::size_t nonzero = 0;
  for (auto b : v) nonzero += b > thr;
  std::printf("otsu threshold %u\n", unsigned(thr));
  std::printf("volume %ux%ux%u, nonzero %zu (%.1f%%)\n", dims[2], dims[1],
              dims[0], nonzero,
              100.0 * double(nonzero) / double(v.size()));

  auto cs = sv::data::snic3d(v, dims, 0.001, 0.12f, thr);
  std::vector<std::uint64_t> sizes;
  for (auto& c : cs) sizes.push_back(c.count);
  std::sort(sizes.begin(), sizes.end());
  std::uint64_t tot = 0;
  for (auto s : sizes) tot += s;
  auto pct = [&](double p) {
    return sizes.empty() ? 0ull : sizes[std::size_t(p * (sizes.size() - 1))];
  };
  std::printf(
      "clusters %zu, voxels claimed %llu / %zu\n"
      "size p10 %llu p50 %llu p90 %llu p99 %llu max %llu (step^3 = 1000)\n",
      cs.size(), (unsigned long long)tot, nonzero,
      (unsigned long long)pct(.10), (unsigned long long)pct(.50),
      (unsigned long long)pct(.90), (unsigned long long)pct(.99),
      (unsigned long long)sizes.back());
}

TEST_CASE("otsuThreshold splits a bimodal histogram") {
  // 70% background at ~103, 30% material at ~150.
  std::vector<std::uint8_t> v;
  for (int i = 0; i < 7000; ++i) v.push_back(std::uint8_t(100 + i % 7));
  for (int i = 0; i < 3000; ++i) v.push_back(std::uint8_t(145 + i % 11));
  const auto t = sv::data::otsuThreshold(v);
  CHECK(t >= 106);
  CHECK(t < 145);

  // Degenerate: uniform volume stays sane.
  std::vector<std::uint8_t> flat(1000, 42);
  CHECK(sv::data::otsuThreshold(flat) <= 42);
}

TEST_CASE("clustersToWorldSpheres places and sizes spheres correctly") {
  // A cluster centered at voxel (z=10, y=20, x=30) of a level-5 array with
  // 4189 voxels (equivalent sphere radius ~10 voxels).
  std::vector<sv::data::SnicCluster> cs{{10.f, 20.f, 30.f, 0.5f, 4189},
                                        {1.f, 1.f, 1.f, 0.9f, 1}};
  auto ws = sv::data::clustersToWorldSpheres(cs, 32.f, 2);
  REQUIRE(ws.size() == 1);  // the count-1 cluster is dropped
  CHECK(ws[0].x == Catch::Approx((30.f + 0.5f) * 32.f));
  CHECK(ws[0].y == Catch::Approx((20.f + 0.5f) * 32.f));
  CHECK(ws[0].z == Catch::Approx((10.f + 0.5f) * 32.f));
  CHECK(ws[0].radius == Catch::Approx(10.0f * 32.f).margin(0.5f * 32.f));
  CHECK(ws[0].value == 0.5f);
}
