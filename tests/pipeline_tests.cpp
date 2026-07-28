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
