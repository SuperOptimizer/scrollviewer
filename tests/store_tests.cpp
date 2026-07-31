#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <thread>

#include "core/ThreadPool.h"
#include "store/DiskCacheStore.h"
#include "store/GdctTranscoder.h"
#include "store/LocalStore.h"
#include "support/FakeChunkStore.h"

using namespace sv;
using namespace sv::store;

namespace fs = std::filesystem;

namespace {

// Synchronous convenience wrapper for tests.
StoreResult readSync(ChunkStore& cs, const std::string& key) {
  std::promise<StoreResult> promise;
  auto future = promise.get_future();
  cs.read(key, {}, [&](StoreResult r) { promise.set_value(std::move(r)); });
  return future.get();
}

fs::path freshDir(const char* name) {
  const auto p = fs::temp_directory_path() / name;
  fs::remove_all(p);
  fs::create_directories(p);
  return p;
}

void writeFile(const fs::path& p, std::string_view text) {
  fs::create_directories(p.parent_path());
  std::ofstream f(p, std::ios::binary);
  f << text;
}

std::string asString(const ByteBuffer& b) {
  return std::string(reinterpret_cast<const char*>(b.data()), b.size());
}

}  // namespace

TEST_CASE("LocalStore reads files and reports NotFound") {
  const auto root = freshDir("sv_test_localstore");
  writeFile(root / "0" / "1" / "2" / "3", "chunkdata");
  auto pool = std::make_shared<ThreadPool>("io", 2);
  LocalStore ls(root, pool);

  auto r = readSync(ls, "0/1/2/3");
  REQUIRE(r.has_value());
  CHECK(asString(*r) == "chunkdata");

  auto missing = readSync(ls, "0/9/9/9");
  REQUIRE_FALSE(missing.has_value());
  CHECK(missing.error().isNotFound());
}

TEST_CASE("ChunkStore::readRange default slices whole object") {
  const auto root = freshDir("sv_test_range");
  writeFile(root / "obj", "0123456789");
  auto pool = std::make_shared<ThreadPool>("io", 1);
  LocalStore ls(root, pool);

  std::promise<StoreResult> promise;
  auto future = promise.get_future();
  ls.readRange("obj", 3, 4, {},
               [&](StoreResult r) { promise.set_value(std::move(r)); });
  auto r = future.get();
  REQUIRE(r.has_value());
  CHECK(asString(*r) == "3456");
}

TEST_CASE("DiskCacheStore caches misses and serves hits without inner") {
  const auto cacheRoot = freshDir("sv_test_diskcache");
  auto pool = std::make_shared<ThreadPool>("io", 2);
  auto fake = std::make_shared<test::FakeChunkStore>();
  const std::string payload = "remote-bytes";
  fake->put("vol/0/0/0/0",
            {reinterpret_cast<const std::byte*>(payload.data()),
             reinterpret_cast<const std::byte*>(payload.data()) + payload.size()});

  DiskCacheStore cache(fake, cacheRoot, 1 << 20, pool);

  // Miss: forwarded to inner (which we must release), then cached.
  std::promise<StoreResult> p1;
  auto f1 = p1.get_future();
  cache.read("vol/0/0/0/0", {}, [&](StoreResult r) { p1.set_value(std::move(r)); });
  while (fake->pendingCount() == 0) std::this_thread::yield();
  fake->releaseAll();
  auto r1 = f1.get();
  REQUIRE(r1.has_value());
  CHECK(asString(*r1) == payload);
  CHECK(fs::exists(cacheRoot / "vol/0/0/0/0"));

  // Hit: served from disk; inner never sees the request.
  auto r2 = readSync(cache, "vol/0/0/0/0");
  REQUIRE(r2.has_value());
  CHECK(asString(*r2) == payload);
  CHECK(fake->pendingCount() == 0);

  // NotFound from inner is propagated and not cached.
  std::promise<StoreResult> p3;
  auto f3 = p3.get_future();
  cache.read("vol/0/9/9/9", {}, [&](StoreResult r) { p3.set_value(std::move(r)); });
  while (fake->pendingCount() == 0) std::this_thread::yield();
  fake->releaseAll();
  auto r3 = f3.get();
  REQUIRE_FALSE(r3.has_value());
  CHECK(r3.error().isNotFound());
  CHECK_FALSE(fs::exists(cacheRoot / "vol/0/9/9/9"));
}

TEST_CASE("DiskCacheStore invalidate forces refetch") {
  const auto cacheRoot = freshDir("sv_test_diskcache_inval");
  auto pool = std::make_shared<ThreadPool>("io", 2);
  auto fake = std::make_shared<test::FakeChunkStore>();
  std::string payload = "v1";
  fake->put("k", {reinterpret_cast<const std::byte*>(payload.data()),
                  reinterpret_cast<const std::byte*>(payload.data()) + 2});
  DiskCacheStore cache(fake, cacheRoot, 1 << 20, pool);

  std::promise<StoreResult> p1;
  auto f1 = p1.get_future();
  cache.read("k", {}, [&](StoreResult r) { p1.set_value(std::move(r)); });
  while (fake->pendingCount() == 0) std::this_thread::yield();
  fake->releaseAll();
  REQUIRE(f1.get().has_value());
  REQUIRE(fs::exists(cacheRoot / "k"));

  cache.invalidate("k");
  CHECK_FALSE(fs::exists(cacheRoot / "k"));
}

TEST_CASE("DiskCacheStore eviction trims to budget, oldest first") {
  const auto cacheRoot = freshDir("sv_test_diskcache_evict");
  auto pool = std::make_shared<ThreadPool>("io", 1);
  auto fake = std::make_shared<test::FakeChunkStore>();
  DiskCacheStore cache(fake, cacheRoot, 25, pool);  // tiny budget

  writeFile(cacheRoot / "a", "0123456789");  // 10 bytes each
  writeFile(cacheRoot / "b", "0123456789");
  writeFile(cacheRoot / "c", "0123456789");
  // Make 'a' clearly oldest.
  fs::last_write_time(cacheRoot / "a",
                      fs::file_time_type::clock::now() - std::chrono::hours(2));

  cache.runEviction();  // 30 > 25: evict down to <= 22 (90% of 25)
  CHECK_FALSE(fs::exists(cacheRoot / "a"));
  CHECK(fs::exists(cacheRoot / "b"));
  CHECK(fs::exists(cacheRoot / "c"));
}

TEST_CASE("GdctTranscoder round-trips a 128^3 brick within tolerance") {
  auto t = makeGdctTranscoder(1.0f);
  if (!t) SKIP("built without gpudct");

  constexpr std::size_t kBrick = 128u * 128u * 128u;
  ByteBuffer brick = ByteBuffer::uninitialized(kBrick);
  // Smooth low-frequency content plus mild texture, like real CT.
  for (std::size_t z = 0, i = 0; z < 128; ++z)
    for (std::size_t y = 0; y < 128; ++y)
      for (std::size_t x = 0; x < 128; ++x, ++i)
        brick.data()[i] = std::byte(
            std::uint8_t(96 + 60.0 * std::sin(x * 0.05) * std::cos(y * 0.07) +
                         30.0 * std::sin(z * 0.11) + ((x * 7 + y * 3) % 5)));

  auto enc = t->encode(brick.span());
  REQUIRE(enc.has_value());
  CHECK(enc->size() < kBrick / 4);  // it had better actually compress

  auto dec = t->decode(enc->span());
  REQUIRE(dec.has_value());
  REQUIRE(dec->size() == kBrick);
  double sumErr = 0.0;
  int maxErr = 0;
  for (std::size_t i = 0; i < kBrick; ++i) {
    const int e = std::abs(int(std::uint8_t(brick.data()[i])) -
                           int(std::uint8_t(dec->data()[i])));
    sumErr += e;
    maxErr = std::max(maxErr, e);
  }
  CHECK(sumErr / double(kBrick) < 4.0);
  CHECK(maxErr < 64);

  // Anything that is not exactly one brick stays verbatim.
  CHECK_FALSE(t->encode(brick.span().subspan(0, 1000)).has_value());
  // Garbage never decodes.
  CHECK_FALSE(t->decode(brick.span().subspan(0, 1000)).has_value());
}

TEST_CASE("DiskCacheStore transcodes eligible chunks, serves them on hit") {
  auto t = makeGdctTranscoder(1.0f);
  if (!t) SKIP("built without gpudct");

  const auto cacheRoot = freshDir("sv_test_diskcache_gdct");
  auto pool = std::make_shared<ThreadPool>("io", 2);
  auto fake = std::make_shared<test::FakeChunkStore>();

  constexpr std::size_t kBrick = 128u * 128u * 128u;
  std::vector<std::byte> brick(kBrick);
  for (std::size_t i = 0; i < kBrick; ++i)
    brick[i] = std::byte(std::uint8_t(128 + 100 * std::sin(i * 0.001)));
  fake->put("0/1/2/3", brick);
  const std::string meta = "{\"zarr_format\":2}";
  fake->put(".zarray", {reinterpret_cast<const std::byte*>(meta.data()),
                        reinterpret_cast<const std::byte*>(meta.data()) +
                            meta.size()});

  DiskCacheStore cache(fake, cacheRoot, 1 << 30, pool);
  cache.setTranscoder(t);

  // First read: fetched from the fake, delivered verbatim, encoded to .gdct
  // asynchronously.
  std::promise<StoreResult> p1;
  auto f1 = p1.get_future();
  cache.read("0/1/2/3", {}, [&](StoreResult r) { p1.set_value(std::move(r)); });
  while (fake->pendingCount() == 0) std::this_thread::yield();
  fake->releaseAll();
  auto r1 = f1.get();
  REQUIRE(r1.has_value());
  CHECK(r1->size() == kBrick);

  const auto gdctPath = cacheRoot / "0" / "1" / "2" / "3.gdct";
  for (int i = 0; i < 500 && !fs::exists(gdctPath); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  REQUIRE(fs::exists(gdctPath));
  CHECK_FALSE(fs::exists(cacheRoot / "0" / "1" / "2" / "3"));
  CHECK(fs::file_size(gdctPath) < kBrick / 4);

  // Second read: served from the transcoded entry without touching the inner
  // store, close to the original.
  std::promise<StoreResult> p2;
  auto f2 = p2.get_future();
  cache.read("0/1/2/3", {}, [&](StoreResult r) { p2.set_value(std::move(r)); });
  auto r2 = f2.get();
  REQUIRE(r2.has_value());
  REQUIRE(r2->size() == kBrick);
  CHECK(fake->pendingCount() == 0);
  double sumErr = 0.0;
  for (std::size_t i = 0; i < kBrick; ++i)
    sumErr += std::abs(int(std::uint8_t(brick[i])) -
                       int(std::uint8_t(r2->data()[i])));
  CHECK(sumErr / double(kBrick) < 4.0);

  // Metadata payloads are not brick-shaped: cached verbatim.
  std::promise<StoreResult> p3;
  auto f3 = p3.get_future();
  cache.read(".zarray", {}, [&](StoreResult r) { p3.set_value(std::move(r)); });
  while (fake->pendingCount() == 0) std::this_thread::yield();
  fake->releaseAll();
  REQUIRE(f3.get().has_value());
  const auto metaPath = cacheRoot / ".zarray";
  for (int i = 0; i < 500 && !fs::exists(metaPath); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  CHECK(fs::exists(metaPath));
  CHECK_FALSE(fs::exists(cacheRoot / ".zarray.gdct"));

  // invalidate removes the transcoded entry too.
  cache.invalidate("0/1/2/3");
  CHECK_FALSE(fs::exists(gdctPath));
}
