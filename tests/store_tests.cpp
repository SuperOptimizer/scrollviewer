#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <future>

#include "core/ThreadPool.h"
#include "store/DiskCacheStore.h"
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
