#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <latch>

#include "core/ByteBuffer.h"
#include "core/LruCache.h"
#include "core/ThreadPool.h"

using namespace sv;

TEST_CASE("ByteBuffer basic ops") {
  auto b = ByteBuffer::uninitialized(16);
  REQUIRE(b.size() == 16);
  b.span()[0] = std::byte{42};
  auto c = ByteBuffer::copyOf(b.span());
  CHECK(c.size() == 16);
  CHECK(c.span()[0] == std::byte{42});
  c.truncate(4);
  CHECK(c.size() == 4);
  c.truncate(100);  // never grows
  CHECK(c.size() == 4);
}

TEST_CASE("ThreadPool runs jobs and shuts down cleanly") {
  std::atomic<int> count = 0;
  {
    ThreadPool pool("test", 4);
    std::latch done(100);
    for (int i = 0; i < 100; ++i)
      pool.post([&](std::stop_token) {
        ++count;
        done.count_down();
      });
    done.wait();
  }
  CHECK(count == 100);
}

TEST_CASE("LruCache evicts least-recently-used by bytes") {
  LruCache<int, int> cache(100);
  for (int i = 0; i < 10; ++i)
    cache.insert(i, std::make_shared<int>(i), 10);
  CHECK(cache.bytesUsed() == 100);

  cache.get(0);                                   // 0 now most recent
  cache.insert(10, std::make_shared<int>(10), 10);  // evicts 1 (LRU)
  CHECK(cache.get(1) == nullptr);
  CHECK(cache.get(0) != nullptr);
  CHECK(cache.bytesUsed() == 100);
}

TEST_CASE("LruCache pinning blocks eviction") {
  LruCache<int, int> cache(30);
  cache.insert(1, std::make_shared<int>(1), 10, /*pins=*/1);
  cache.insert(2, std::make_shared<int>(2), 10);
  cache.insert(3, std::make_shared<int>(3), 10);
  cache.insert(4, std::make_shared<int>(4), 10);  // over budget: evicts 2
  // contains() does not touch LRU order; get() would promote key 1 to MRU
  // and change which entry the next eviction picks.
  CHECK(cache.contains(1));                       // pinned survived
  CHECK_FALSE(cache.contains(2));

  cache.unpin(1);
  cache.insert(5, std::make_shared<int>(5), 10);  // now 1 is evictable LRU
  CHECK(cache.get(1) == nullptr);
}

TEST_CASE("LruCache goes over budget rather than evicting pinned") {
  LruCache<int, int> cache(20);
  cache.insert(1, std::make_shared<int>(1), 10, 1);
  cache.insert(2, std::make_shared<int>(2), 10, 1);
  cache.insert(3, std::make_shared<int>(3), 10, 1);
  CHECK(cache.bytesUsed() == 30);  // all pinned, none evicted
  CHECK(cache.get(1) != nullptr);
  CHECK(cache.get(2) != nullptr);
  CHECK(cache.get(3) != nullptr);
}

TEST_CASE("LruCache value survives eviction via shared_ptr") {
  LruCache<int, int> cache(10);
  auto held = cache.insert(1, std::make_shared<int>(41), 10);
  cache.insert(2, std::make_shared<int>(42), 10);  // evicts 1
  CHECK(cache.get(1) == nullptr);
  CHECK(*held == 41);  // still alive for the holder
}
