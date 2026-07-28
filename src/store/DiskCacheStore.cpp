#include "store/DiskCacheStore.h"

#include <algorithm>
#include <cstdio>
#include <system_error>
#include <vector>

#include "core/Log.h"

namespace sv::store {

namespace fs = std::filesystem;

namespace {

bool writeFileAtomic(const fs::path& target, std::span<const std::byte> data) {
  std::error_code ec;
  fs::create_directories(target.parent_path(), ec);
  const fs::path tmp = target.string() + ".tmp";
  {
    std::FILE* f = nullptr;
#ifdef _WIN32
    _wfopen_s(&f, tmp.c_str(), L"wb");
#else
    f = std::fopen(tmp.c_str(), "wb");
#endif
    if (!f) return false;
    const std::size_t wrote = std::fwrite(data.data(), 1, data.size(), f);
    std::fclose(f);
    if (wrote != data.size()) {
      fs::remove(tmp, ec);
      return false;
    }
  }
  fs::rename(tmp, target, ec);
  if (ec) {
    // Lost a race with a concurrent writer of the same key: their copy is as
    // good as ours.
    fs::remove(tmp, ec);
    return fs::exists(target);
  }
  return true;
}

}  // namespace

DiskCacheStore::DiskCacheStore(std::shared_ptr<ChunkStore> inner,
                               fs::path cacheRoot, std::uint64_t byteBudget,
                               std::shared_ptr<ThreadPool> ioPool)
    : inner_(std::move(inner)),
      root_(std::move(cacheRoot)),
      budget_(byteBudget),
      ioPool_(std::move(ioPool)) {}

fs::path DiskCacheStore::pathFor(const std::string& key) const {
  return root_ / key;
}

void DiskCacheStore::read(std::string key, std::stop_token st,
                          StoreCallback cb) {
  ioPool_->post([this, key = std::move(key), st = std::move(st),
                 cb = std::move(cb)](std::stop_token) mutable {
    if (st.stop_requested()) {
      cb(std::unexpected(StoreError{StoreError::Kind::Cancelled, 0, {}}));
      return;
    }

    // Cache hit path.
    const fs::path p = pathFor(key);
    std::error_code ec;
    const auto size = fs::file_size(p, ec);
    if (!ec) {
      ByteBuffer buf = ByteBuffer::uninitialized(size);
      std::FILE* f = nullptr;
#ifdef _WIN32
      _wfopen_s(&f, p.c_str(), L"rb");
#else
      f = std::fopen(p.c_str(), "rb");
#endif
      if (f) {
        const std::size_t got = std::fread(buf.data(), 1, buf.size(), f);
        std::fclose(f);
        if (got == buf.size()) {
          cb(std::move(buf));
          return;
        }
      }
      // Unreadable/short entry: fall through to refetch.
      fs::remove(p, ec);
    }

    // Miss: forward to remote; persist on success.
    inner_->read(key, st,
                 [this, key, cb = std::move(cb)](StoreResult r) mutable {
                   if (r) {
                     if (!writeFileAtomic(pathFor(key), r->span()))
                       logWarn("disk cache write failed for {}", key);
                   }
                   cb(std::move(r));
                 });
  });
}

void DiskCacheStore::invalidate(const std::string& key) {
  std::error_code ec;
  fs::remove(pathFor(key), ec);
}

void DiskCacheStore::runEviction() {
  struct FileInfo {
    fs::path path;
    fs::file_time_type mtime;
    std::uint64_t size;
  };
  std::vector<FileInfo> files;
  std::uint64_t total = 0;
  std::error_code ec;
  for (auto it = fs::recursive_directory_iterator(
           root_, fs::directory_options::skip_permission_denied, ec);
       it != fs::recursive_directory_iterator(); it.increment(ec)) {
    if (ec) break;
    if (!it->is_regular_file(ec)) continue;
    const auto size = it->file_size(ec);
    if (ec) continue;
    total += size;
    files.push_back({it->path(), it->last_write_time(ec), size});
  }
  if (total <= budget_) return;

  std::ranges::sort(files, {}, &FileInfo::mtime);  // oldest first
  const std::uint64_t target = budget_ * 9 / 10;
  std::uint64_t freed = 0;
  for (const auto& f : files) {
    if (total - freed <= target) break;
    if (fs::remove(f.path, ec)) freed += f.size;
  }
  logInfo("disk cache eviction freed {} MB", freed >> 20);
}

}  // namespace sv::store
