#include "store/LocalStore.h"

#include <cstdio>
#include <system_error>

namespace sv::store {

namespace fs = std::filesystem;

LocalStore::LocalStore(fs::path root, std::shared_ptr<ThreadPool> ioPool)
    : root_(std::move(root)), ioPool_(std::move(ioPool)) {}

void LocalStore::read(std::string key, std::stop_token st, StoreCallback cb) {
  ioPool_->post([path = root_ / key, st = std::move(st),
                 cb = std::move(cb)](std::stop_token poolSt) mutable {
    if (st.stop_requested() || poolSt.stop_requested()) {
      cb(std::unexpected(StoreError{StoreError::Kind::Cancelled, 0, {}}));
      return;
    }

    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    if (ec) {
      const auto kind = (ec == std::errc::no_such_file_or_directory)
                            ? StoreError::Kind::NotFound
                            : StoreError::Kind::Io;
      cb(std::unexpected(StoreError{kind, ec.value(), ec.message()}));
      return;
    }

    ByteBuffer buf = ByteBuffer::uninitialized(size);
    std::FILE* f = nullptr;
#ifdef _WIN32
    _wfopen_s(&f, path.c_str(), L"rb");
#else
    f = std::fopen(path.c_str(), "rb");
#endif
    if (!f) {
      cb(std::unexpected(StoreError{StoreError::Kind::Io, errno,
                                    "failed to open file"}));
      return;
    }
    const std::size_t got = std::fread(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    if (got != buf.size()) {
      cb(std::unexpected(
          StoreError{StoreError::Kind::Io, 0, "short read"}));
      return;
    }
    cb(std::move(buf));
  });
}

}  // namespace sv::store
