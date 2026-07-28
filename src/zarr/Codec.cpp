#include "zarr/Codec.h"

#include <blosc2.h>
#include <zstd.h>

#include <cstring>

namespace sv::zarr {

namespace {

// Identity codec for compressor: null (e.g. the masked S3 volumes).
class RawCodec final : public Codec {
 public:
  Result<void> decode(std::span<const std::byte> src,
                      std::span<std::byte> dst) const override {
    if (src.size() != dst.size())
      return std::unexpected(Error::fmt(
          "raw chunk size {} != expected {}", src.size(), dst.size()));
    std::memcpy(dst.data(), src.data(), src.size());
    return {};
  }
};

// Decodes blosc1 frames (what numcodecs writes) via the blosc2 context API —
// no global blosc init, fully thread-safe.
class BloscCodec final : public Codec {
 public:
  Result<void> decode(std::span<const std::byte> src,
                      std::span<std::byte> dst) const override {
    if (src.size() < BLOSC_MIN_HEADER_LENGTH)
      return std::unexpected(Error{"blosc chunk shorter than header"});

    std::int32_t nbytes = 0, cbytes = 0, blocksize = 0;
    if (blosc2_cbuffer_sizes(src.data(), &nbytes, &cbytes, &blocksize) < 0)
      return std::unexpected(Error{"invalid blosc header"});
    if (static_cast<std::size_t>(cbytes) > src.size())
      return std::unexpected(Error::fmt(
          "blosc frame truncated: header says {} bytes, have {}", cbytes, src.size()));
    if (static_cast<std::size_t>(nbytes) != dst.size())
      return std::unexpected(Error::fmt(
          "blosc decompressed size {} != expected {}", nbytes, dst.size()));

    blosc2_dparams params = BLOSC2_DPARAMS_DEFAULTS;
    params.nthreads = 1;  // pipeline parallelism comes from the decode pool
    blosc2_context* ctx = blosc2_create_dctx(params);
    if (!ctx) return std::unexpected(Error{"blosc2_create_dctx failed"});
    const int rc = blosc2_decompress_ctx(ctx, src.data(),
                                         static_cast<std::int32_t>(src.size()),
                                         dst.data(),
                                         static_cast<std::int32_t>(dst.size()));
    blosc2_free_ctx(ctx);
    if (rc < 0 || static_cast<std::size_t>(rc) != dst.size())
      return std::unexpected(Error::fmt("blosc decompress failed (rc={})", rc));
    return {};
  }
};

// Bare zstd frames (zarr "zstd" numcodecs codec, no blosc wrapper).
class ZstdCodec final : public Codec {
 public:
  Result<void> decode(std::span<const std::byte> src,
                      std::span<std::byte> dst) const override {
    const std::size_t rc =
        ZSTD_decompress(dst.data(), dst.size(), src.data(), src.size());
    if (ZSTD_isError(rc))
      return std::unexpected(Error::fmt("zstd: {}", ZSTD_getErrorName(rc)));
    if (rc != dst.size())
      return std::unexpected(Error::fmt(
          "zstd decompressed size {} != expected {}", rc, dst.size()));
    return {};
  }
};

}  // namespace

Result<std::shared_ptr<const Codec>> makeCodec(CompressorId id) {
  switch (id) {
    case CompressorId::None:  return std::make_shared<const RawCodec>();
    case CompressorId::Blosc: return std::make_shared<const BloscCodec>();
    case CompressorId::Zstd:  return std::make_shared<const ZstdCodec>();
    case CompressorId::Gzip:
      return std::unexpected(Error{"gzip codec not implemented yet"});
  }
  return std::unexpected(Error{"unknown compressor"});
}

}  // namespace sv::zarr
