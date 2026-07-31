#include "store/GdctTranscoder.h"

#ifdef SV_HAVE_GPUDCT

#include <cstring>

#include <gpudct/gpudct.hpp>

#include "core/Log.h"

namespace sv::store {

namespace {

constexpr std::size_t kBrickBytes =
    std::size_t{gpudct::kBrickDim} * gpudct::kBrickDim * gpudct::kBrickDim;

class GdctTranscoder final : public CacheTranscoder {
 public:
  explicit GdctTranscoder(float quality) {
    opts_.quality = quality;
    // One brick per archive, decoded on an IO-pool worker: the pool is the
    // parallelism, not the codec.
    opts_.threads = 1;
  }

  std::optional<ByteBuffer> encode(std::span<const std::byte> raw) override {
    if (raw.size() != kBrickBytes) return std::nullopt;
    std::vector<std::uint8_t> out;
    const auto st = gpudct::encode(
        raw.data(), {gpudct::kBrickDim, gpudct::kBrickDim, gpudct::kBrickDim},
        gpudct::DType::u8, opts_, out, gpudct::Backend::cpu_simd);
    if (st != gpudct::Status::ok) {
      logWarn("gdct encode failed: {}", gpudct::status_message(st));
      return std::nullopt;
    }
    return ByteBuffer::copyOf(
        {reinterpret_cast<const std::byte*>(out.data()), out.size()});
  }

  std::optional<ByteBuffer> decode(std::span<const std::byte> stored) override {
    const std::span<const std::uint8_t> archive{
        reinterpret_cast<const std::uint8_t*>(stored.data()), stored.size()};
    gpudct::VolumeInfo info;
    if (gpudct::inspect(archive, info) != gpudct::Status::ok) return std::nullopt;
    if (info.dims.voxels() * dtype_size(info.dtype) != kBrickBytes)
      return std::nullopt;
    ByteBuffer out = ByteBuffer::uninitialized(kBrickBytes);
    gpudct::DecodeOptions dopts;
    dopts.threads = 1;
    const auto st = gpudct::decode_into(
        archive, dopts,
        {reinterpret_cast<std::uint8_t*>(out.data()), out.size()}, info,
        gpudct::Backend::cpu_simd);
    if (st != gpudct::Status::ok) return std::nullopt;
    return out;
  }

  const char* suffix() const override { return ".gdct"; }

 private:
  gpudct::EncodeOptions opts_;
};

}  // namespace

std::shared_ptr<CacheTranscoder> makeGdctTranscoder(float quality) {
  return std::make_shared<GdctTranscoder>(quality);
}

}  // namespace sv::store

#else  // !SV_HAVE_GPUDCT

namespace sv::store {

std::shared_ptr<CacheTranscoder> makeGdctTranscoder(float) { return nullptr; }

}  // namespace sv::store

#endif
