#pragma once

#include <memory>
#include <span>

#include "core/Result.h"
#include "zarr/ArrayMeta.h"

namespace sv::zarr {

// Stateless, thread-safe chunk decoder.
class Codec {
 public:
  virtual ~Codec() = default;

  // dst must be sized to the exact expected decompressed byte count
  // (chunkVoxels * dtypeSize). Fails on size mismatch or corrupt input.
  virtual Result<void> decode(std::span<const std::byte> src,
                              std::span<std::byte> dst) const = 0;
};

// Factory for the compressor declared in .zarray.
Result<std::shared_ptr<const Codec>> makeCodec(CompressorId id);

}  // namespace sv::zarr
