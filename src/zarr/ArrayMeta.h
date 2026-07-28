#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

#include "core/Result.h"
#include "zarr/Dtype.h"

namespace sv::zarr {

enum class CompressorId { None, Blosc, Zstd, Gzip };

// Parsed zarr v2 .zarray for one multiscale level. Axis order is z,y,x
// (index 0..2) matching the on-disk layout of Vesuvius volumes.
struct ArrayMeta {
  std::array<std::uint64_t, 3> shape{};   // z,y,x voxels
  std::array<std::uint32_t, 3> chunks{};  // chunk shape
  Dtype dtype = Dtype::U8;
  char dimensionSeparator = '.';
  double fillValue = 0.0;
  CompressorId compressor = CompressorId::None;
  int zarrFormat = 2;

  std::array<std::uint32_t, 3> chunkGridDims() const {
    return {static_cast<std::uint32_t>((shape[0] + chunks[0] - 1) / chunks[0]),
            static_cast<std::uint32_t>((shape[1] + chunks[1] - 1) / chunks[1]),
            static_cast<std::uint32_t>((shape[2] + chunks[2] - 1) / chunks[2])};
  }

  std::size_t chunkVoxels() const {
    return std::size_t{chunks[0]} * chunks[1] * chunks[2];
  }

  std::size_t chunkBytes() const { return chunkVoxels() * dtypeSize(dtype); }

  // Relative chunk key, e.g. "12.3.4" (flat) or "12/3/4" (nested).
  std::string chunkKey(std::uint32_t z, std::uint32_t y, std::uint32_t x) const;
};

// Strict parse; unsupported layouts (order "F", filters, exotic dtypes or
// compressors) are errors, never silent misreads.
Result<ArrayMeta> parseZarrayV2(std::string_view json);

}  // namespace sv::zarr
