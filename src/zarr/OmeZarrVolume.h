#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "core/ByteBuffer.h"
#include "core/Result.h"
#include "zarr/ArrayMeta.h"
#include "zarr/Codec.h"
#include "zarr/OmeMultiscale.h"

namespace sv::zarr {

// Multiscale OME-zarr volume model. Layering: sv_zarr knows nothing about
// storage backends; open() takes a synchronous reader for the handful of tiny
// metadata files (callers adapt a ChunkStore or plain file reads).
class OmeZarrVolume {
 public:
  using MetaReader =
      std::function<Result<ByteBuffer>(const std::string& relPath)>;

  static Result<std::shared_ptr<OmeZarrVolume>> open(const MetaReader& read);

  int levelCount() const { return static_cast<int>(levels_.size()); }
  const ArrayMeta& level(int i) const { return levels_[i].meta; }
  // Isotropic scale factor of level i vs level 0 (2^i for Vesuvius pyramids).
  double levelScale(int i) const { return levels_[i].scale; }
  const Codec& codec(int i) const { return *levels_[i].codec; }

  // Store key of a chunk, relative to the volume root: "<level>/<chunk key>".
  std::string chunkStoreKey(int level, std::uint32_t z, std::uint32_t y,
                            std::uint32_t x) const;

 private:
  struct Level {
    std::string path;  // dataset path, e.g. "0"
    ArrayMeta meta;
    double scale = 1.0;
    std::shared_ptr<const Codec> codec;
  };
  std::vector<Level> levels_;
};

}  // namespace sv::zarr
