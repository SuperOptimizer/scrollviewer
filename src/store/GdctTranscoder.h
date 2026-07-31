#pragma once

#include <memory>

#include "store/DiskCacheStore.h"

namespace sv::store {

// Recompresses raw uncompressed zarr chunks in the disk cache with gpudct
// (lossy 3D-DCT; ~12-20x smaller at visually transparent quality). Only
// payloads that are exactly one 128^3 u8 brick are encoded — everything else
// (metadata, edge cases, foreign sizes) is stored verbatim. Callers must
// verify the volume really is uncompressed u8 with 128^3 chunks before
// installing this: a lossy round-trip of anything else corrupts it.
//
// `quality` scales the quantizer: 0.5 ~ 19x / 38 dB PSNR, 1.0 ~ 12x / 42 dB,
// 2.0 ~ 8x / 46 dB on full-resolution scroll bricks.
//
// Returns nullptr when built without gpudct.
std::shared_ptr<CacheTranscoder> makeGdctTranscoder(float quality);

}  // namespace sv::store
