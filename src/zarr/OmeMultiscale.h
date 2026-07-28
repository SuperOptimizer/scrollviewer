#pragma once

#include <array>
#include <string>
#include <vector>

#include "core/Result.h"

namespace sv::zarr {

// One entry of the OME-zarr "multiscales[0].datasets" list.
struct MultiscaleDataset {
  std::string path;                       // e.g. "0", "1", ...
  std::array<double, 3> scale{1, 1, 1};   // z,y,x scale vs level 0
};

struct OmeMultiscale {
  std::vector<std::string> axisNames;     // expected z,y,x
  std::vector<MultiscaleDataset> datasets;
};

// Parses OME-zarr v0.4-style .zattrs. Tolerates missing "axes" (older data)
// by assuming z,y,x.
Result<OmeMultiscale> parseOmeAttrs(std::string_view json);

}  // namespace sv::zarr
