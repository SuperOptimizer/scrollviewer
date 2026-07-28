#pragma once

#include <cstddef>
#include <string_view>

#include "core/Result.h"

namespace sv::zarr {

// v1 supports the dtypes found in Vesuvius volumes. Everything else is
// rejected loudly at parse time.
enum class Dtype { U8, U16LE, U16BE };

constexpr std::size_t dtypeSize(Dtype d) {
  return d == Dtype::U8 ? 1 : 2;
}

// numpy typestr as used by zarr v2, e.g. "|u1", "<u2", ">u2".
Result<Dtype> parseDtypeV2(std::string_view typestr);

}  // namespace sv::zarr
