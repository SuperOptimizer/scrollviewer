#include "zarr/ArrayMeta.h"

#include <nlohmann/json.hpp>

namespace sv::zarr {

using nlohmann::json;

Result<Dtype> parseDtypeV2(std::string_view typestr) {
  if (typestr == "|u1") return Dtype::U8;
  if (typestr == "<u2") return Dtype::U16LE;
  if (typestr == ">u2") return Dtype::U16BE;
  return std::unexpected(Error::fmt("unsupported dtype '{}'", typestr));
}

std::string ArrayMeta::chunkKey(std::uint32_t z, std::uint32_t y,
                                std::uint32_t x) const {
  const char s = dimensionSeparator;
  return std::to_string(z) + s + std::to_string(y) + s + std::to_string(x);
}

namespace {

Result<std::array<std::uint64_t, 3>> parseDims3(const json& arr,
                                                const char* field) {
  if (!arr.is_array() || arr.size() != 3)
    return std::unexpected(Error::fmt("'{}' must be a 3-element array", field));
  std::array<std::uint64_t, 3> out{};
  for (int i = 0; i < 3; ++i) {
    if (!arr[i].is_number_unsigned() || arr[i].get<std::uint64_t>() == 0)
      return std::unexpected(Error::fmt("'{}[{}]' must be a positive integer", field, i));
    out[i] = arr[i].get<std::uint64_t>();
  }
  return out;
}

Result<CompressorId> parseCompressor(const json& c) {
  if (c.is_null()) return CompressorId::None;
  if (!c.is_object() || !c.contains("id") || !c["id"].is_string())
    return std::unexpected(Error{"'compressor' must be null or an object with string 'id'"});
  const std::string id = c["id"].get<std::string>();
  if (id == "blosc") return CompressorId::Blosc;
  if (id == "zstd") return CompressorId::Zstd;
  if (id == "gzip") return CompressorId::Gzip;
  return std::unexpected(Error::fmt("unsupported compressor '{}'", id));
}

}  // namespace

Result<ArrayMeta> parseZarrayV2(std::string_view json_text) {
  const json j = json::parse(json_text, nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded())
    return std::unexpected(Error{".zarray is not valid JSON"});
  if (!j.is_object())
    return std::unexpected(Error{".zarray must be a JSON object"});

  ArrayMeta m;

  if (!j.contains("zarr_format") || !j["zarr_format"].is_number_integer() ||
      j["zarr_format"].get<int>() != 2)
    return std::unexpected(Error{"only zarr_format 2 is supported"});
  m.zarrFormat = 2;

  if (!j.contains("shape"))
    return std::unexpected(Error{"missing 'shape'"});
  auto shape = parseDims3(j["shape"], "shape");
  if (!shape) return std::unexpected(shape.error());
  m.shape = *shape;

  if (!j.contains("chunks"))
    return std::unexpected(Error{"missing 'chunks'"});
  auto chunks = parseDims3(j["chunks"], "chunks");
  if (!chunks) return std::unexpected(chunks.error());
  for (int i = 0; i < 3; ++i) {
    if ((*chunks)[i] > 4096)
      return std::unexpected(Error::fmt("chunk dim {} too large ({})", i, (*chunks)[i]));
    m.chunks[i] = static_cast<std::uint32_t>((*chunks)[i]);
  }

  if (!j.contains("dtype") || !j["dtype"].is_string())
    return std::unexpected(Error{"missing 'dtype'"});
  auto dt = parseDtypeV2(j["dtype"].get<std::string>());
  if (!dt) return std::unexpected(dt.error());
  m.dtype = *dt;

  if (!j.contains("order") || !j["order"].is_string() ||
      j["order"].get<std::string>() != "C")
    return std::unexpected(Error{"only order 'C' is supported"});

  if (j.contains("filters") && !j["filters"].is_null()) {
    if (!j["filters"].is_array() || !j["filters"].empty())
      return std::unexpected(Error{"filters are not supported"});
  }

  if (j.contains("fill_value") && j["fill_value"].is_number())
    m.fillValue = j["fill_value"].get<double>();

  if (!j.contains("compressor"))
    return std::unexpected(Error{"missing 'compressor'"});
  auto comp = parseCompressor(j["compressor"]);
  if (!comp) return std::unexpected(comp.error());
  m.compressor = *comp;

  if (j.contains("dimension_separator")) {
    const auto& ds = j["dimension_separator"];
    if (!ds.is_string() ||
        (ds.get<std::string>() != "." && ds.get<std::string>() != "/"))
      return std::unexpected(Error{"dimension_separator must be '.' or '/'"});
    m.dimensionSeparator = ds.get<std::string>()[0];
  }

  return m;
}

}  // namespace sv::zarr
