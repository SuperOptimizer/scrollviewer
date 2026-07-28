#include "zarr/OmeMultiscale.h"

#include <nlohmann/json.hpp>

namespace sv::zarr {

using nlohmann::json;

Result<OmeMultiscale> parseOmeAttrs(std::string_view json_text) {
  const json j = json::parse(json_text, nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded())
    return std::unexpected(Error{".zattrs is not valid JSON"});

  if (!j.contains("multiscales") || !j["multiscales"].is_array() ||
      j["multiscales"].empty())
    return std::unexpected(Error{".zattrs has no 'multiscales'"});
  const json& ms = j["multiscales"][0];

  OmeMultiscale out;

  if (ms.contains("axes") && ms["axes"].is_array()) {
    for (const auto& ax : ms["axes"]) {
      if (ax.is_object() && ax.contains("name") && ax["name"].is_string())
        out.axisNames.push_back(ax["name"].get<std::string>());
    }
  }
  if (out.axisNames.empty()) out.axisNames = {"z", "y", "x"};
  if (out.axisNames != std::vector<std::string>{"z", "y", "x"})
    return std::unexpected(Error{"only z,y,x axis order is supported"});

  if (!ms.contains("datasets") || !ms["datasets"].is_array() ||
      ms["datasets"].empty())
    return std::unexpected(Error{"'multiscales[0].datasets' missing or empty"});

  for (const auto& d : ms["datasets"]) {
    if (!d.is_object() || !d.contains("path") || !d["path"].is_string())
      return std::unexpected(Error{"dataset entry missing 'path'"});
    MultiscaleDataset entry;
    entry.path = d["path"].get<std::string>();
    if (d.contains("coordinateTransformations") &&
        d["coordinateTransformations"].is_array()) {
      for (const auto& t : d["coordinateTransformations"]) {
        if (t.is_object() && t.value("type", "") == "scale" &&
            t.contains("scale") && t["scale"].is_array() &&
            t["scale"].size() == 3) {
          for (int i = 0; i < 3; ++i)
            entry.scale[i] = t["scale"][i].get<double>();
        }
      }
    }
    out.datasets.push_back(std::move(entry));
  }

  return out;
}

}  // namespace sv::zarr
