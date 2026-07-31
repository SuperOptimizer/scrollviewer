#include "data/Snic3D.h"

#include <algorithm>
#include <cmath>
#include <queue>

namespace sv::data {

namespace {

struct QEl {
  float d;
  std::uint32_t idx;
  std::uint32_t k;
};
struct QElGreater {
  bool operator()(const QEl& a, const QEl& b) const { return a.d > b.d; }
};

struct Accum {
  double z = 0, y = 0, x = 0, sum = 0;
  std::uint32_t n = 0;
};

}  // namespace

std::vector<SnicCluster> snic3d(std::span<const std::uint8_t> vol,
                                const std::array<std::uint32_t, 3>& dimsZyx,
                                double seedFraction, float compactness,
                                std::uint8_t minIntensity) {
  const std::uint32_t dz = dimsZyx[0], dy = dimsZyx[1], dx = dimsZyx[2];
  const std::size_t n = std::size_t{dz} * dy * dx;
  if (n == 0 || vol.size() < n || seedFraction <= 0.0) return {};

  // Grid step from the requested cluster density.
  const std::uint32_t step = std::max<std::uint32_t>(
      2, std::uint32_t(std::lround(std::cbrt(1.0 / seedFraction))));
  const float invS2 = 1.f / (float(step) * float(step));
  const float invM2 = 1.f / (compactness * compactness);

  std::vector<std::int32_t> label(n, -1);
  std::vector<Accum> acc;
  std::priority_queue<QEl, std::vector<QEl>, QElGreater> pq;

  auto at = [&](std::uint32_t z, std::uint32_t y,
                std::uint32_t x) -> std::size_t {
    return (std::size_t{z} * dy + y) * dx + x;
  };

  // Seeds on a jittered-free regular grid; empty voxels never seed.
  for (std::uint32_t z = step / 2; z < dz; z += step)
    for (std::uint32_t y = step / 2; y < dy; y += step)
      for (std::uint32_t x = step / 2; x < dx; x += step) {
        const std::size_t i = at(z, y, x);
        if (vol[i] <= minIntensity) continue;
        pq.push({0.f, std::uint32_t(i), std::uint32_t(acc.size())});
        acc.push_back({});
      }
  if (acc.empty()) return {};

  while (!pq.empty()) {
    const QEl e = pq.top();
    pq.pop();
    if (label[e.idx] >= 0) continue;
    label[e.idx] = std::int32_t(e.k);

    Accum& a = acc[e.k];
    const std::uint32_t x = e.idx % dx;
    const std::uint32_t y = (e.idx / dx) % dy;
    const std::uint32_t z = std::uint32_t(e.idx / (std::size_t{dx} * dy));
    const float iv = float(vol[e.idx]) * (1.f / 255.f);
    a.z += z;
    a.y += y;
    a.x += x;
    a.sum += iv;
    ++a.n;

    const float inv = 1.f / float(a.n);
    const float cz = float(a.z) * inv, cy = float(a.y) * inv,
                cx = float(a.x) * inv, cm = float(a.sum) * inv;

    const auto push = [&](std::uint32_t nz, std::uint32_t ny,
                          std::uint32_t nx) {
      const std::size_t ni = at(nz, ny, nx);
      if (label[ni] >= 0 || vol[ni] <= minIntensity) return;
      const float ddz = float(nz) - cz, ddy = float(ny) - cy,
                  ddx = float(nx) - cx;
      const float di = float(vol[ni]) * (1.f / 255.f) - cm;
      const float d =
          (ddz * ddz + ddy * ddy + ddx * ddx) * invS2 + di * di * invM2;
      pq.push({d, std::uint32_t(ni), e.k});
    };
    if (x > 0) push(z, y, x - 1);
    if (x + 1 < dx) push(z, y, x + 1);
    if (y > 0) push(z, y - 1, x);
    if (y + 1 < dy) push(z, y + 1, x);
    if (z > 0) push(z - 1, y, x);
    if (z + 1 < dz) push(z + 1, y, x);
  }

  std::vector<SnicCluster> out;
  out.reserve(acc.size());
  for (const Accum& a : acc) {
    if (a.n == 0) continue;
    const float inv = 1.f / float(a.n);
    out.push_back({float(a.z) * inv, float(a.y) * inv, float(a.x) * inv,
                   float(a.sum) * inv, a.n});
  }
  return out;
}

std::uint8_t otsuThreshold(std::span<const std::uint8_t> vol) {
  std::array<std::uint64_t, 256> h{};
  for (const std::uint8_t b : vol) ++h[b];
  const double total = double(vol.size());
  if (total == 0) return 0;
  double sumAll = 0;
  for (int i = 0; i < 256; ++i) sumAll += double(i) * double(h[i]);

  double sumB = 0, wB = 0, best = -1.0;
  int thr = 0;
  for (int t = 0; t < 256; ++t) {
    wB += double(h[t]);
    if (wB == 0) continue;
    const double wF = total - wB;
    if (wF == 0) break;
    sumB += double(t) * double(h[t]);
    const double mB = sumB / wB, mF = (sumAll - sumB) / wF;
    const double between = wB * wF * (mB - mF) * (mB - mF);
    if (between > best) {
      best = between;
      thr = t;
    }
  }
  return std::uint8_t(thr);
}

std::vector<WorldSphere> clustersToWorldSpheres(
    std::span<const SnicCluster> clusters, float voxelScale,
    std::uint32_t minCount) {
  std::vector<WorldSphere> out;
  out.reserve(clusters.size());
  constexpr float kPi = 3.14159265f;
  for (const SnicCluster& c : clusters) {
    if (c.count < minCount) continue;
    const float radius =
        std::cbrt(3.f * float(c.count) / (4.f * kPi)) * voxelScale;
    out.push_back({(c.x + 0.5f) * voxelScale, (c.y + 0.5f) * voxelScale,
                   (c.z + 0.5f) * voxelScale, radius, c.mean});
  }
  return out;
}

}  // namespace sv::data
