#include "data/TifXyzSurface.h"

#include <tiffio.h>

#include <algorithm>
#include <cstring>

namespace sv::data {

namespace {

// libtiff client-IO shims over an in-memory buffer.
struct MemTiff {
  const std::byte* data;
  toff_t size;
  toff_t pos = 0;
};

tmsize_t memRead(thandle_t h, void* buf, tmsize_t n) {
  auto* m = static_cast<MemTiff*>(h);
  const toff_t left = m->size - m->pos;
  const tmsize_t take = static_cast<tmsize_t>(
      std::min<toff_t>(left, static_cast<toff_t>(n)));
  std::memcpy(buf, m->data + m->pos, static_cast<std::size_t>(take));
  m->pos += static_cast<toff_t>(take);
  return take;
}
tmsize_t memWrite(thandle_t, void*, tmsize_t) { return 0; }
toff_t memSeek(thandle_t h, toff_t off, int whence) {
  auto* m = static_cast<MemTiff*>(h);
  toff_t base = 0;
  if (whence == SEEK_CUR) base = m->pos;
  else if (whence == SEEK_END) base = m->size;
  m->pos = std::min(base + off, m->size);
  return m->pos;
}
int memClose(thandle_t) { return 0; }
toff_t memSize(thandle_t h) { return static_cast<MemTiff*>(h)->size; }

struct FloatImage {
  std::uint32_t w = 0, h = 0;
  std::vector<float> px;
};

Result<FloatImage> readFloatTiff(std::span<const std::byte> bytes) {
  MemTiff mem{bytes.data(), static_cast<toff_t>(bytes.size())};
  TIFFSetWarningHandler(nullptr);
  TIFF* tif = TIFFClientOpen("mem", "r", &mem, memRead, memWrite, memSeek,
                             memClose, memSize, nullptr, nullptr);
  if (!tif) return std::unexpected(Error{"not a readable TIFF"});

  FloatImage img;
  std::uint16_t bps = 0, fmt = SAMPLEFORMAT_UINT, spp = 1;
  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &img.w);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &img.h);
  TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
  TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &fmt);
  TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
  if (img.w == 0 || img.h == 0 || bps != 32 || fmt != SAMPLEFORMAT_IEEEFP ||
      spp != 1) {
    TIFFClose(tif);
    return std::unexpected(
        Error::fmt("unsupported tifxyz TIFF: {}x{} bps={} fmt={} spp={}",
                   img.w, img.h, bps, fmt, spp));
  }

  img.px.resize(std::size_t{img.w} * img.h);
  const tmsize_t rowBytes = TIFFScanlineSize(tif);
  if (rowBytes != static_cast<tmsize_t>(img.w * sizeof(float))) {
    TIFFClose(tif);
    return std::unexpected(Error{"unexpected TIFF scanline size"});
  }
  for (std::uint32_t row = 0; row < img.h; ++row) {
    if (TIFFReadScanline(tif, img.px.data() + std::size_t{row} * img.w,
                         row) < 0) {
      TIFFClose(tif);
      return std::unexpected(Error::fmt("TIFF scanline {} unreadable", row));
    }
  }
  TIFFClose(tif);
  return img;
}

}  // namespace

Result<std::shared_ptr<TifXyzSurface>> TifXyzSurface::load(
    std::span<const std::byte> xTif, std::span<const std::byte> yTif,
    std::span<const std::byte> zTif) {
  auto x = readFloatTiff(xTif);
  if (!x) return std::unexpected(x.error());
  auto y = readFloatTiff(yTif);
  if (!y) return std::unexpected(y.error());
  auto z = readFloatTiff(zTif);
  if (!z) return std::unexpected(z.error());
  if (x->w != y->w || x->w != z->w || x->h != y->h || x->h != z->h)
    return std::unexpected(Error{"tifxyz x/y/z dimensions differ"});

  auto s = std::make_shared<TifXyzSurface>();
  s->w_ = x->w;
  s->h_ = x->h;
  s->pts_.resize(std::size_t{s->w_} * s->h_);
  for (std::size_t i = 0; i < s->pts_.size(); ++i)
    s->pts_[i] = {x->px[i], y->px[i], z->px[i]};
  return s;
}

void TifXyzSurface::applyAffine(
    const std::array<std::array<double, 4>, 3>& m) {
  for (auto& p : pts_) {
    if (p[0] < 0.f || p[1] < 0.f || p[2] < 0.f) continue;
    const double x = p[0], y = p[1], z = p[2];
    const double nx = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
    const double ny = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
    const double nz = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
    // A transformed point outside the target volume becomes invalid.
    if (nx < 0 || ny < 0 || nz < 0) {
      p = {-1.f, -1.f, -1.f};
    } else {
      p = {float(nx), float(ny), float(nz)};
    }
  }
}

}  // namespace sv::data
