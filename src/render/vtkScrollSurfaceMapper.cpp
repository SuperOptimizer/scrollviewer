#include "render/vtkScrollSurfaceMapper.h"

#include <vtk_glew.h>

#include <vtkCamera.h>
#include <vtkImageData.h>
#include <vtkInformation.h>
#include <vtkMath.h>
#include <vtkMatrix4x4.h>
#include <vtkObjectFactory.h>
#include <vtkOpenGLQuadHelper.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkOpenGLShaderCache.h>
#include <vtkOpenGLState.h>
#include <vtkRenderer.h>
#include <vtkShaderProgram.h>
#include <vtkTextureObject.h>
#include <vtkVolume.h>

#include <cmath>
#include <format>
#include <vector>

#include "core/Log.h"

namespace sv::render {

vtkStandardNewMacro(vtkScrollSurfaceMapper);

vtkScrollSurfaceMapper::vtkScrollSurfaceMapper() {
  vtkNew<vtkImageData> dummy;  // prop dispatch needs a non-null input
  dummy->SetDimensions(1, 1, 1);
  dummy->AllocateScalars(VTK_UNSIGNED_CHAR, 1);
  this->SetInputData(dummy);
}
vtkScrollSurfaceMapper::~vtkScrollSurfaceMapper() = default;

int vtkScrollSurfaceMapper::FillInputPortInformation(int port,
                                                     vtkInformation* info) {
  if (!this->Superclass::FillInputPortInformation(port, info)) return 0;
  info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);
  return 1;
}

void vtkScrollSurfaceMapper::SetVolumeExtent(
    const std::array<std::uint64_t, 3>& shapeZyx, double spacing) {
  ShapeZyx = shapeZyx;
  Spacing = spacing;
  UpdateBounds();
}

void vtkScrollSurfaceMapper::SetSurface(
    std::shared_ptr<const data::TifXyzSurface> s, float gridStep,
    float surfScale) {
  Surface = std::move(s);
  GridStep = gridStep;
  SurfScale = surfScale;
  SurfaceDirty = true;
  UpdateBounds();
}

void vtkScrollSurfaceMapper::SetSlab(float frontVoxels, float behindVoxels) {
  SlabFront = frontVoxels;
  SlabBehind = behindVoxels;
  UpdateBounds();
}

void vtkScrollSurfaceMapper::UpdateBounds() {
  const double vw = Spacing;
  const double W = Surface ? double(Surface->width()) : 1.0;
  const double H = Surface ? double(Surface->height()) : 1.0;
  Bounds[0] = 0;
  Bounds[1] = W * GridStep * vw;
  Bounds[2] = 0;
  Bounds[3] = H * GridStep * vw;
  Bounds[4] = -double(SlabFront) * vw;
  Bounds[5] = double(SlabBehind) * vw;
  this->Modified();
}

double* vtkScrollSurfaceMapper::GetBounds() { return Bounds; }

void vtkScrollSurfaceMapper::ReleaseGraphicsResources(vtkWindow*) {
  Quad.reset();
  if (PosTex) {
    glDeleteTextures(1, &PosTex);
    glDeleteTextures(1, &NrmTex);
    PosTex = NrmTex = 0;
    SurfaceDirty = Surface != nullptr;
  }
}

void vtkScrollSurfaceMapper::EnsureSurfaceTextures() {
  if (!SurfaceDirty || !Surface) return;
  SurfaceDirty = false;

  const std::uint32_t W = Surface->width(), H = Surface->height();
  const auto& pts = Surface->points();
  // Position RGBA32F (.a = validity) and normal RGB32F, bilinearly
  // interpolated in the shader — the 20x grid-to-voxel upsampling rides on
  // hardware filtering.
  std::vector<float> pos(std::size_t{W} * H * 4, 0.f);
  std::vector<float> nrm(std::size_t{W} * H * 3, 0.f);
  auto P = [&](std::uint32_t u, std::uint32_t v) {
    return pts[std::size_t{v} * W + u];
  };
  auto ok = [&](std::uint32_t u, std::uint32_t v) {
    return Surface->valid(u, v);
  };
  for (std::uint32_t v = 0; v < H; ++v)
    for (std::uint32_t u = 0; u < W; ++u) {
      const std::size_t i = std::size_t{v} * W + u;
      if (!ok(u, v)) continue;
      const auto p = P(u, v);
      pos[i * 4 + 0] = p[0];
      pos[i * 4 + 1] = p[1];
      pos[i * 4 + 2] = p[2];
      pos[i * 4 + 3] = 1.f;
      auto diff = [&](int du, int dv, int axis) -> float {
        const bool hasP = u + du < W && v + dv < H && ok(u + du, v + dv);
        const bool hasM =
            int(u) - du >= 0 && int(v) - dv >= 0 && ok(u - du, v - dv);
        if (hasP && hasM)
          return P(u + du, v + dv)[axis] - P(u - du, v - dv)[axis];
        if (hasP) return 2.f * (P(u + du, v + dv)[axis] - p[axis]);
        if (hasM) return 2.f * (p[axis] - P(u - du, v - dv)[axis]);
        return 0.f;
      };
      const float du[3] = {diff(1, 0, 0), diff(1, 0, 1), diff(1, 0, 2)};
      const float dv[3] = {diff(0, 1, 0), diff(0, 1, 1), diff(0, 1, 2)};
      float nx = du[1] * dv[2] - du[2] * dv[1];
      float ny = du[2] * dv[0] - du[0] * dv[2];
      float nz = du[0] * dv[1] - du[1] * dv[0];
      const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
      if (len > 0.f) {
        nrm[i * 3 + 0] = nx / len;
        nrm[i * 3 + 1] = ny / len;
        nrm[i * 3 + 2] = nz / len;
      }
    }

  auto makeTex = [&](unsigned int& id, GLint internal, GLenum fmt,
                     const float* data) {
    if (!id) glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    glTexImage2D(GL_TEXTURE_2D, 0, internal, GLsizei(W), GLsizei(H), 0, fmt,
                 GL_FLOAT, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
  };
  glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
  makeTex(PosTex, GL_RGBA32F, GL_RGBA, pos.data());
  makeTex(NrmTex, GL_RGB32F, GL_RGB, nrm.data());
  logInfo("surface window: {}x{} grid textures uploaded", W, H);
}

std::string vtkScrollSurfaceMapper::BuildFragmentShader() const {
  const int levels = Cache ? Cache->levelCount() : 1;
  std::string dirDecls, dirCases;
  for (int i = 0; i < levels; ++i) {
    dirDecls += std::format("uniform usampler3D pageDir{};\n", i);
    dirCases += std::format(
        "    case {}: return texelFetch(pageDir{}, c, 0).r;\n", i, i);
  }

  return std::format(R"glsl(//VTK::System::Dec
in vec2 texCoord;
out vec4 fragColor;

uniform sampler3D brickPool;
{0}
uniform usampler3D tilePool;
uniform int tilePoolDimX;
uniform int tilePoolDimY;
uniform sampler2D posTex;   // tifxyz positions (volume voxels), .a = valid
uniform sampler2D nrmTex;

uniform mat4 invViewProj;
uniform vec3 volSizeWorld;
uniform vec3 levelShape[{1}];
uniform vec3 gridDim[{1}];
uniform vec3 poolTexels;
uniform float chunkDim;
uniform float borderedDim;
uniform float voxelWorld0;
uniform float gridStep;     // native voxels per grid texel
uniform float surfScale;    // tifxyz voxels -> volume voxels
uniform vec3 flatExtent;    // box extents in world (x, y) and z = [zMin, zMax] via flatZ
uniform vec2 flatZ;         // (zMin, zMax) world
uniform float desiredLevel;
uniform float sampleStepScale;
uniform float winWidth;
uniform float winCenter;
uniform float opacityScale;

const int kLevels = {1};

uint fetchDir(int level, ivec3 c) {{
  switch (level) {{
{2}  }}
  return 0u;
}}

ivec3 tileTexel(uint t, ivec3 brick) {{
  int ti = int(t);
  ivec3 tileCoord = ivec3(ti % tilePoolDimX,
                          (ti / tilePoolDimX) % tilePoolDimY,
                          ti / (tilePoolDimX * tilePoolDimY));
  return tileCoord * 16 + (brick & ivec3(15));
}}

uint fetchPageEntry(int level, ivec3 brick) {{
  uint d = fetchDir(level, brick >> 4);
  if (d < 2u) return (d == 1u) ? 0x80000000u : 0u;
  return texelFetch(tilePool, tileTexel(d - 2u, brick), 0).r;
}}

// Sample the scroll volume at world position pw with coarse fallback.
float sampleVolume(vec3 pw, int lod) {{
  vec3 pn = pw / volSizeWorld;
  if (any(lessThan(pn, vec3(0.0))) || any(greaterThanEqual(pn, vec3(1.0))))
    return 0.0;
  for (int i = lod; i < kLevels; ++i) {{
    vec3 v = pn * levelShape[i];
    ivec3 b = ivec3(clamp(floor(v / chunkDim), vec3(0.0), gridDim[i] - 1.0));
    uint pe = fetchPageEntry(i, b);
    if ((pe & 0x80000000u) != 0u) return 0.0;
    if ((pe & 0x40000000u) != 0u) {{
      vec3 poolBase = vec3(float(pe & 0x3FFu), float((pe >> 10) & 0x3FFu),
                           float((pe >> 20) & 0x3FFu)) * borderedDim + 1.0;
      vec3 inBrick = clamp(v - vec3(b) * chunkDim, vec3(0.0), vec3(chunkDim));
      return texture(brickPool, (poolBase + inBrick) / poolTexels).r;
    }}
  }}
  return 0.0;
}}

bool rayBox(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax, out float t0, out float t1) {{
  vec3 inv = 1.0 / rd;
  vec3 tlo = (bmin - ro) * inv;
  vec3 thi = (bmax - ro) * inv;
  vec3 tmin = min(tlo, thi), tmax = max(tlo, thi);
  t0 = max(max(tmin.x, tmin.y), tmin.z);
  t1 = min(min(tmax.x, tmax.y), tmax.z);
  return t1 > max(t0, 0.0);
}}

float jitter(vec2 co) {{
  return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}}

uniform float wppConst;
uniform float wppPerDist;

void main() {{
  vec2 ndc = texCoord * 2.0 - 1.0;
  vec4 nearW = invViewProj * vec4(ndc, -1.0, 1.0);
  vec4 farW  = invViewProj * vec4(ndc,  1.0, 1.0);
  vec3 ro = nearW.xyz / nearW.w;
  vec3 rd = normalize(farW.xyz / farW.w - ro);

  vec3 bmin = vec3(0.0, 0.0, flatZ.x);
  vec3 bmax = vec3(flatExtent.x, flatExtent.y, flatZ.y);
  float t0, t1;
  if (!rayBox(ro, rd, bmin, bmax, t0, t1)) discard;
  t0 = max(t0, 0.0);

  int minLod = int(clamp(desiredLevel, 0.0, float(kLevels - 1)));
  float dtMin = voxelWorld0 * exp2(float(minLod)) * sampleStepScale;
  float t = t0 + jitter(gl_FragCoord.xy) * dtMin;
  vec4 acc = vec4(0.0);

  for (int s = 0; s < 4096 && t < t1 && acc.a < 0.98; ++s) {{
    vec3 p = ro + t * rd;
    // Flat -> volume warp. The slab is measured in this volume's voxels; the
    // surface grid is in tifxyz voxels (surfScale apart).
    vec2 uv = p.xy / vec2(flatExtent.x, flatExtent.y);
    vec4 pg = texture(posTex, uv);
    float wpp = wppConst + wppPerDist * t;
    int lod = clamp(int(floor(log2(max(1.0, wpp / voxelWorld0)))), minLod,
                    kLevels - 1);
    float dt = voxelWorld0 * exp2(float(lod)) * sampleStepScale;
    if (pg.a > 0.999) {{
      vec3 n = texture(nrmTex, uv).xyz;
      vec3 pw = (pg.xyz * surfScale + normalize(n) * (p.z / voxelWorld0)) *
                voxelWorld0;
      float dens = sampleVolume(pw, lod);
      float w = clamp((dens - winCenter) / winWidth + 0.5, 0.0, 1.0);
      float a = 1.0 - pow(1.0 - w * opacityScale, dt / dtMin);
      acc.rgb += (1.0 - acc.a) * a * vec3(w);
      acc.a   += (1.0 - acc.a) * a;
    }}
    t += dt;
  }}
  if (acc.a < 0.004) discard;
  fragColor = acc;
}}
)glsl",
                     dirDecls, levels, dirCases);
}

void vtkScrollSurfaceMapper::Render(vtkRenderer* ren, vtkVolume*) {
  auto* renWin = static_cast<vtkOpenGLRenderWindow*>(ren->GetRenderWindow());
  if (!Cache || !Cache->initialized() || !Surface) return;
  ++FrameIndex;

  EnsureSurfaceTextures();

  if (!Quad) {
    const std::string fs = BuildFragmentShader();
    Quad = std::make_unique<vtkOpenGLQuadHelper>(renWin, nullptr, fs.c_str(),
                                                 nullptr);
    if (!Quad->Program || !Quad->Program->GetCompiled()) {
      logError("surface raymarch shader failed to compile");
      Quad.reset();
      return;
    }
  }
  renWin->GetShaderCache()->ReadyShaderProgram(Quad->Program);
  vtkShaderProgram* prog = Quad->Program;

  // Shared textures (created in the main window's context, visible here
  // through the Qt share group).
  Cache->poolTexture()->Activate();
  prog->SetUniformi("brickPool", Cache->poolTexture()->GetTextureUnit());
  for (int i = 0; i < Cache->levelCount(); ++i) {
    auto* dir = Cache->directoryTexture(i);
    dir->Activate();
    prog->SetUniformi(std::format("pageDir{}", i).c_str(),
                      dir->GetTextureUnit());
  }
  Cache->tilePoolTexture()->Activate();
  prog->SetUniformi("tilePool", Cache->tilePoolTexture()->GetTextureUnit());
  const auto tpd = Cache->tilePoolDims();
  prog->SetUniformi("tilePoolDimX", int(tpd[2]));
  prog->SetUniformi("tilePoolDimY", int(tpd[1]));

  const int posUnit = 28, nrmUnit = 29;  // clear of VTK's units this frame
  glActiveTexture(GL_TEXTURE0 + posUnit);
  glBindTexture(GL_TEXTURE_2D, PosTex);
  glActiveTexture(GL_TEXTURE0 + nrmUnit);
  glBindTexture(GL_TEXTURE_2D, NrmTex);
  prog->SetUniformi("posTex", posUnit);
  prog->SetUniformi("nrmTex", nrmUnit);

  vtkCamera* cam = ren->GetActiveCamera();
  vtkMatrix4x4* wcdc = cam->GetCompositeProjectionTransformMatrix(
      ren->GetTiledAspectRatio(), -1, 1);
  vtkNew<vtkMatrix4x4> inv;
  vtkMatrix4x4::Invert(wcdc, inv);
  inv->Transpose();
  prog->SetUniformMatrix("invViewProj", inv);

  const float volSize[3] = {float(double(ShapeZyx[2]) * Spacing),
                            float(double(ShapeZyx[1]) * Spacing),
                            float(double(ShapeZyx[0]) * Spacing)};
  prog->SetUniform3f("volSizeWorld", volSize);
  const int levels = Cache->levelCount();
  std::vector<float> levelShape(levels * 3), gridDim(levels * 3);
  for (int i = 0; i < levels; ++i) {
    const auto g = Cache->levelDims(i).grid;
    for (int a = 0; a < 3; ++a) {
      const std::uint64_t full = ShapeZyx[2 - a];
      levelShape[i * 3 + a] =
          float(std::max<std::uint64_t>(1, (full + ((1ull << i) - 1)) >> i));
    }
    gridDim[i * 3 + 0] = float(g[2]);
    gridDim[i * 3 + 1] = float(g[1]);
    gridDim[i * 3 + 2] = float(g[0]);
  }
  prog->SetUniform3fv("levelShape", levels,
                      reinterpret_cast<const float(*)[3]>(levelShape.data()));
  prog->SetUniform3fv("gridDim", levels,
                      reinterpret_cast<const float(*)[3]>(gridDim.data()));
  const auto slotDims = Cache->poolSlotDims();
  const float bd = float(Cache->borderedDim());
  const float poolTexels[3] = {slotDims[2] * bd, slotDims[1] * bd,
                               slotDims[0] * bd};
  prog->SetUniform3f("poolTexels", poolTexels);
  prog->SetUniformf("chunkDim", float(ChunkDim));
  prog->SetUniformf("borderedDim", bd);
  prog->SetUniformf("voxelWorld0", float(Spacing));
  prog->SetUniformf("gridStep", GridStep);
  prog->SetUniformf("surfScale", SurfScale);
  const float flatExtent[3] = {float(Bounds[1]), float(Bounds[3]), 0.f};
  prog->SetUniform3f("flatExtent", flatExtent);
  const float flatZ[2] = {float(Bounds[4]), float(Bounds[5])};
  prog->SetUniform2f("flatZ", flatZ);
  prog->SetUniformf("desiredLevel", DesiredLevel);
  prog->SetUniformf("sampleStepScale", SampleStepScale);
  prog->SetUniformf("winWidth", Window);
  prog->SetUniformf("winCenter", Level);
  prog->SetUniformf("opacityScale", OpacityScale);

  const int vpHeight = std::max(1, ren->GetSize()[1]);
  float wppC = 0.f, wppD = 0.f;
  if (cam->GetParallelProjection()) {
    wppC = float(2.0 * cam->GetParallelScale() / vpHeight);
  } else {
    wppD = float(
        2.0 * std::tan(vtkMath::RadiansFromDegrees(cam->GetViewAngle()) / 2) /
        vpHeight);
  }
  prog->SetUniformf("wppConst", wppC);
  prog->SetUniformf("wppPerDist", wppD);

  vtkOpenGLState* state = renWin->GetState();
  state->vtkglDepthMask(GL_FALSE);
  state->vtkglDisable(GL_DEPTH_TEST);
  state->vtkglEnable(GL_BLEND);
  state->vtkglBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
                                GL_ONE_MINUS_SRC_ALPHA);
  Quad->Render();
  state->vtkglDepthMask(GL_TRUE);
  state->vtkglEnable(GL_DEPTH_TEST);

  glActiveTexture(GL_TEXTURE0 + posUnit);
  glBindTexture(GL_TEXTURE_2D, 0);
  glActiveTexture(GL_TEXTURE0 + nrmUnit);
  glBindTexture(GL_TEXTURE_2D, 0);
  Cache->poolTexture()->Deactivate();
  for (int i = 0; i < Cache->levelCount(); ++i)
    Cache->directoryTexture(i)->Deactivate();
  Cache->tilePoolTexture()->Deactivate();
}

}  // namespace sv::render
