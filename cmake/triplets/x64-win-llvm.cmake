# Custom triplet: x64 Windows, dynamic CRT/libs, clang-cl + lld, release-only
# dependencies (the app itself builds RelWithDebInfo; debug builds of Qt/VTK
# double the build time for little benefit in this project).
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)
set(VCPKG_BUILD_TYPE release)

# Ports like `opengl` locate the Windows SDK via these (vcvars) variables;
# vcpkg's sanitized build environment drops them unless passed through.
set(VCPKG_ENV_PASSTHROUGH "WindowsSDKVersion;WindowsSdkDir")

# Ports that need MSVC cl (all C-only, so the output is ABI-compatible):
# - autotools ports (libiconv/gettext family): vcpkg's msys 'compile'
#   wrapper mishandles clang-cl argument passing.
# - glew: built CRT-free, counts on MSVC expanding memset as an intrinsic;
#   clang-cl emits a real call that has nothing to link against.
# Everything else builds with clang-cl + lld.
if(NOT PORT MATCHES "^(libiconv|libcharset|gettext|gettext-libintl|libintl|glew)$")
  set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE "D:/scrollviewer/cmake/llvm-toolchain.cmake")
endif()
