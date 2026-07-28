# Chainload toolchain: compile everything (vcpkg ports and this project) with
# the VS-bundled LLVM (clang-cl + lld-link) instead of MSVC cl.
set(SV_LLVM_BIN "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin")

set(CMAKE_C_COMPILER "${SV_LLVM_BIN}/clang-cl.exe" CACHE FILEPATH "")
set(CMAKE_CXX_COMPILER "${SV_LLVM_BIN}/clang-cl.exe" CACHE FILEPATH "")
set(CMAKE_LINKER "${SV_LLVM_BIN}/lld-link.exe" CACHE FILEPATH "")
set(CMAKE_AR "${SV_LLVM_BIN}/llvm-lib.exe" CACHE FILEPATH "")
# Full SDK paths: the chainloaded environment does not put the Windows SDK
# bin directory on PATH.
set(SV_SDK_BIN "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64")
set(CMAKE_RC_COMPILER "${SV_SDK_BIN}/rc.exe" CACHE FILEPATH "")
set(CMAKE_MT "${SV_SDK_BIN}/mt.exe" CACHE FILEPATH "")

# rc.exe relies on %INCLUDE%, which vcpkg does not populate for chainloaded
# toolchains — pass the SDK include roots explicitly. Must be a cache set
# BEFORE including vcpkg's windows.cmake (its own non-FORCE cache set of
# CMAKE_RC_FLAGS then becomes a no-op).
set(SV_SDK_INC "C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0")
set(SV_MSVC_INC "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36231/include")
set(CMAKE_RC_FLAGS
    "/c65001 /DWIN32 /I\"${SV_MSVC_INC}\" /I\"${SV_SDK_INC}/ucrt\" /I\"${SV_SDK_INC}/um\" /I\"${SV_SDK_INC}/shared\""
    CACHE STRING "")

# cmcldeps preprocesses .rc files through the C compiler for dependency
# scanning; clang-cl rejects rc-only flags like /c65001. Scan-less RC is fine.
set(CMAKE_NINJA_CMCLDEPS_RC OFF CACHE BOOL "")

# Fall through to vcpkg's standard Windows toolchain for runtime/flag setup.
if(DEFINED Z_VCPKG_ROOT_DIR)
  include("${Z_VCPKG_ROOT_DIR}/scripts/toolchains/windows.cmake")
elseif(DEFINED ENV{VCPKG_ROOT})
  include("$ENV{VCPKG_ROOT}/scripts/toolchains/windows.cmake")
endif()
