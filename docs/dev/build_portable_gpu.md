# Build Portable GPU Support

Portable GPU support enables the OpenVINO GPU plugin on AArch64 OpenCL
devices. It is experimental and opt-in:

```text
-DENABLE_EXPERIMENTAL_PORTABLE_GPU=ON
```

This single option enables the GPU plugin, selects the OpenCL runtime, and
disables Intel-specific oneDNN GPU and CM paths. Do not set those options
individually.

## Prerequisites

All build hosts require Git, CMake, Ninja, Python 3, and a C++ toolchain.
Initialize the OpenVINO submodules before configuring:

```bash
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino
git submodule update --init --recursive
```

Use these external dependency versions for clvk-based targets:

| Dependency | Revision |
| --- | --- |
| clvk | `a56c4a442a4671fc08478ab39463a1da28e4a818` |
| clspv | `545a65b5f4fda9a7c285bb7fd75eda86aa2d853b` |
| clspv LLVM | `c105848fd29d3b46eeb794bb6b10dad04f903b09` |
| MoltenVK | `1.4.0` |
| Android NDK | `29.0.14206865` |

Clone clvk, check out the pinned revision, initialize its submodules, check out
the pinned `external/clspv` revision, and run:

```bash
python3 external/clspv/utils/fetch_sources.py --deps llvm --shallow
```

Do not patch or vendor clvk, clspv, LLVM, MoltenVK, Mesa, or a device OpenCL
library in the OpenVINO source tree.

## Configure OpenVINO

The common configuration is:

```bash
cmake -S . -B <build-dir> -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_EXPERIMENTAL_PORTABLE_GPU=ON
cmake --build <build-dir> --parallel
cmake --install <build-dir> --prefix <install-dir> --component core
cmake --install <build-dir> --prefix <install-dir> --component gpu
```

Add the platform arguments described below.

### Android AArch64

Install the pinned NDK with Android SDK Manager. Configure with:

```bash
-DCMAKE_TOOLCHAIN_FILE=<android-ndk>/build/cmake/android.toolchain.cmake
-DANDROID_ABI=arm64-v8a
-DANDROID_PLATFORM=35
-DANDROID_STL=c++_shared
```

The plugin loads the device-provided `libOpenCL.so` through the Android NDK
`libdl` API. Do not copy a vendor OpenCL library into the build or application.
Applications must place this declaration inside the manifest `<application>`
element:

```xml
<uses-native-library android:name="libOpenCL.so" android:required="true" />
```

Package the installed OpenVINO libraries and `libc++_shared.so` from the same
NDK, then deploy them with the application or `adb`.

### Raspberry Pi 5

Build only on an x86_64 Linux host with `gcc-aarch64-linux-gnu` and
`g++-aarch64-linux-gnu`. Use a target SDK/sysroot that matches the deployed
64-bit OS and provides Vulkan headers and an AArch64 `libvulkan.so`. Never
compile OpenVINO, clvk, clspv, or SPIR-V tools on the Pi.

Cross-build clvk with:

```bash
-DCMAKE_TOOLCHAIN_FILE=<openvino>/cmake/arm64.toolchain.cmake
-DCLVK_VULKAN_IMPLEMENTATION=custom
-DVulkan_INCLUDE_DIRS=<target-sysroot>/usr/include
-DVulkan_LIBRARIES=<target-sysroot>/usr/lib/aarch64-linux-gnu/libvulkan.so
-DLLVM_HOST_TRIPLE=aarch64-linux-gnu
-DLLVM_TARGET_ARCH=AArch64
```

Build the `OpenCL`, `clspv`, `simple_test`, and `spirv-val` targets. Configure
OpenVINO with the same AArch64 toolchain file. Deploy the installed OpenVINO
libraries, OpenCL ICD loader, clvk library, and `clspv` executable. At runtime,
set `OCL_ICD_FILENAMES` to clvk and `CLVK_CLSPV_PATH` to the deployed compiler.
The target OS supplies Mesa V3DV and the Vulkan loader.

### Apple Silicon

Build natively on arm64 macOS with Xcode command-line tools. Download the
official MoltenVK `1.4.0` release and build clvk with:

```bash
-DCMAKE_OSX_ARCHITECTURES=arm64
-DCLVK_VULKAN_IMPLEMENTATION=custom
-DVulkan_INCLUDE_DIRS=<vulkan-headers>/include
-DVulkan_LIBRARIES=<moltenvk>/libMoltenVK.dylib
```

Add `-DCMAKE_OSX_ARCHITECTURES=arm64` to the OpenVINO configuration. Package
OpenVINO, clvk, clspv, and MoltenVK as arm64 binaries. Use relative `@rpath`
and `@loader_path` install names and re-sign modified binaries. Set
`OCL_ICD_FILENAMES` to clvk and `CLVK_CLSPV_PATH` to clspv. clvk must link
directly to MoltenVK; a separate Vulkan loader is not required.

### Intel GPU Reference Build

Use the regular supported GPU configuration on x86_64 and leave
`ENABLE_EXPERIMENTAL_PORTABLE_GPU` off:

```bash
cmake -S . -B build-intel -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_INTEL_GPU=ON \
  -DGPU_RT_TYPE=OCL
cmake --build build-intel --parallel
```

## Verify the Runtime

Before inference, confirm the expected physical device: Qualcomm/Adreno on
Android, V3D through V3DV and clvk on Pi, Apple GPU through MoltenVK and clvk
on macOS, or Intel GPU on the reference system. Reject software Vulkan devices
such as `llvmpipe` and `lavapipe`.

For a validation build, add `-DENABLE_TESTS=ON` and
`-DENABLE_FUNCTIONAL_TESTS=ON`, build `ov_gpu_func_tests`, and run:

```bash
ov_gpu_func_tests --disable_tests_skipping \
  --gtest_filter='PortableEltwiseFamily*.*'
```

The filter must enumerate and pass all 28 Eltwise family cases without skips
or CPU fallback.
