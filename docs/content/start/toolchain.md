# Set Up the Current Toolchain

The supported integration lane comes from the `linx-isa` superproject. Its
submodule pins keep the v0.57 ISA, LLVM compiler, QEMU, runtime support, and
LinxCoreModel on one reviewed baseline. Use those pins together; do not combine
an arbitrary compiler binary with unrelated ISA or model sources.

## Install host tools

On Debian or Ubuntu:

```console
sudo apt-get update
sudo apt-get install -y git cmake ninja-build gcc g++ python3 pkg-config
```

On macOS, install Git, CMake, Ninja, and Python with Homebrew. Model execution
is substantially faster on a Linux build host, but compiler and documentation
checks also work on macOS.

## Clone the pinned workspace

```console
git clone --recurse-submodules https://github.com/LinxISA/linx-isa.git
cd linx-isa
git submodule sync --recursive
git submodule update --init --recursive
export LINX_ROOT="$PWD"
```

The SuperNPUBench checkout used by the integration flow is
`$LINX_ROOT/workloads/SuperNPUBench`.

## Build the in-repository compiler

Configure the current LLVM tree with the LinxISA experimental target, then
build Clang, LLD, and the object-inspection tools:

```console
cmake -S compiler/llvm/llvm -B compiler/llvm/build-linxisa-clang -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=LinxISA \
  -DLLVM_TARGETS_TO_BUILD=""

cmake --build compiler/llvm/build-linxisa-clang \
  --target clang lld llvm-objdump llvm-objcopy llvm-readobj -j 12
```

Select that build for benchmark commands:

```console
export COMPILER_DIR="$LINX_ROOT/compiler/llvm/build-linxisa-clang/bin"
test -x "$COMPILER_DIR/clang++"
test -x "$COMPILER_DIR/ld.lld"
test -x "$COMPILER_DIR/llvm-objdump"
"$COMPILER_DIR/clang" --version
"$COMPILER_DIR/clang" --print-targets | grep -E 'linx32|linx64'
```

The current compiler line is LLVM 23. The superproject submodule pin, not a
version copied into this manual, determines the exact compiler source.

## Build a provenance-checked QEMU

The canonical runner accepts an explicit QEMU binary, but its reproducible
default is a clean build matched to the pinned QEMU source:

```console
cd "$LINX_ROOT"
export QEMU="$(tools/bringup/run_qemu_build_clean.sh)"
"$QEMU" --version
```

The helper records the QEMU source identity beside the binary and reuses the
build directory when the pin has not changed.

## Build the cycle model

The AI workload runner builds the optimized LinxCoreModel automatically. To
prepare it explicitly:

```console
cmake -S tools/LinxCoreModel -B tools/LinxCoreModel/build \
  -DOPT_LEVEL=O3 -DDISABLE_DEBUG_SYMBOLS=ON
cmake --build tools/LinxCoreModel/build --target gfsim
test -x tools/LinxCoreModel/bin/gfsim
```

Do not validate an arbitrary ELF directly in the cycle model. The supported
flow first compiles it with the pinned compiler, proves it in QEMU, and only
then passes the same artifact to `gfsim`.

## Check the selected benchmark set

List the current SuperNPUBench smoke cases without compiling them:

```console
cd "$LINX_ROOT"
python3 tools/bringup/run_ai_workload_flow.py \
  --profile smoke --kind supernpu --list
```

The list is generated from current `compile.all` manifests. It therefore
tracks renamed cases such as `TLoad` and `TStore` without a second handwritten
inventory in this manual.
