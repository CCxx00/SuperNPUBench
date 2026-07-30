# Set Up the Toolchain

SuperNPUBench uses a packaged LLVM + musl cross-toolchain. Build it from the
toolchain repository so the compiler, sysroot, runtime libraries, and tile API
headers come from a matching revision.

## Install host tools

On Debian or Ubuntu:

```console
sudo apt-get update
sudo apt-get install -y git make cmake ninja-build gcc g++ python3 autoconf m4
```

On Apple Silicon, install CMake, Ninja, GNU Make, and GNU tar with Homebrew.
Invoke the build with `gmake`, and put GNU tar's `gnubin` directory on `PATH`
before packaging.

## Build LLVM and the sysroot

```console
git clone https://github.com/LinxISA/linx-toolchain-build.git
cd linx-toolchain-build
make init-src
make WITH_TARGET=linx64v5-linux-musl
```

`make init-src` checks out the compiler, musl, kernel headers, allocator, and
tile API components selected by the build repository. Re-running the build
resumes from stamps; use `make clean` only when a full rebuild is required.

The install tree is:

```text
output/linx_blockisa_llvm_musl/
|-- bin/
|-- lib/
`-- sysroot/
```

## Select the compiler

```console
export COMPILER_DIR="$PWD/output/linx_blockisa_llvm_musl/bin"
test -x "$COMPILER_DIR/clang++"
test -x "$COMPILER_DIR/llvm-objdump"
"$COMPILER_DIR/clang" --version
```

The current chain targets `linx64v5-unknown-linux-musl` and uses the compiler
branch pinned by `linx-toolchain-build` (`dev-llvm15_56` at the time this page
was updated). Do not combine binaries, headers, or a sysroot from independent
toolchain revisions.

## Optional package

```console
make package
```

The archive is written to
`output/linx_blockisa_llvm_musl.tar.gz`. Tagged toolchain releases may also
provide host-qualified Linux and experimental macOS packages.

## Verify with a benchmark dry run

From the SuperNPUBench repository:

```console
cd benchmark/one-level-arch/test/kernel/fa
make -n TESTCASE=fa_2d_unroll Sq=256 Skv=512 Tm=8 Tk=16 X=1 Y=2
```

The command must resolve `clang++`, `ld.lld`, and `llvm-objdump` beneath
`COMPILER_DIR`. Continue with [build, run, and inspect](build-run.md) for a real
compile.
