# Build, Run, and Inspect Generated Code

Use separate checks for compilation, disassembly, and execution. A successful
compile validates the C++ and tile-operation surface; it does not validate
tensor results.

## Compile one active case

```console
export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin
cd benchmark/one-level-arch/test/kernel/fa
make TESTCASE=fa_2d_unroll Sq=256 Skv=512 Tm=8 Tk=16 X=1 Y=2
```

The active FlashAttention and matmul manifests use shapes whose physical tiles
do not exceed 4 KiB. Start from a catalogued command before changing tile
dimensions.

## Generate disassembly

```console
make TESTCASE=fa_2d_unroll Sq=256 Skv=512 Tm=8 Tk=16 X=1 Y=2 diss
```

The `diss` target runs the toolchain's `llvm-objdump` and writes a `.diss` file
next to the generated ELF under `benchmark/one-level-arch/output/`.

Look for operation blocks such as `BSTART.TLOAD`, `BSTART.TMATMUL`, and
`BSTART.TSTORE`. Exact scheduling and surrounding scalar instructions can
change with the compiler revision.

## Run with the Makefile QEMU target

The harness accepts a QEMU-compatible block model through the `QEMU` variable:

```console
make TESTCASE=fa_2d_unroll Sq=256 Skv=512 Tm=8 Tk=16 X=1 Y=2 \
  sim QEMU=/path/to/qemu-linx
```

Pass the executable explicitly. The repository does not publish a developer's
local simulator path.

## Run with the current models

For functional or cycle-oriented execution, build SuperScalarModel and pass it
the ELF:

```console
bin/gfrun -f /path/to/kernel.elf
bin/gfsim -f /path/to/kernel.elf
```

Pure tile-operation kernels that target VectorLite may require:

```console
bin/gfsim -f /path/to/kernel.elf -s core.singleTierMode=true
```

## Compile all active suites

```console
./compile_all.sh one-level
```

The [benchmark catalog](../benchmarks/index.md) is generated from each active
`compile.all` command and is the authoritative list of documented cases.

## Diagnose by stage

| Stage | Typical signal | First check |
| --- | --- | --- |
| Frontend | C++ type or template diagnostic | element types, shape constants, valid extents |
| Tile lowering | compiler crash or rejected operation | smallest reproducer and compiler revision |
| Link | missing symbol or relocation | matching toolchain sysroot and startup objects |
| Load | ELF or target error | target triple and simulator revision |
| Runtime | trap, wrong result, or timeout | disassembly, generated data, and execution trace |
