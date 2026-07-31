# Build, Run, and Inspect Generated Code

Treat source validation, compilation, QEMU execution, and cycle-model
execution as separate gates. A later gate consumes only artifacts that passed
the earlier gates.

## Compile a current TileOP smoke case

Use the in-repository LLVM build from the toolchain chapter:

```console
export LINX_ROOT=/path/to/linx-isa
export SUPER_NPU_ROOT="$LINX_ROOT/workloads/SuperNPUBench"
export COMPILER_DIR="$LINX_ROOT/compiler/llvm/build-linxisa-clang/bin"

cd "$SUPER_NPU_ROOT/benchmark/two-level-arch/test/tileop_api"
make TESTCASE=TAdd PLAT=linx
make TESTCASE=TLoad PLAT=linx
```

These bounded direct-boot cases are useful compiler checks. `TLoad` and
`TStore` are the current memory-operation case names; retired copy-in/copy-out
case names are not part of the active manifest.

## Inspect a one-level benchmark manifest

The public benchmark catalog documents the active one-level kernel manifests.
Inspect an exact command without invoking the retired compiler flags still
present in this source lane:

```console
cd "$SUPER_NPU_ROOT/benchmark/one-level-arch/test/kernel/fa"
make -n TESTCASE=fa_2d_unroll Sq=256 Skv=512 Tm=8 Tk=16 X=1 Y=2 \
  PLAT=linx COMPILER_DIR="$COMPILER_DIR"
```

These manifests remain the source-backed benchmark inventory, but their shared
make harness still uses the retired `-mlxbc` option and references a one-level
include tree that is not present on current `main`. Treat the commands as
catalog evidence until that lane is migrated. Current LLVM 23 compile and run
validation uses the promoted TileOP cases and canonical runner below.

## Inspect block disassembly for a compiler-supported case

```console
cd "$SUPER_NPU_ROOT/benchmark/two-level-arch/test/tileop_api"
make TESTCASE=TAdd PLAT=linx diss
```

The `diss` target uses the matching `llvm-objdump`. Look for operation blocks
such as `BSTART.TLOAD`, `BSTART.TMATMUL`, and `BSTART.TSTORE`. Scheduling and
surrounding scalar instructions may change as the compiler evolves.

## Run the canonical compiler-to-model flow

From the superproject root, build a HEAD-matched QEMU and run one bounded case:

```console
cd "$LINX_ROOT"
export COMPILER_DIR="$LINX_ROOT/compiler/llvm/build-linxisa-clang/bin"
export QEMU="$(tools/bringup/run_qemu_build_clean.sh)"

python3 tools/bringup/run_ai_workload_flow.py \
  --profile smoke \
  --kind supernpu \
  --case '=supernpu-tileop_api-TAdd' \
  --run-id supernpu-tadd
```

The runner applies these hard-break stages in order:

1. Validate manifests and source paths.
2. Compile, link, disassemble, and inspect the ELF with the pinned LLVM tools.
3. Execute the legal compiler artifact in QEMU.
4. Build and smoke-test the optimized LinxCoreModel.
5. Run only the QEMU-passing ELF through plain `gfsim -f <elf>`.

Run `--profile pr --kind supernpu --list` to see the larger promoted set. Use
an exact `--case '=...'` selector when reproducing one row.

## Inspect run artifacts

Each run writes its manifest, report, logs, ELF, disassembly, and failure packet
under:

```text
workloads/generated/<run-id>/ai-bringup/
```

Read `summary.md` first, then open the first failing stage log. A QEMU-passing
ELF that fails or times out in `gfsim` remains a model-lane issue until trace or
static evidence identifies an earlier architectural mismatch.

## Check the published catalog

```console
cd "$SUPER_NPU_ROOT"
python3 docs/scripts/generate_benchmark_manual.py
git diff --exit-code -- docs/content/benchmarks docs/mkdocs.yml README.md
```

The [benchmark catalog](../benchmarks/index.md) is regenerated from every
active one-level `compile.all` row and is the authoritative source inventory.
Compiler promotion status is recorded by the canonical runner, not inferred
from catalog presence.
