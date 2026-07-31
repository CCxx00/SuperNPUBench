# SuperNPUBench

SuperNPUBench is a C++ tile-kernel benchmark repository for superscalar NPU
compiler and model development. It contains direct compiler-validation cases,
one-level kernel manifests, instruction microbenchmarks, and the standalone
programming manual published from `docs/`.

## Repository Structure

```text
SuperNPUBench/
|-- benchmark/
|   |-- two-level-arch/   # promoted compiler and direct-boot cases
|   `-- one-level-arch/   # source-backed tile-kernel catalog
|-- microbenchmark/       # instruction-family microbenchmarks
|-- docs/                 # standalone MkDocs manual, generators, and checks
`-- compile_all.sh
```

Build output remains under architecture-local `output/` directories and is
not committed.

## Current Integration Baseline

Use this repository through the `linx-isa` superproject. Its submodule pins
select the matching v0.57 ISA state, LLVM 23 compiler sources, QEMU, and
LinxCoreModel. Do not combine an arbitrary compiler binary with unrelated ISA
or model revisions.

```bash
git clone --recurse-submodules https://github.com/LinxISA/linx-isa.git
cd linx-isa
git submodule sync --recursive
git submodule update --init --recursive
export LINX_ROOT="$PWD"
```

Configure and build the in-repository compiler:

```bash
cmake -S compiler/llvm/llvm -B compiler/llvm/build-linxisa-clang -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=LinxISA \
  -DLLVM_TARGETS_TO_BUILD=""

cmake --build compiler/llvm/build-linxisa-clang \
  --target clang lld llvm-objdump llvm-objcopy llvm-readobj -j 12

export COMPILER_DIR="$LINX_ROOT/compiler/llvm/build-linxisa-clang/bin"
"$COMPILER_DIR/clang" --version
```

The superproject pin is the source of truth for the exact compiler revision;
rebuild after that pin changes.

## Quick Start

Compile the promoted memory and arithmetic TileOP cases:

```bash
cd "$LINX_ROOT/workloads/SuperNPUBench/benchmark/two-level-arch/test/tileop_api"
make TESTCASE=TAdd PLAT=linx COMPILER_DIR="$COMPILER_DIR"
make TESTCASE=TLoad PLAT=linx COMPILER_DIR="$COMPILER_DIR"
make TESTCASE=TStore PLAT=linx COMPILER_DIR="$COMPILER_DIR"
```

Run one case through the canonical compiler, QEMU, and cycle-model sequence:

```bash
cd "$LINX_ROOT"
export QEMU="$(tools/bringup/run_qemu_build_clean.sh)"
python3 tools/bringup/run_ai_workload_flow.py \
  --profile smoke \
  --kind supernpu \
  --case '=supernpu-tileop_api-TAdd' \
  --run-id supernpu-tadd
```

The runner stops on the first failed stage. A compiler-passing ELF executes in
QEMU before the same artifact is promoted to plain `gfsim -f <elf>`.

<!-- BENCHMARK-CATALOG:START -->
## Benchmark catalog

The active one-level manifests contain **87 build variants**. Every name below
has a source-backed page with its build command and tile intrinsic surface in
the website's **Benchmark Reference** section.
Catalog presence records source inventory; it does not imply promotion on the
current compiler and model flow.

<details><summary><strong>broadcast</strong> (4 names, 6 variants)</summary>

`broadcast`, `broadcast_vec_019`, `broadcast_vec_039`, `broadcast_vec_07`

</details>

<details><summary><strong>concat</strong> (2 names, 4 variants)</summary>

`concat_gather`, `concat_scatter`

</details>

<details><summary><strong>control</strong> (1 name, 6 variants)</summary>

`hashtable_lookup_simd`

</details>

<details><summary><strong>deepseek</strong> (23 names, 23 variants)</summary>

`aux_fi`, `batched_transpose`, `cast_back_per_channel`, `cast_back_per_token`, `engram_hash_layer`, `expand_to_fused`, `expand_to_mhc_bwd`, `expand_to_mhc_fwd`, `fn_normw_merge_fwd`, `fused_weight`, `get_fused_mapping`, `group_count`, `inplace_unique_group_indices`, `mask_indices_by_tp`, `multilayer_recompute`, `normalize_weight`, `per_channel_cast`, `per_token_cast`, `reduce_fused`, `rms_norm`, `sinkhorn_fwd`, `swiglu_forward_and_per_token_cast`, `topk_gate`

</details>

<details><summary><strong>element_wise/gelu</strong> (1 name, 1 variant)</summary>

`gelu`

</details>

<details><summary><strong>fa</strong> (4 names, 13 variants)</summary>

`fa_2d_unroll`, `fa_HIF4_HIF4`, `fa_softmax_pto`, `sfa`

</details>

<details><summary><strong>flashMLA</strong> (1 name, 2 variants)</summary>

`flashMLA`

</details>

<details><summary><strong>gather</strong> (1 name, 1 variant)</summary>

`gather`

</details>

<details><summary><strong>matmul</strong> (1 name, 16 variants)</summary>

`matmul`

</details>

<details><summary><strong>multi_thread/fa</strong> (1 name, 1 variant)</summary>

`fa_2d_unroll_gmma`

</details>

<details><summary><strong>multi_thread/matmul</strong> (2 names, 2 variants)</summary>

`matmul`, `matmul_partial`

</details>

<details><summary><strong>multi_thread/vec</strong> (2 names, 2 variants)</summary>

`tadd`, `trowsum`

</details>

<details><summary><strong>reduction/reducemax_col</strong> (1 name, 1 variant)</summary>

`reducemax_col`

</details>

<details><summary><strong>reduction/reducemax_row</strong> (1 name, 1 variant)</summary>

`reducemax_row`

</details>

<details><summary><strong>reduction/reducesum_col</strong> (1 name, 2 variants)</summary>

`reducesum_col`

</details>

<details><summary><strong>reduction/reducesum_row</strong> (1 name, 1 variant)</summary>

`reducesum_row`

</details>

<details><summary><strong>sort</strong> (1 name, 1 variant)</summary>

`topk`

</details>

<details><summary><strong>transpose</strong> (1 name, 4 variants)</summary>

`transpose`

</details>

<!-- BENCHMARK-CATALOG:END -->

The public catalog covers active one-level `compile.all` manifests. Those
commands are source inventory, not current LLVM 23 promotion evidence: the
one-level make lane still contains a retired compiler option and an unresolved
include-root dependency. Use `make -n` to inspect a catalog command and use the
canonical runner for current compile/model validation.

## Microbenchmarks

`microbenchmark/` organizes cases by matrix, vector, memory, and scalar
instruction families. See [the microbenchmark README](microbenchmark/README.md)
for its generated case inventory and lane-specific commands.

## Documentation

The website source is self-contained under `docs/`. Build and verify it with:

```bash
python3 -m pip install -r docs/requirements.txt
python3 docs/scripts/generate_benchmark_manual.py
python3 docs/scripts/generate_deepseek_manifest.py
python3 docs/scripts/generate_deepseek_manual.py
python3 docs/scripts/sync_golden_manual.py --check --linx-root "$LINX_ROOT"
docs/build.sh
```

Start with:

- [toolchain setup](docs/content/start/toolchain.md)
- [build, run, and disassembly workflow](docs/content/start/build-run.md)
- [C++ tile kernel programming guide](docs/content/programming/cpp-programming-guide.md)
- [intrinsic reference](docs/content/intrinsics/index.md)
- [benchmark reference](docs/content/benchmarks/index.md)

## Adding or Updating a Benchmark

1. Add the kernel and its test entrypoint under the appropriate architecture.
2. Add every supported variant to the local `compile.all` manifest.
3. Regenerate the benchmark and DeepSeek pages when their source inventories
   change.
4. Compile a bounded promoted case with the pinned compiler.
5. Run `docs/build.sh` before publication.
