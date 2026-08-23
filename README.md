# SuperNPUBench

SuperNPUBench is a high-performance operator library and benchmark platform for
NPU tile-programming ISA. It ships **two architecture backends** under `benchmark/`
(two-level-arch = LinxISA, one-level-arch = PTO ISA) plus an instruction-level
**microbenchmark** suite, all driven by the same Linx toolchain.

> **IMPORTANT**: Only `benchmark/one-level-arch/` and `microbenchmark/` are
> compilable with the current toolchain. The `benchmark/two-level-arch/`
> (LinxISA) kernels are **not** compilable — they require a different ISA mode
> not supported by the current `linx_blockisa_llvm_musl` build. Do **not**
> include `two-level-arch` in batch compilation (`compile_all.sh two-level` will
> fail).

## Repository Structure

```
SuperNPUBench/
├── benchmark/
│   ├── two-level-arch/      # Linx two-level block ISA
│   │   ├── kernels/         # header-only operator implementations
│   │   ├── test/            # test suites + build system
│   │   └── compile_all.sh
│   ├── one-level-arch/      # PTO one-level tile ISA
│   │   ├── kernels/
│   │   ├── test/
│   │   │   ├── common/      # shared Makefile.common, _start.s
│   │   │   └── kernel/      # per-operator test cases
│   │   └── compile_all.sh
├── microbenchmark/          # instruction-level micro-bench (cube/vector/memory/scalar)
├── docs/                    # documentation
│   ├── programming/        # PTO C++ Programming Guide
│   └── workflow/           # end-to-end workflow docs
└── compile_all.sh           # top-level: two-level | one-level | all
```

> Build outputs (`output/`, `**/output/`) and `.DS_Store` are gitignored.

## Architecture Backends

### two-level-arch (LinxISA)
- Block-structured ISA with heterogeneous cores: BCC (main), Cube (matrix), Vector, MTC/TMA (data transfer).
- Programming model: block instructions (VPAR/VSEQ, CUBE, TMA, TEPL).

### one-level-arch (PTO ISA)
- Tile-centric ISA with explicit memory hierarchy: Vec, Mat, Left, Right, Acc.
- Programming model: tile operations via Linx-TileOP-API C++ templates.
- Programming guide: [`docs/programming/pto c++ programming guide.md`](docs/programming/pto%20c++%20programming%20guide.md).

Both backends share the same operator set and test layout; their kernel
implementations differ in ISA style.

## Operator Overview

Each backend implements operator categories:

| Operator | Description |
|----------|-------------|
| **matmul** | FP4/BF16/FP32/FP16/FP8 matrix multiply; quantization, mixed precision, A/B reuse, GMMA shared-tile |
| **fa** | Flash Attention; 2D unroll, SFA (block-sparse), HIF4 quantization, softmax_pto, unaligned boundary |
| **flashMLA** | Flash MLA (multi-head latent attention) |
| **transpose** | 3D~6D tensor transpose; multiple dtypes |
| **reduction** | Row/column max & sum; single-tree, unaligned, cumsum, reduceprod |
| **gelu** | GELU activation; exact (erf) and tanh approximation |
| **broadcast** | 2D~5D broadcast; vectorized variants |
| **gather** | Data gathering; large-scale, power-of-2 dims |
| **concat** | Concatenation; gather/scatter modes |
| **control** | `hashtable_lookup_simd` (pure tile-op, single-tier gfsim) |
| **sort** | `topk` (radix-bucket histogram) |
| **deepseek** | 22 migrated DeepSeek kernels (engram/mhc/moe/quant/transpose) |

## Setup Environment

SuperNPUBench compiles with the **Linx toolchain** (`linx_blockisa_llvm_musl`,
clang-15, target `linx64v5-unknown-linux-musl`). Build it once from the
[`linx-toolchain-build`](https://github.com/LinxISA/linx-toolchain-build) repo,
which clones the matching ISA sources and produces the `linx_blockisa_llvm_musl`
install tree that `COMPILER_DIR` points at.

### 1. Clone the build repo

```bash
git clone https://github.com/LinxISA/linx-toolchain-build.git
cd linx-toolchain-build
```

### 2. Install host build tools

```bash
sudo apt-get install -y git make cmake ninja-build gcc g++ python3 autoconf m4
```

### 3. Initialize component sources

`make init-src` clones the five component repos under `src/` on their pinned
branches (run it again any time to fetch updates):

| Directory | Repository | Branch |
| --- | --- | --- |
| `src/llvm-project` | `LinxISA/llvm-project` | `dev-llvm15_56` |
| `src/musl` | `LinxISA/linx-musl` | `linx` |
| `src/jemalloc` | `LinxISA/jemalloc` | `linx` |
| `src/linux-linxisa` | `LinxISA/linux` | `main` |
| `src/Linx-TileOP-API` | `LinxISA/Linx-TileOP-API` | `linx` |

```bash
make init-src
```

### 4. Build the toolchain

Only `linx64v5-linux-musl` is supported by the top-level Makefile:

```bash
make WITH_TARGET=linx64v5-linux-musl
```

This builds, in order: LLVM/clang/lld → kernel headers → musl → compiler-rt →
libc++/libc++abi/libunwind → jemalloc → Linx-TileOP-API headers. Progress is
tracked by stamp files under `stamps/`, so re-running `make` resumes from the
last completed step; `make clean` rebuilds from scratch. The install tree is
written to `output/linx_blockisa_llvm_musl/`:

```
output/linx_blockisa_llvm_musl/
├── bin/        # clang, clang++, ld.lld, llvm-ar/nm/ranlib,
│              # linx64v5-linux-musl-clang(++) symlinks
├── lib/        # clang runtime, libc++, ...
└── sysroot/    # musl + kernel headers + runtime libs
```

### 5. Point SuperNPUBench at the toolchain

```bash
export COMPILER_DIR=$(pwd)/output/linx_blockisa_llvm_musl/bin
$COMPILER_DIR/clang --version
# clang version 15.0.4 (linx64v5-musl-local ...)
# Target: linx64v5-unknown-linux-musl
```

Then proceed to [Quick Start](#quick-start).

### (Optional) Package

```bash
make package     # -> output/linx_blockisa_llvm_musl.tar.gz
```

## Quick Start

### 1. Environment

Build the Linx toolchain once (see [Setup Environment](#setup-environment)), then
point `COMPILER_DIR` at it:

```bash
export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin
```

### 2. Compile an operator

```bash
# one-level-arch (PTO ISA)
cd benchmark/one-level-arch/test/kernel/matmul
make TESTCASE=matmul TYPE=MASK MODE=MASK_FP32 M=256 N=256 K=256 tM=16 tN=16 tK=64

# deepseek kernel
cd benchmark/one-level-arch/test/kernel/deepseek
make TESTCASE=fused_weight diss
```

### 3. Batch / full compilation

```bash
# one-level-arch only (recommended)
./compile_all.sh one-level

# microbenchmark
cd microbenchmark && bash compile_all.sh all
```

> **Do NOT run `compile_all.sh two-level` or `compile_all.sh all`** —
> `two-level-arch` kernels cannot compile with the current toolchain.

Artifacts land in `benchmark/<arch>/output/kernel/<operator>/elf/`.

## Microbenchmark

`microbenchmark/` is an instruction-level bench organized by ISA family,
generated by `gen_cases.py`.

| family | covers | cases |
| --- | --- | ---: |
| cube (CUBE) | TMATMUL / TMATMUL_BIAS / TMATMUL_MX / ACCCVT | 9 |
| vector (TEPL) | elementwise / tile-scalar / reduce / expand / TCI sequence (toolchain-exposed subset) | 128 |
| memory (TLSU) | TLOAD / TSTORE / TMOV / MGATHER / MSCATTER (+mask, layout) | 25 |
| scalar (GPR) | int ALU / load-store / float / conversion × throughput+latency | 124 |
| **total** | | **286** |

```bash
cd microbenchmark && make TESTCASE=tmatmul_fp16_64x64x64   # one case
cd microbenchmark && bash compile_all.sh all               # all families
```

See [`microbenchmark/README.md`](microbenchmark/README.md) for details.

## Running on the Models

Compiled ELF binaries run on the **SuperScalarModel** simulator suite. Build
`gfrun`/`gfsim` from the [SuperScalarModel](../SuperScalarModel) repo, then
point them at the ELF:

- `gfrun` — functional model (correctness)
- `gfsim` — cycle-accurate model (timing)

```bash
# from the SuperScalarModel repo root (where bin/ lives)
bin/gfrun -f /path/to/SuperNPUBench/benchmark/one-level-arch/output/kernel/<op>/elf/<name>.elf
bin/gfsim -f /path/to/SuperNPUBench/benchmark/one-level-arch/output/kernel/<op>/elf/<name>.elf
```

### Tile-op kernels: single-tier gfsim mode

Kernels written purely with tile ops using TEPL template instructions (e.g.
`control/hashtable_lookup_simd`) run on the VectorLite engine, which gfsim only
steps in **single-tier mode**:

```bash
bin/gfsim -f <elf> -s core.singleTierMode=true
```

Without this flag the engine is inert and the run deadlocks. `gfrun` does not
need the flag.

## Build System

### Makefile parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `TESTCASE` | Test case name | `matmul`, `fa_2d_unroll` |
| `TYPE` | Operator type (matmul) | `HIF4_HIF4`, `A16W4`, `MASK` |
| `MODE` | Operator mode | `MASK_FP32`, `BF16x2_NOGATHER` |
| `M/N/K` | Matrix dimensions | `M=256 N=2048 K=2048` |
| `tM/tN/tK` | Tile sizes | `tM=128 tN=128 tK=128` |
| `COMPILER_DIR` | Compiler path | `/path/to/linx/bin` |
| `PLAT` | Platform | `linx` (default), `cpu` |

### Build targets

```bash
make TESTCASE=<case> all      # compile
make TESTCASE=<case> diss     # disassembly
make TESTCASE=<case> sim      # run in simulator
make TESTCASE=<case> debug    # debug mode
make clean                    # clean current operator
make clean_all                # clean all
```

## Documentation

- **PTO C++ Programming Guide**: [`docs/programming/pto c++ programming guide.md`](docs/programming/pto%20c++%20programming%20guide.md)
- **End-to-end Workflow**: [`docs/workflow/operator_to_chip_execution_flow.md`](docs/workflow/operator_to_chip_execution_flow.md)
- **Per-operator README**: see `benchmark/one-level-arch/kernels/<operator>/README.md`
- **Microbenchmark**: [`microbenchmark/README.md`](microbenchmark/README.md)
- **TileOP-API Reference**: [Linx-TileOP-API tileop-usage docs](https://github.com/LinxISA/Linx-TileOP-API/tree/linx/docs/tileop-usage)

## Toolchain

- Compiler: `linx_blockisa_llvm_musl` (clang-15, linx64v5-musl)
- Flags: `-mlxbc -fenable-matrix -O2 -mllvm -enable-all-vector-as-tilereg=true -std=c++20`
- Target: Linx64 V5

## Development Guide

### Adding an operator

1. Add header-only kernel under `benchmark/<arch>/kernels/<operator>/`.
2. Create test dir under `benchmark/<arch>/test/kernel/<operator>/` with
   `Makefile`, `compile.all`, `src/`.
3. Add the operator to `compile_all.sh`.

### Conventions

- Header-only kernels; PTO tile-programming paradigm.
- Build artifacts not tracked (`.gitignore`).

## Related Links

- [LinxISA](https://linxisa.github.io/linx-isa/)
- [PTO ISA](https://pto-isa.github.io/docs/isa/tile/)
- [Linx-TileOP-API](https://github.com/LinxISA/Linx-TileOP-API/tree/linx/docs/tileop-usage)

## License

See LICENSE.


---

> **当前验证基线**：2026-08-23（437 个已编译 ELF 全量 gfrun 复测；编译器用 blessed
> linx-toolchain-build-latest（dev-llvm15_56 + ADR 0069 B.IOT/B.IOS 重编码）、gfrun 用
> SuperScalarModel（feat/gfrun-cooperative-tmatmul-fp16-bf16-rerun，含 ADR 0069 解码）；
> 总 PASS 366，通过率 83.8%——本次为 blessed 编译器的正确模型配对）

# gfrun 执行结果汇总 — 2026-08-23

## 验证环境

| 组件 | 分支/版本 | Commit |
|---|---|---|
| gfrun / SuperScalarModel | `feat/gfrun-cooperative-tmatmul-fp16-bf16-rerun` | `a5dca25a`（08-23 11:10 构建） |
| llvm-project | `dev-llvm15_56` | `611105f2b` |
| Linx-TileOP-API | detached | `a795b973020d` |

编译器用 AGENTS.md 指定的 **linx-toolchain-build-latest**（`dev-llvm15_56`），含 2026-08-21 双侧落地的 **ADR 0069**：B.IOT/B.IOS 的 `Inst{18-15}` 由 4-bit PE_MASK 重定义为 4-bit **SizeCode**、`Inst{11-9}` 由 3-bit TSize 重定义为 3-bit **PEMode**，SizeCode 0 = 仅源（store 正确编码）。gfrun 用 **SuperScalarModel** 的 `feat/gfrun-cooperative-tmatmul-fp16-bf16-rerun` 分支（`a5dca25a`），其解码器已含 ADR 0069 的 `SizeCode/PEMode/decodeBIOTDstSizeCode`，并附带 fp16/bf16 TMATMUL 支持。**本次为 blessed 编译器（ADR 0069）的正确模型配对**——见下文"与 08-21 差异"中的三方对比。执行：`gfrun -t 1 -f <elf>`，multi_thread 加 `-s softcore.multiThreadNum=4`，单 ELF 90s 超时。PASS = 退出码 0 + `Reach the End of Benchmark` + `R2 = 0`。

## 总体结果

| 范围 | ELF 数 | PASS | FAIL | TIMEOUT | 通过率 |
|---|---:|---:|---:|---:|---:|
| microbenchmark | 348 | 302 | 44 | 2 | 86.8% |
| one-level | 89 | 64 | 25 | 0 | 71.9% |
| **合计** | **437** | **366** | **69** | **2** | **83.8%** |

## 分类结果

| 类别 | 总数 | PASS | FAIL | TIMEOUT |
|---|---:|---:|---:|---:|
| microbenchmark/scalar | 124 | 124 | 0 | 0 |
| microbenchmark/cube | 6 | 2 | 4 | 0 |
| microbenchmark/fixp | 63 | 25 | 36 | 2 |
| microbenchmark/memory | 19 | 17 | 2 | 0 |
| microbenchmark/vector | 136 | 134 | 2 | 0 |
| one-level/broadcast | 6 | 6 | 0 | 0 |
| one-level/concat | 4 | 3 | 1 | 0 |
| one-level/transpose | 4 | 4 | 0 | 0 |
| one-level/gather | 1 | 1 | 0 | 0 |
| one-level/matmul | 16 | 9 | 7 | 0 |
| one-level/deepseek | 20 | 14 | 6 | 0 |
| one-level/control | 6 | 0 | 6 | 0 |
| one-level/fa | 10 | 10 | 0 | 0 |
| one-level/flashMLA | 2 | 2 | 0 | 0 |
| one-level/reduction | 5 | 5 | 0 | 0 |
| one-level/element_wise | 1 | 1 | 0 | 0 |
| one-level/multi_thread/fa | 3 | 3 | 0 | 0 |
| one-level/multi_thread/matmul | 9 | 4 | 5 | 0 |
| one-level/multi_thread/vec | 2 | 2 | 0 | 0 |

> 注：`norm` suite 未被 `compile_all.sh` 驱动收录，无 ELF；`fa/sfa`(2)、`fa/fa_hif4`、`sort/topk`、`deepseek/sinkhorn_fwd`/`topk_gate`/`expand_to_fused` 编译失败，无 ELF 未计入上表（见下"编译覆盖"）。

## 编译覆盖

成功生成 ELF：437 个（microbenchmark 348 + one-level 89）。编译失败、未进入 gfrun（13 个，逐个复现确认）——失败集与 08-21（老 toolchain）**完全一致**；ADR 0069 重做 B.IOT/B.IOS SizeCode+PEMode 后，`mgather_mask`/`mscatter_mask`/`sinkhorn_fwd` 的"Match Instruction Error / unknown operand"仍存在（汇编器 mask 匹配问题独立于 SizeCode 重编码）：

| # | 用例 | 报错位置 | 根因 |
|---|---|---|---|
| 1 | mgather_mask_fp16 | template_asm.hpp:507 | B.IOT mask 不被后端汇编器匹配（Match Instruction Error） |
| 2 | mgather_mask_fp32 | template_asm.hpp:507 | 同上 |
| 3 | mscatter_mask_fp16 | template_asm.hpp:538 | 同上（scatter 方向） |
| 4 | mscatter_mask_fp32 | template_asm.hpp:538 | 同上 |
| 5 | tci_i16 | template_asm.hpp:6437 | TCI `ValidRow==1` 断言（short 16×16） |
| 6 | tci_i32 | template_asm.hpp:6437 | TCI `ValidRow==1` 断言（int 16×16） |
| 7 | fa/sfa Sq=256 | template_asm.hpp:2261 | TMATMUL `output shape must be A.Rows x B.Cols` |
| 8 | fa/sfa Sq=512 | template_asm.hpp:2261 | 同上 |
| 9 | fa/fa_hif4 | fa_hif4.hpp:55 / pto_tile.hpp:716,721 | `QuantType` 未声明 + fp4 tile 行/对齐断言 + TLOAD 无匹配 |
| 10 | sort/topk | topk.hpp:51,93 | `no matching function for call to 'TLOAD'`（需 global_tensor 包装） |
| 11 | deepseek/sinkhorn_fwd | template_asm.hpp:7142,5774 | `unknown operand`（B.IOT mask） |
| 12 | deepseek/topk_gate | template_asm.hpp:6437 | TCI `ValidRow==1` 断言 + clang 前端 exit 134 |
| 13 | deepseek/expand_to_fused | — | clang 前端 SIGABRT（编译器内部崩溃） |

## 运行失败清单（69 FAIL + 2 TIMEOUT）

全部为模型侧限制（断言/未建模/超时），与编译器无关。按类别与失败性质分组：

### microbenchmark/cube — 4 FAIL（CUBE 目的容量断言）
`tmatmul_fp16_64x64x64`、`tmatmul_bias_fp16_64x64x64`、`tmatmul_mx_fp16_64x64x64`、`tmatmul_i8_64x64x64` —— 均 `destinationBytes != 0 && dimensionsArePowersOfTwo ...` 断言。

### microbenchmark/fixp — 36 FAIL + 2 TIMEOUT（fixp tmatmul 系列，最大头）
- **CUBE 源操作数计数 `srcTile.size()+sharedRight==requiredSources`（17）**：`fixp_tmatmul_f16_prelu`、`fixp_tmatmul_s8_prelu`、`fixp_tmatmul_rowmax_init`、`fixp_tmatmul_v_deqf16`、`fixp_tmatmul_v_qf_{f32,f16,bf16,fp8,hif8,s16,s4,s8}`、`fixp_tmatmul_v_qs_bf16`、`fixp_tmatmul_v_reqs8`、`fixp_tmatmul_v_s8_relu`、`fixp_tmatmul_v_shifts16`、`fixp_tmatmul_vqf_s8_prelu`。
- **CUBE 目的容量 `destinationBytes != 0`（8）**：`fixp_tmatmul_gemv`、`fixp_tmatmul_gemv_{acc,bias,mx,mx_acc,mx_bias,mx_s8,s8}`。
- **fp8/fp4 对 dtype `block->dataType==FP32`（4）**：`fixp_tmatmul_mx`、`fixp_tmatmul_mx_s8`、`fixp_tmatmul_mxacc`、`fixp_tmatmul_mxbias`。
- **累加器 dtype `accInfo dataType==FP32||INT32`（2）**：`fixp_tmatmul_acc`、`fixp_tmatmul_acc_s8`。
- **源 validRow `source validRow==1||m`（2）**：`fixp_tmatmul_bias`、`fixp_tmatmul_bias_s8`。
- **缺 FPATR 描述符 `hasFixpAttr`（1）**：`fixp_tmatmul_rowgroup_maxabs`。
- **gfrun 崩溃 rc=134（2）**：`fixp_tmatmul_s_qf_hif8`、`fixp_tmatmul_s_qf_s4`。
- **TIMEOUT >90s（2）**：`fixp_tmatmul_shared`、`fixp_tmatmul_s8_shared`（08-21 main 上快速 FAIL，exp 上能跑但超时，疑似 shared-tile tmatmul 路径慢/近死循环，值得单独排查）。

### microbenchmark/memory — 2 FAIL（真·保留 TEPL selector）
`mgather_fp16_16x16`、`mgather_fp32_16x16` —— `reserved/deleted TEPL selector`（MGATHER 是 TMA 块，通用 bIsIllegal 信息；模型侧未实现）。

### microbenchmark/vector — 2 FAIL（THISTOGRAM 后续校验）
`thistogram_i16_16x16`、`thistogram_i32_16x16` —— `reserved/deleted TEPL selector`（THISTOGRAM selector 本身 active，但块完成时 datatype/operand 后续校验置 bIsIllegal）。

### one-level/control — 6 FAIL（低精度 dtype 元组未定义）
`hashtable_lookup_simd_kNum6144_kMaxProbe512_knum_col{256,512,1024}_debug_{on,off}` —— `dataType==INT8||UINT8||INT16 ...` 数据类型元组未定义断言。

### one-level/deepseek — 6 FAIL（CUBE 操作数校验）
- `aux_fi`、`get_fused_mapping`、`group_count`、`mask_indices_by_tp` —— `inst->srcs.size()==3 && inst->dsts.empty() && IsCompatibleLogicalTile` 断言（4）。
- `inplace_unique_group_indices` —— `priorSources==0 && srcs.size()==3 && dsts.size()==1 ...`（1）。
- `swiglu_forward_and_per_token_cast` —— `source->tileInfo->dataType == block->dataType`（1）。

### one-level/matmul — 7 FAIL（CUBE 右描述符 / fp4 对）
- `matmul_A16W4_B1_{M256_N2048_K2048,M512_N1280_K2048,M512_N512_K4096}` —— `rightInfo->validRow==k && rightInfo->validCol==n`（CUBE 右描述符需 K×N，3）。
- `matmul_HIF4_HIF4_MX_NOGATHER{,_REUSEA}_{M256_N2048_K2048,M512_N1280_K4096}` —— `block->dataType==FP32 && (fp8Pair||fp4Pair)`（fp4 对 dtype 未建模，4）。

### one-level/multi_thread/matmul — 5 FAIL（低精度 cooperative TMATMUL 未建模）
- `kernel_multi_thread_matmul_matmul_lowp_FP8_...` —— `srcType==FP32||FP16||BF16` 不支持（1）。
- `..._lowp_{HIF4X2,HIFP8,MXFP4,MXFP8}_...` —— `unsupported cooperative TMATMUL profile`（4）。

### one-level/concat — 1 FAIL（缺结束标记）
`kernel_concat_concat_scatter_DType__half_tM512_IN_SHAPE256_8_OUT_SHAPE256_8` —— 退出码非 0 / 缺 `Reach the End` 标记。

## 本次更新要点

- **编译器回归 blessed（ADR 0069）+ 正确模型配对**：08-21 用老 toolchain（temp/shared-32kb-debug）+ main 模型；本次用 blessed `linx-toolchain-build-latest`（`dev-llvm15_56` + ADR 0069 B.IOT 重编码）+ exp 模型（`feat/...-tmatmul-fp16-bf16-rerun`，含 ADR 0069 解码）。exp 是 blessed 编译器的正确配对。
- **编译覆盖与 08-21 一致（437 ELF，同 13 个编译失败）**：ADR 0069 重做 B.IOT SizeCode+PEMode 后，4 个 B.IOT-mask 编译失败（`mgather_mask`/`mscatter_mask`×2、`sinkhorn_fwd`）仍存在——汇编器 mask 匹配独立于 SizeCode 重编码。tci/sfa/fa_hif4/topk/topk_gate/expand_to_fused 历史失败不变。
- **运行 339→366 PASS（+27）**：增量来自 exp 模型补的功能——fa 全过（multi_thread/fa bf16/fp16 从 FAIL→PASS，08-21 因 main 模型 TMATMUL FP32-only 而 FAIL）、multi_thread/matmul lowp 从 8→4 FAIL（仍 5 个低精度 cooperative profile 未建模）。
- **剩余 69 FAIL/2 TIMEOUT 纯为模型侧限制**：fixp tmatmul 系列（36F+2T，CUBE 源/目的容量、fp4/fp8 对 dtype、累加器 dtype）、cube 4（目的容量）、control 6（INT8/16 dtype 元组）、deepseek 6（CUBE 操作数校验）、matmul 7（右描述符 K×N + fp4 对）、multi_thread/matmul 5（低精度 cooperative）、memory/vector 4（MGATHER/THISTOGRAM 模型未实现）、concat 1（fp16 scatter 缺标记）。
- **无编译器侧回归**：所有 FAIL 在 08-21（老 compiler+main）上同样 FAIL（69 个 `FAIL→FAIL`），是持久模型边界。

## 与 2026-08-21 基线的差异

> 本轮三方对比，揭示 ADR 0069 的编译器↔模型版本配对关系：
> - **08-21**：老 compiler（`a84c4d10a`，无 ADR 0069）+ main 模型（`7b691d4d`，无 ADR 0069）= 版本匹配 → **339 P/98 F/0 T**。
> - **08-23（本轮）**：blessed compiler（`611105f2b`，**含 ADR 0069**）+ exp 模型（`a5dca25a`，**含 ADR 0069**）= 版本匹配 → **366 P/69 F/2 T**。
> - **08-23 旁证（blessed + main）**：blessed compiler + main 模型 = **版本错位** → 仅 **124 P/313 F/0 T**，其中 **262 个为 `reserved/deleted TEPL selector`**——即 ADR 0069 新 B.IOT 编码（store 的 SizeCode=0 仅源）被 main 模型按旧语义误读为 `ACC<0KB>` 目的 → bIsIllegal。换 exp 模型后这 262 个中 242 个翻 PASS，证实纯属编译器领先、模型落后的编码错位，非 benchmark 代码回归。

| 类别 | 08-21 (ELF/P/F) | 08-23 (ELF/P/F/T) | 变化 |
|---|---|---|---|
| micro/scalar | 124/124/0 | 124/124/0/0 | — |
| micro/cube | 6/2/4 | 6/2/4/0 | — |
| micro/fixp | 63/4/59 | 63/25/36/2 | PASS +21 / FAIL −23 / +2 TIMEOUT（exp 补 fixp tmatmul 形态） |
| micro/memory | 19/17/2 | 19/17/2/0 | — |
| micro/vector | 136/132/4 | 136/134/2/0 | PASS +2 / FAIL −2（ADR 0069 B.IOT 解码修复 vector tile store） |
| one-level/broadcast | 6/6/0 | 6/6/0/0 | — |
| one-level/concat | 4/3/1 | 4/3/1/0 | — |
| one-level/control | 6/0/6 | 6/0/6/0 | — |
| one-level/deepseek | 20/14/6 | 20/14/6/0 | — |
| one-level/element_wise | 1/1/0 | 1/1/0/0 | — |
| one-level/fa | 10/10/0 | 10/10/0/0 | — |
| one-level/matmul | 16/9/7 | 16/9/7/0 | — |
| one-level/multi_thread/fa | 3/1/2 | 3/3/0/0 | PASS +2（bf16/fp16，exp 补 TMATMUL fp16/bf16） |
| one-level/multi_thread/matmul | 9/2/7 | 9/4/5/0 | PASS +2 / FAIL −2（部分 lowp 仍不支持） |

> 其余类别（flashMLA / gather / reduction / transpose / multi_thread/vec）与 08-21 完全一致。micro/vector 的 +2 PASS、micro/fixp 的 +21 PASS 是本轮两大正向变化，均由 exp 模型（ADR 0069 + fp16/bf16 TMATMUL）解释；老 compiler 下这些 tile store 在 main 模型本就 PASS，故 08-21 未体现。

---

# 历史验证记录 — 2026-08-21

> **历史基线**：gfrun `SuperScalarModel-main`（main `7b691d4d`）、llvm `temp/shared-32kb-debug` `a84c4d10a`、TileOP-API `temp/shared-32kb-debug` `ffa257738f`（linx-toolchain-build，32 KiB shared），437 ELF，339 PASS / 98 FAIL / 0 TIMEOUT，通过率 77.6%。注：08-21 用老 compiler（无 ADR 0069）+ main 模型（无 ADR 0069），版本匹配；blessed compiler + main 模型会因 ADR 0069 B.IOT 编码错位骤降至 124 PASS（见 08-23 段三方对比）。

# gfrun 执行结果汇总 — 2026-08-21

## 验证环境

| 组件 | 分支/版本 | Commit |
|---|---|---|
| gfrun / SuperScalarModel | `main` | `7b691d4d` |
| llvm-project | `temp/shared-32kb-debug` | `a84c4d10a` |
| Linx-TileOP-API | `temp/shared-32kb-debug` | `ffa257738f` |

编译器本轮换用 **linx-toolchain-build**（`temp/shared-32kb-debug`，TLOAD Shared 放宽至 32 KiB），非 AGENTS.md 指定的 `linx-toolchain-build-latest`（`dev-llvm15_56`，8 KB cap）；gfrun 换用 **SuperScalarModel-main**（`main` 分支 `7b691d4d`，08-21 11:13 构建），非 08-20 的 `exp` 分支（`5a64c34d`）。**故与 08-20 的差异主要反映工具链+模型切换，非纯代码回归。** 执行：`gfrun -t 1 -f <elf>`，multi_thread 加 `-s softcore.multiThreadNum=4`，单 ELF 90s 超时。PASS = 退出码 0 + `Reach the End of Benchmark` + `R2 = 0`。

## 总体结果

| 范围 | ELF 数 | PASS | FAIL | TIMEOUT | 通过率 |
|---|---:|---:|---:|---:|---:|
| microbenchmark | 348 | 279 | 69 | 0 | 80.2% |
| one-level | 89 | 60 | 29 | 0 | 67.4% |
| **合计** | **437** | **339** | **98** | **0** | **77.6%** |

## 分类结果

| 类别 | 总数 | PASS | FAIL | TIMEOUT |
|---|---:|---:|---:|---:|
| microbenchmark/scalar | 124 | 124 | 0 | 0 |
| microbenchmark/cube | 6 | 2 | 4 | 0 |
| microbenchmark/fixp | 63 | 4 | 59 | 0 |
| microbenchmark/memory | 19 | 17 | 2 | 0 |
| microbenchmark/vector | 136 | 132 | 4 | 0 |
| one-level/broadcast | 6 | 6 | 0 | 0 |
| one-level/concat | 4 | 3 | 1 | 0 |
| one-level/transpose | 4 | 4 | 0 | 0 |
| one-level/gather | 1 | 1 | 0 | 0 |
| one-level/matmul | 16 | 9 | 7 | 0 |
| one-level/deepseek | 20 | 14 | 6 | 0 |
| one-level/control | 6 | 0 | 6 | 0 |
| one-level/fa | 10 | 10 | 0 | 0 |
| one-level/flashMLA | 2 | 2 | 0 | 0 |
| one-level/reduction | 5 | 5 | 0 | 0 |
| one-level/element_wise | 1 | 1 | 0 | 0 |
| one-level/multi_thread/fa | 3 | 1 | 2 | 0 |
| one-level/multi_thread/matmul | 9 | 2 | 7 | 0 |
| one-level/multi_thread/vec | 2 | 2 | 0 | 0 |

> 注：`norm` suite 未生成 ELF（未被 `compile_all.sh` 驱动收录，08-20 计 1）；`fa/sfa`(2)、`fa/fa_hif4`、`sort/topk`、`deepseek/sinkhorn_fwd`/`topk_gate`/`expand_to_fused` 编译失败，无 ELF 未计入上表。

## 编译覆盖

成功生成 ELF：437 个（microbenchmark 348 + one-level 89）。编译失败、未进入 gfrun（13 个，逐个复现确认）：

| # | 用例 | 报错位置 | 根因 | 08-20 | 类型 |
|---|---|---|---|---|---|
| 1 | mgather_mask_fp16 | template_asm.hpp:507 | B.IOT mask 不被后端汇编器匹配 | 同失败 | 后端 |
| 2 | mgather_mask_fp32 | template_asm.hpp:507 | 同上 | 同失败 | 后端 |
| 3 | mscatter_mask_fp16 | template_asm.hpp:538 | 同上（scatter 方向） | 同失败 | 后端 |
| 4 | mscatter_mask_fp32 | template_asm.hpp:538 | 同上 | 同失败 | 后端 |
| 5 | tci_i16 | template_asm.hpp:6437 | TCI `ValidRow==1` 断言（short 16×16） | 编译通过 | 工具链 |
| 6 | tci_i32 | template_asm.hpp:6437 | TCI `ValidRow==1` 断言（int 16×16） | 编译通过 | 工具链 |
| 7 | fa/sfa Sq=256 | template_asm.hpp:2261 | TMATMUL `output shape must be A.Rows x B.Cols` | 同失败 | kernel |
| 8 | fa/sfa Sq=512 | template_asm.hpp:2261 | 同上 | 同失败 | kernel |
| 9 | fa/fa_hif4 | pto_tile.hpp:716,721 | fp4 tile `Rows%InnerRows==0` / 32B 对齐 | 同失败 | kernel |
| 10 | sort/topk | topk.hpp:51,93 | `no matching function for call to 'TLOAD'`（需 global_tensor 包装） | 同失败 | 测试侧 |
| 11 | deepseek/sinkhorn_fwd | template_asm.hpp:5664 | `unknown operand`（B.IOT mask） | 同失败 | 后端 |
| 12 | deepseek/topk_gate | template_asm.hpp:6437 | TCI `ValidRow==1` 断言 + clang frontend exit 134 | 编译通过 | 工具链 |
| 13 | deepseek/expand_to_fused | — | clang 前端 SIGABRT（编译器内部崩溃） | 同失败 | 工具链 |

> 9 个与 08-20 一致（历史失败）。工具链切换引入 3 个新增编译失败（#5 #6 `tci_i16/i32`、#12 `topk_gate`），均为旧 TileOP（`ffa257738`）收紧 TCI `ValidRow==1` 所致；同时 08-20 失败的 `fixp/lrelu_only`（B.IOR zero stride）与 `matmul_gmma`（`TMATMUL_FIXP` 改名）在旧工具链下恢复编译。一进一出反映 `temp/shared-32kb-debug` 与 `dev-llvm15_56` 两条分支的检查差异：旧分支放宽 shared tile 至 32 KiB（big-tile multi_thread 可编译），但收紧 TCI ValidRow。

## 本次更新要点

- **环境切换（最重要）**：编译器 `linx-toolchain-build-latest`（`dev-llvm15_56`，8 KB TLOAD Shared）→ `linx-toolchain-build`（`temp/shared-32kb-debug`，32 KiB shared）；gfrun `exp` 分支（`5a64c34d`）→ `main` 分支（`7b691d4d`）。与 08-20 的差异主要来自工具链+模型切换，非纯代码回归。
- **编译覆盖 +4（433→437）**：one-level +6 全来自 multi_thread 大 tile（fa/matmul 的 bf16/fp16/lowp 在 32 KiB-shared 工具链下首次可编译）；非 multi_thread one-level −4（`topk_gate` 编译失败、`element_wise`/`fa_hif4`/`norm` 编译覆盖下降）。micro −2（`tci_i16/i32` 旧 TileOP TCI ValidRow 断言；`fixp/lrelu_only` 反而恢复编译）。
- **micro 结果与 08-20 完全一致（279P/69F）**：scalar 124/124、cube 2/6、fixp 4/63、memory 17/19 逐例不变；vector 134→132 PASS 仅因 `tci`×2 编译失败（非运行回归）。
- **one-level FAIL +9（20→29）**：全部来自新编译的大 tile multi_thread bf16/fp16/lowp（gfrun TMATMUL 仅支持 FP32，bf16/fp16 触发 `dataType==FP32` 断言；5 个 lowp 触发 `unsupported cooperative TMATMUL profile`）+ concat fp16 scatter 1 例（main 模型未跑出结束标记，08-20 exp 模型 PASS）。
- **multi_thread 大 tile**：float fa/matmul（shared+reuseB）PASS；bf16/fp16 FAIL（TMATMUL FP32-only，已知模型限制）；5 lowp FAIL（lowp 量化 TMATMUL 未建模）。
- **无 scalar/cube/fixp/memory 新增运行回归**；持续模型限制不变（cube 2/6、fixp 4/63、control 0/6、deepseek 14/20、matmul 9/16）。

## 与 2026-08-20 基线的差异

> 08-20 用 blessed `linx-toolchain-build-latest`（8 KB）+ gfrun `exp`（`5a64c34d`）；本轮用 `linx-toolchain-build`（32 KiB）+ gfrun `main`（`7b691d4d`）。下表 delta 均由工具链+模型切换解释，无代码层回归或修复。

| 类别 | 08-20 (ELF/P/F) | 08-21 (ELF/P/F) | 变化 |
|---|---|---|---|
| micro/scalar | 124/124/0 | 124/124/0 | — |
| micro/cube | 6/2/4 | 6/2/4 | — |
| micro/fixp | 63/4/59 | 63/4/59 | — |
| micro/memory | 19/17/2 | 19/17/2 | — |
| micro/vector | 138/134/4 | 136/132/4 | 编译 −2（tci）→ PASS −2 |
| one-level/broadcast | 6/6/0 | 6/6/0 | — |
| one-level/concat | 4/4/0 | 4/3/1 | PASS −1（fp16 scatter，main 模型） |
| one-level/control | 6/0/6 | 6/0/6 | — |
| one-level/deepseek | 21/14/7 | 20/14/6 | 编译 −1（topk_gate）→ FAIL −1 |
| one-level/element_wise | 2/2/0 | 1/1/0 | 编译 −1 |
| one-level/fa | 11/11/0 | 10/10/0 | 编译 −1（fa_hif4） |
| one-level/matmul | 16/9/7 | 16/9/7 | — |
| one-level/norm | 1/1/0 | 0/0/0 | 编译 −1（suite 未驱动） |
| one-level/multi_thread/fa | 1/1/0 | 3/1/2 | +2 编译 → +2 FAIL（bf16/fp16） |
| one-level/multi_thread/matmul | 1/1/0 | 9/2/7 | +8 编译 → +7 FAIL（bf16/fp16/lowp） |

> 其余类别（flashMLA / gather / reduction / transpose / multi_thread/vec）与 08-20 完全一致。micro 5 类中 4 类逐例不变；唯一变化的 vector 也仅是编译覆盖（tci 编译失败），非运行回归。

---

# 历史验证记录 — 2026-08-20

> **历史基线**：gfrun `exp/shared-capacity-lb-semantics-20260819` `5a64c34d`、llvm `dev-llvm15_56` `b945a5d0`、TileOP-API detached `c02dae65`（blessed `linx-toolchain-build-latest` 工具链），433 ELF，344 PASS / 89 FAIL / 0 TIMEOUT，通过率 79.4%。

# gfrun 执行结果汇总 — 2026-08-20

## 验证环境

| 组件 | 分支/版本 | Commit |
|---|---|---|
| gfrun / SuperScalarModel | `exp/shared-capacity-lb-semantics-20260819` | `5a64c34d` |
| llvm-project | `dev-llvm15_56` | `b945a5d0` |
| Linx-TileOP-API | detached HEAD | `c02dae65` |

gfrun 08-20 19:28 重编（`5a64c34d`「honor zero B.IOR stride for raw tile spill」）；工具链 08-20 09:47 重编本轮未变。执行：`gfrun -t 1 -f <elf>`，multi_thread 加 `-s softcore.multiThreadNum=4`，单 ELF 90s 超时。PASS = 退出码 0 + `Reach the End of Benchmark` + `R2 = 0`。

## 总体结果

| 范围 | ELF 数 | PASS | FAIL | TIMEOUT | 通过率 |
|---|---:|---:|---:|---:|---:|
| microbenchmark | 350 | 281 | 69 | 0 | 80.3% |
| one-level | 83 | 63 | 20 | 0 | 75.9% |
| **合计** | **433** | **344** | **89** | **0** | **79.4%** |

## 分类结果

| 类别 | 总数 | PASS | FAIL | TIMEOUT |
|---|---:|---:|---:|---:|
| microbenchmark/scalar | 124 | 124 | 0 | 0 |
| microbenchmark/cube | 6 | 2 | 4 | 0 |
| microbenchmark/fixp | 63 | 4 | 59 | 0 |
| microbenchmark/memory | 19 | 17 | 2 | 0 |
| microbenchmark/vector | 138 | 134 | 4 | 0 |
| one-level/broadcast | 6 | 6 | 0 | 0 |
| one-level/concat | 4 | 4 | 0 | 0 |
| one-level/transpose | 4 | 4 | 0 | 0 |
| one-level/gather | 1 | 1 | 0 | 0 |
| one-level/matmul | 16 | 9 | 7 | 0 |
| one-level/deepseek | 21 | 14 | 7 | 0 |
| one-level/control | 6 | 0 | 6 | 0 |
| one-level/fa | 11 | 11 | 0 | 0 |
| one-level/flashMLA | 2 | 2 | 0 | 0 |
| one-level/reduction | 5 | 5 | 0 | 0 |
| one-level/element_wise | 2 | 2 | 0 | 0 |
| one-level/norm | 1 | 1 | 0 | 0 |
| one-level/multi_thread/fa | 1 | 1 | 0 | 0 |
| one-level/multi_thread/matmul | 1 | 1 | 0 | 0 |
| one-level/multi_thread/vec | 2 | 2 | 0 | 0 |

> 注：fa `sfa`(2) 因 TMATMUL 形状契约编译失败；`fa_HIF4_HIF4`、`sort`、`matmul_gmma`、`deepseek sinkhorn_fwd`/`expand_to_fused` 08-19 即已编译失败，均无 ELF 未计入上表。

## 编译覆盖

成功生成 ELF：433 个（microbenchmark 350 + one-level 83）。编译失败、未进入 gfrun（12 个，逐个复现确认）：

| # | 用例 | 报错位置 | 根因 | 08-19 | 可修方 |
|---|---|---|---|---|---|
| 1 | mgather_mask_fp16 | template_asm.hpp:502 | B.IOT mask=15 不被后端汇编器匹配 | 同失败 | 后端 |
| 2 | mgather_mask_fp32 | template_asm.hpp:502 | 同上 | 同失败 | 后端 |
| 3 | mscatter_mask_fp16 | template_asm.hpp:533 | 同上（scatter 方向） | 同失败 | 后端 |
| 4 | mscatter_mask_fp32 | template_asm.hpp:533 | 同上 | 同失败 | 后端 |
| 5 | fixp/lrelu_only | template_asm.hpp:3577 | B.IOR `[zero,a1],[]` 不被匹配（FPATR local） | 同失败 | 后端 |
| 6 | fa/sfa Sq=256 | template_asm.hpp:2191 | TMATMUL static_assert：C::Cols≠B::Cols | 编译通过 | kernel |
| 7 | fa/sfa Sq=512 | template_asm.hpp:2191 | 同上 | 编译通过 | kernel |
| 8 | fa/fa_HIF4_HIF4 | pto_tile.hpp:716,721 | fp4 tile Tm=8 不满足 32B 对齐 | 同失败 | kernel |
| 9 | sort/topk | topk.hpp:93 | TLOAD 拒绝裸指针（需 global_tensor 包装） | 同失败 | 测试侧 |
| 10 | matmul/GMMA | matmul_gmma.cpp:121 | `TMATMUL_FIXP` 已改名（PTO 0.58→`TMATMUL`） | 同失败 | 测试侧 |
| 11 | deepseek/sinkhorn_fwd | template_asm.hpp:5704 | B.IOT mask=15,last 不被匹配 | 同失败 | 后端 |
| 12 | deepseek/expand_to_fused | — | clang 前端 SIGABRT（编译器内部崩溃） | 同失败 | 工具链 |

> 10 个与 08-19 一致（历史失败），仅 #6 #7（fa/sfa×2）为本轮新增编译回归（TMATMUL `A.Rows x B.Cols` 形状契约，TileOP-API `c02dae6` 收紧）。#9 #10 测试侧可修（一行改名 / 包 global_tensor），#6–8 需 kernel 级形状重构，#1–5 #11 #12 需后端汇编器或工具链修复。

## 本次更新要点

- **gfrun `5a64c34d` 修复 4 例 tM2048 回归**：上轮 `2d467114` 把 B.IOR 行步幅改为按 bytes 解读（符合 ISA spec）后，3 个旧式 broadcast（TCOPYIN/TCOPYOUT 路径）+ 1 个 element_wise/gelu bf16 tM2048 在 `TMAEngine.cpp:161`「invalid raw tile spill transport」断言失败。`5a64c34d` 显式处理零步幅 raw tile spill，4 例全部恢复 PASS。全量 433 ELF 仅这 4 例相对上轮变化，其余 430 个结果完全一致，无新增回归。
- **当日 gfrun 演进**：`78adfe32`（TEPL expand 段错误修复、CUBE 共享输出修复）→ `2d467114`（B.IOR 行步幅改 bytes）→ `5a64c34d`（零步幅 raw tile spill 修复）。micro 全程 281P/69F 不变。
- **对比 08-19 基线**（434 ELF / 341P / 92F / 1T）：PASS +3（concat TIMEOUT→PASS、multi_thread/matmul FAIL→PASS、fa +1 Tm4_Tk8 新增）、FAIL −3、TIMEOUT −1。唯一编译回归 = fa `sfa`×2（TMATMUL 形状契约，与编译器/模型重建无关）。
- **持续模型限制**（与 08-19 一致）：cube 2/6、fixp 4/63、control 0/6、deepseek 14/21、matmul 9/16（A16W4/HIF4 系列）、memory 17/19、vector 134/138（`tabs_i16/i32`、`thistogram_i16/i32`，reserved TEPL selector）。

---

# 历史验证记录

## 2026-08-19

gfrun `01f9ec10`、llvm `86959776b`、TileOP-API `8b2ee78`，434 ELF（350 micro + 84 one-level），**341 PASS / 92 FAIL / 1 TIMEOUT，通过率 78.6%**。

对比 08-18（412 同名用例，归一化 scalar `f32`/`fp32` 命名后）：

| 状态变化 | 数量 |
|---|---:|
| PASS → PASS | 317 |
| FAIL → FAIL | 89 |
| FAIL → PASS | 4 |
| TIMEOUT → TIMEOUT | 1 |
| PASS → FAIL | 1 |

算子通过率变更：4 例 FAIL→PASS（3 个 broadcast `tM2048` + 1 个 BF16 GELU）；1 例 PASS→FAIL（multi_thread `matmul_shared`，按 M 轴四分 local `tileC` 后触发 CUBE D 断言）。另 08-18 独有 17 例、本轮独有 22 例（覆盖扩展）。编译失败 10 个（与本轮 #1–5、#8–12 一致）。

## 2026-08-18

gfrun `a68dba29`、TileOP-API `8b2ee78`（TileDType 修复），429 ELF（342 micro + 82 one-level + 5 multi_thread），**321 PASS / 108 FAIL / 2 TIMEOUT，通过率 74.8%**。首次全量基线。

算子通过率变更（相对 TileDType 修复前）：
- **fa**：10 例由全 FAIL 恢复 PASS（8×`fa_2d_unroll` + 2×`fa_softmax_pto`）。
- **multi_thread**：fa（Sq128/Sq512）+ `matmul_shared` 由 FAIL 恢复，5/5 全通过。
- **flashMLA / norm / reduction**：由 FAIL 恢复，分别 2/2、1/1、5/5。
- **fixp**：由 27/63 退化为 4/63（−23）——先前 27 例"通过"是 4KB 硬编码 tile 下的幻觉通过，TileDType 修复暴露 gfrun CUBE/tmatmul validator 对非 4KB tile 的契约偏差。
- 106 个 FAIL 均为 gfrun 功能模型校验断言（非算子 bug、非非法指令）。
