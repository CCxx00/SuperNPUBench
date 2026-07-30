---
name: linx-bench-workflow
description: Use when building the Linx compiler toolchain (linx-toolchain-build), compiling SuperNPUBench operators to ELF binaries and disassembly, building the SuperScalarModel simulator (gfrun/gfsim), or running simulation tests on compiled ELF files. Trigger keywords include "linx-toolchain-build", "SuperNPUBench", "SuperScalarModel", "gfrun", "gfsim", "linx_blockisa_llvm_musl", "compile benchmark", "microbenchmark", "one-level-arch", "toolchain build", "linx64v5", "simulator run", "tile ISA". Use ONLY for the cross-repo Linx/PTO-ISA toolchain → benchmark → simulator workflow.
---

# Linx 工具链 → Benchmark 编译 → 仿真器运行 端到端工作流

本 skill 记录从三个仓库出发的完整工作流：

```
linx-toolchain-build  →  SuperNPUBench  →  SuperScalarModel
   (编译器工具链)         (算子编译)         (仿真器运行)
```

三个仓库为同级目录，假设位于同一 `GitHub/` 根下：

| 仓库 | 作用 |
|------|------|
| `linx-toolchain-build/` | 构建 Linx LLVM + musl 工具链 (`linx_blockisa_llvm_musl`) |
| `SuperNPUBench/` | 算子库与 benchmark，编译生成 ELF 二进制 + 反汇编 |
| `SuperScalarModel/` | PTO-ISA 仿真器，`gfrun`(功能模型) + `gfsim`(周期时序模型) |

---

## 阶段一：构建编译器工具链 (linx-toolchain-build)

### 环境要求

- macOS: 需 GNU make >= 4 (`brew install make`，用 `gmake`) 和 GNU tar (`brew install gnu-tar`)
- Linux: `sudo apt-get install -y git make cmake ninja-build gcc g++ python3 autoconf m4`

### 步骤

```bash
cd linx-toolchain-build

# 1. 初始化组件源码（克隆 llvm-project/musl/jemalloc/linux/Linx-TileOP-API）
make init-src

# 2. 构建工具链（仅支持 linx64v5-linux-musl）
make WITH_TARGET=linx64v5-linux-musl

# 3. (可选) 打包
make package
```

构建由 `stamps/` 下的 stamp 文件跟踪进度，重新运行 `make` 会从最后完成的步骤恢复。
`make clean` 从头重建。

### 产物

```
output/linx_blockisa_llvm_musl/
├── bin/        # clang, clang++, ld.lld, llvm-ar/nm/ranlib, llvm-objdump
├── lib/        # clang runtime, libc++, ...
└── sysroot/    # musl + kernel headers + runtime libs
```

- 编译器: clang-15, target `linx64v5-unknown-linux-musl`
- 关键编译选项: `-mlxbc -fenable-matrix -O2 -mllvm -enable-all-vector-as-tilereg=true -std=c++20`

### 验证

```bash
export COMPILER_DIR=$(pwd)/output/linx_blockisa_llvm_musl/bin
$COMPILER_DIR/clang --version
# clang version 15.0.4 (linx64v5-musl-local ...)
# Target: linx64v5-unknown-linux-musl
```

### macOS 已知问题

- Apple clang 拒绝 `sancov.cpp` 中的 initializer-list → 改 `SpecialCaseList::createOrDie({{...}}` 为 `std::vector<std::string>{...}`，再 `ninja -C build/build-llvm-musl` 恢复
- 内核 headers 步骤需要 GNU Make >= 4.0

---

## 阶段二：编译 SuperNPUBench 算子

### 环境设置

```bash
export COMPILER_DIR=/path/to/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
cd SuperNPUBench
```

### A. benchmark/one-level-arch (PTO ISA 算子)

13 个算子类别：matmul, broadcast, concat, gather, transpose, gelu, reduction(×4), control, fa, sort。

```bash
# 全部编译（compile.all 内部调用 make 生成 ELF）
bash benchmark/one-level-arch/compile_all.sh

# 或从仓库根用顶层脚本
./compile_all.sh one-level

# 单个算子单条用例
cd benchmark/one-level-arch/test/kernel/matmul
make TESTCASE=matmul TYPE=HIF4_HIF4 VER=MX_NOGATHER M=256 N=2048 K=2048 tM=64 tN=64 tK=64
```

产物路径: `benchmark/one-level-arch/output/kernel/<operator>/elf/<operator_category>/<name>.elf`

### B. microbenchmark (指令级微基准)

4 个 ISA 族：cube(CUBE, 9 cases), vector(TEPL), memory(TLSU), scalar(GPR, 124 cases)。

```bash
# 必须从 microbenchmark/ 目录运行（脚本用相对路径 cd）
cd microbenchmark
bash compile_all.sh all          # 全部
bash compile_all.sh scalar       # 单族

# 单条用例
cd microbenchmark/scalar
make TESTCASE=add_i32_lat
```

产物路径: `output/microbenchmark/<family>/elf/<family>/<case>.elf`

### C. 生成反汇编文件 (.diss)

`compile.all` 脚本默认只做 `all`(ELF)。反汇编需额外步骤：

```bash
OBJDUMP=$COMPILER_DIR/llvm-objdump

# 方式1：单条用例的 diss 目标
make TESTCASE=<name> diss    # 生成 <target>.elf.diss

# 方式2：批量生成所有 ELF 的反汇编（推荐）
find benchmark/one-level-arch/output output/microbenchmark \
  -name "*.elf" -type f ! -name "*.diss" | while read -r elf; do
  "$OBJDUMP" -dl "$elf" > "${elf}.diss"
done
```

### 已知编译失败

| 算子 | 原因 |
|------|------|
| `fa` (fa_2d_unroll) | 编译器 Assertion crash (Issue #6)，避免 Ydim=1 |
| `sort` (topk) | `TEXPANDSCALAR` 未在工具链中暴露 |
| micro `tmatmul_acc` | 工具链 matmul.ac 后端 crash |
| micro vector 子集 | 部分 TEPL opcode 未暴露（见 microbenchmark/README.md VECTOR_SKIP） |

### 构建目标参考 (Makefile)

| 目标 | 作用 |
|------|------|
| `make` / `all` | 编译生成 ELF |
| `make diss` | 生成反汇编 (`llvm-objdump -dl`) |
| `make sim` | 在 QEMU 中运行 (PLAT=linx) |
| `make clean` | 清理当前算子 |
| `make clean_all` | 清理所有 |

### Makefile 参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `TESTCASE` | 测试用例名 | `matmul`, `fa_2d_unroll` |
| `TYPE` | 算子类型 | `HIF4_HIF4`, `A16W4`, `MASK` |
| `MODE` | 算子模式 | `MASK_FP32` |
| `M/N/K` | 矩阵维度 | `M=256 N=2048 K=2048` |
| `tM/tN/tK` | 分块大小 | `tM=64 tN=64 tK=64` |
| `COMPILER_DIR` | 编译器路径 | 必须设置 |
| `PLAT` | 平台 | `linx`(默认), `cpu` |

---

## 阶段三：编译 SuperScalarModel 仿真器

### 环境要求

| 组件 | 要求 |
|------|------|
| C++ 编译器 | GCC 8+ 或 Clang 10+ |
| CMake | >= 3.10 |
| Python | 3.8+ |
| toml | `pip install toml`（或用项目 `.venv`） |
| libelf | Linux: `libelf-dev`，macOS: `brew install libelf` |

### 步骤

```bash
cd SuperScalarModel

# 推荐方式（配置 + 编译）
python3 build.py all -j8

# 全新重建
python3 build.py all --clean -j8

# 仅编译单个目标
python3 build.py build --target gfrun -j8
python3 build.py build --target gfsim -j8

# 如果系统 python 缺 toml，用项目 venv
.venv/bin/python3 build.py all -j8
```

### 产物

产物输出到源码树（非 `build/`）：

| 二进制 | 源码 | 作用 |
|--------|------|------|
| `bin/gfrun` | `emulator/main.cpp` | 功能模型（正确性验证） |
| `bin/gfsim` | `TimingSim/core/main.cpp` | 周期精确时序模型 |

库文件输出到 `lib/`：`libsoft_core.a`, `libmodel_lib.a`, `libisa_blockisa.a`, `libsoftfloat.a` 等。

### 关键 build.py 选项

| 选项 | 说明 |
|------|------|
| `-j, --parallel` | 并行任务数 |
| `--clean` | 清理后重新配置编译 |
| `-t, --build-type` | `Release`/`Debug`/`RelWithDebInfo` |
| `-O, --opt-level` | `O0`~`O3` |
| `--tests` | 编译单元测试（需 googletest） |
| `--asan` / `--ubsan` | 启用 sanitizer |
| `--warnings-as-errors` | 警告视为错误 |

---

## 阶段四：运行仿真器测试

### gfrun — 功能模型（正确性）

```bash
cd SuperScalarModel

# 基本运行
bin/gfrun -f /path/to/<program>.elf

# 控制
bin/gfrun -f <elf> -X <start_pc_hex> -r <stop_pc_hex>   # PC 范围
bin/gfrun -f <elf> -c <max_blocks> -t 1                  # 块数/trace
```

成功标志：
```
Suaccelss to Reach the End of Benchmark! R2 = 0
```

### gfsim — 周期时序模型

```bash
# 基本运行
bin/gfsim -f /path/to/<program>.elf

# trace 模式
bin/gfsim -f <input> -t 1

# 覆盖配置
bin/gfsim -f <input> -s core.bp_mode=0 core.soc_enable=true

# 纯 tile-op 算子（如 control/hashtable_lookup_simd）需 single-tier 模式
bin/gfsim -f <elf> -s core.singleTierMode=true

# PipeView / SwimLane 可视化
bin/gfsim -f <input> -p 1 --pipefile my_trace      # 指令级
bin/gfsim -f <input> --swimlane 1 --swimfile my_swim  # 块级
```

gfsim 输出 PMU 统计，关键指标：
- `Total Cycles` / `Sim Total Cycles` — 总周期数
- `Cube/Vector/TMA Tileop Total Cycles` — 各引擎周期
- `superScalar Run Tileop Total Cycles` — 运行态周期
- 各类 Stall 统计

### 典型 ELF 路径

```bash
# microbenchmark
bin/gfrun -f ../SuperNPUBench/output/microbenchmark/scalar/elf/scalar/add_i32_lat.elf
bin/gfrun -f ../SuperNPUBench/output/microbenchmark/cube/elf/cube/tmatmul_fp16_64x64x64.elf
bin/gfrun -f ../SuperNPUBench/output/microbenchmark/memory/elf/memory/tload_fp16_16x16.elf

# one-level-arch benchmark
bin/gfrun -f ../SuperNPUBench/benchmark/one-level-arch/output/kernel/matmul/elf/kernel_matmul/matmul_MASK_MASK_FP32_M256_N256_K256_tM32_tN32_tK64.elf
```

---

## 验证基线（参考运行结果）

以下为一次完整流程的实际结果，可作为验证基线：

### 工具链
- clang 15.0.4, target `linx64v5-unknown-linux-musl`
- 安装树 `output/linx_blockisa_llvm_musl/` 完整 (bin/lib/sysroot/include)

### SuperNPUBench 编译产物
| 类别 | ELF 数 | .diss 数 |
|------|--------|----------|
| one-level-arch (11/13 算子成功) | 52 | 52 |
| microbenchmark (cube/vector/memory/scalar) | 293 | 293 |
| **合计** | **345** | **345** |

### gfrun 功能测试（全部通过）

| ELF | Block 数 | Inst 数 | 结果 |
|-----|----------|---------|------|
| micro/scalar/add_i32_lat | 1031 | 9256 | R2=0 ✓ |
| micro/cube/tmatmul_fp16_64x64x64 | 8213 | 123226 | R2=0 ✓ |
| micro/memory/tload_fp16_16x16 | 272 | 3150 | R2=0 ✓ |
| one-level/matmul_MASK_FP32 | 1309 | 6165 | R2=0 ✓ |

### gfsim 时序测试

| ELF | Total Cycles | Cube Cycles | TMA Cycles |
|-----|-------------|-------------|------------|
| micro/scalar/add_i32_lat | 3227 | 0 | 0 |
| one-level/matmul_MASK_FP32 | 41399 | 30082 | 11955 |

---

## 快速复现脚本（一键全流程）

```bash
#!/bin/bash
set -e
WORKSPACE_ROOT=${WORKSPACE_ROOT:-$HOME/src}
TC=$WORKSPACE_ROOT/linx-toolchain-build
BENCH=$WORKSPACE_ROOT/SuperNPUBench
SIM=$WORKSPACE_ROOT/SuperScalarModel
export COMPILER_DIR=$TC/output/linx_blockisa_llvm_musl/bin
OBJDUMP=$COMPILER_DIR/llvm-objdump

# 阶段一：工具链（若已构建则跳过）
cd "$TC" && make WITH_TARGET=linx64v5-linux-musl

# 阶段二A：one-level-arch 编译
cd "$BENCH" && bash benchmark/one-level-arch/compile_all.sh

# 阶段二B：microbenchmark 编译
cd "$BENCH/microbenchmark" && bash compile_all.sh all

# 阶段二C：批量生成反汇编
cd "$BENCH"
find benchmark/one-level-arch/output output/microbenchmark \
  -name "*.elf" -type f ! -name "*.diss" | while read -r elf; do
  "$OBJDUMP" -dl "$elf" > "${elf}.diss"
done

# 阶段三：仿真器编译
cd "$SIM" && .venv/bin/python3 build.py all -j8

# 阶段四：简易运行测试
echo "=== gfrun: add_i32_lat ==="
"$SIM/bin/gfrun" -f "$BENCH/output/microbenchmark/scalar/elf/scalar/add_i32_lat.elf"
echo "=== gfrun: matmul FP32 ==="
"$SIM/bin/gfrun" -f "$BENCH/benchmark/one-level-arch/output/kernel/matmul/elf/kernel_matmul/matmul_MASK_MASK_FP32_M256_N256_K256_tM32_tN32_tK64.elf"
echo "=== gfsim: add_i32_lat ==="
"$SIM/bin/gfsim" -f "$BENCH/output/microbenchmark/scalar/elf/scalar/add_i32_lat.elf"
```

---

## 集成自测（SuperScalarModel 内置）

SuperScalarModel 提供 `scripts/ci_self_test.py` 作为集成回归测试，运行 pass list 中的所有 ELF：

```bash
cd SuperScalarModel

# PR 级（轻量，< 60s/ELF）
python3 scripts/ci_self_test.py --tool gfrun
python3 scripts/ci_self_test.py --tool gfsim

# 编译后测试
python3 scripts/ci_self_test.py --tool gfsim --build

# 按子串过滤单条用例
python3 scripts/ci_self_test.py --tool gfrun --filter tmatmul_fp32

# nightly 级（重量级）
python3 scripts/ci_self_test.py --tool gfsim --pass-list tests/gfsim-pass-list-nightly.txt
```

Pass list 文件: `tests/gfrun-pass-list.txt`, `tests/gfsim-pass-list.txt`。
