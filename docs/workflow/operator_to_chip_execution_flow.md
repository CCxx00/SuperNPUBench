# 从算子到芯片执行：端到端工作流完整指南

> **文档目的**：任何人都可以按照本文档，从零开始完成"构建编译器工具链 → 编译算子生成二进制与反汇编 → 构建仿真器 → 运行仿真测试"的全流程。

本文档覆盖三个仓库的协作流程：

```
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│  linx-toolchain-build   │     │     SuperNPUBench       │     │   SuperScalarModel       │
│                         │     │                         │     │                          │
│  构建 Linx LLVM + musl   │────▶│  算子编译 → ELF + 反汇编 │────▶│  gfrun 功能模型          │
│  编译器工具链             │     │  (benchmark + micro)    │     │  gfsim 周期时序模型       │
│                         │     │                         │     │  仿真运行 + PMU 统计      │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
       阶段一                          阶段二                         阶段三 & 四
```

三个仓库假设为同级目录，位于同一根目录（如 `~/Documents/GitHub/`）下：

```
GitHub/
├── linx-toolchain-build/    # 阶段一
├── SuperNPUBench/           # 阶段二
└── SuperScalarModel/        # 阶段三 & 四
```

---

## 目录

- [0. 环境准备](#0-环境准备)
- [1. 阶段一：构建编译器工具链](#1-阶段一构建编译器工具链)
- [2. 阶段二：编译 SuperNPUBench 算子](#2-阶段二编译-supernpubench-算子)
- [3. 阶段三：构建 SuperScalarModel 仿真器](#3-阶段三构建-superscalarmodel-仿真器)
- [4. 阶段四：运行仿真器测试](#4-阶段四运行仿真器测试)
- [5. 一键全流程脚本](#5-一键全流程脚本)
- [6. 验证基线（参考结果）](#6-验证基线参考结果)
- [7. 故障排查](#7-故障排查)
- [8. 附录：目录结构与文件清单](#8-附录目录结构与文件清单)

---

## 0. 环境准备

### 0.1 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | macOS (Apple Silicon) 或 Linux (x86_64/aarch64) |
| 磁盘空间 | ≥ 20 GB（工具链源码 + 构建产物 + 算子二进制 + 仿真器） |
| 内存 | ≥ 16 GB（LLVM 编译消耗较大） |

### 0.2 macOS 依赖安装

```bash
# Xcode 命令行工具（提供 clang/make）
xcode-select --install

# Homebrew 包
brew install cmake make gnu-tar libelf python@3

# Python 依赖（用于 SuperScalarModel 的 build.py 配置工具）
pip3 install toml
```

> **关键**：macOS 自带的 `make` 是 GNU 3.81，不满足内核头文件步骤要求的 `GNU Make >= 4.0`。
> 安装 `brew install make` 后会提供 `gmake`。构建时用 `gmake` 替代 `make`，或将
> `$(brew --prefix)/opt/make/libexec/gnubin` 加入 `PATH` 前面。

> **关键**：macOS 自带的 `tar` 是 libarchive 版，不支持 `make package` 需要的
> `--format=gnu`。安装 `brew install gnu-tar` 后将
> `$(brew --prefix)/opt/gnu-tar/libexec/gnubin` 加入 `PATH` 前面。

### 0.3 Linux 依赖安装

```bash
sudo apt-get update
sudo apt-get install -y git make cmake ninja-build gcc g++ python3 autoconf m4 libelf-dev

# Python 依赖
pip3 install toml
```

### 0.4 克隆三个仓库

```bash
cd ~/Documents/GitHub   # 或任意工作目录

git clone https://github.com/LinxISA/linx-toolchain-build.git
git clone https://github.com/PTO-ISA/SuperNPUBench.git
git clone https://github.com/LinxISA/SuperScalarModel.git
```

---

## 1. 阶段一：构建编译器工具链

**仓库**：`linx-toolchain-build/`
**产物**：`output/linx_blockisa_llvm_musl/`（完整 LLVM + musl 交叉工具链）
**预计耗时**：首次约 1-3 小时（取决于机器配置）；增量构建数分钟

### 1.1 工具链概述

该工具链为 Linx/PTO-ISA 定制的 LLVM 15 交叉编译器，目标三元组为
`linx64v5-unknown-linux-musl`，支持 BlockISA 块结构化指令集和 Tile 算子。

构建顺序（由 stamp 文件跟踪，支持断点续编）：

```
LLVM/clang/lld → kernel headers → musl → compiler-rt → libc++/libc++abi/libunwind → jemalloc → Linx-TileOP-API headers
```

### 1.2 初始化组件源码

```bash
cd linx-toolchain-build

# 克隆 5 个组件仓库到 src/ 下（首次运行；之后可重复运行拉取更新）
make init-src
```

此命令会在 `src/` 下克隆以下仓库到指定分支：

| 目录 | 仓库 | 分支 |
|------|------|------|
| `src/llvm-project` | `https://github.com/LinxISA/llvm-project.git` | `dev-llvm15_56` |
| `src/musl` | `https://github.com/LinxISA/linx-musl.git` | `linx` |
| `src/jemalloc` | `https://github.com/LinxISA/jemalloc.git` | `linx` |
| `src/linux-linxisa` | `https://github.com/LinxISA/linux.git` | `main` |
| `src/Linx-TileOP-API` | `https://github.com/LinxISA/Linx-TileOP-API.git` | `linx` |

验证：

```bash
ls src/
# 应看到: Linx-TileOP-API  jemalloc  linux-linxisa  llvm-project  musl
```

### 1.3 构建工具链

```bash
# macOS 上用 gmake 替代 make：
gmake WITH_TARGET=linx64v5-linux-musl

# Linux 上：
make WITH_TARGET=linx64v5-linux-musl
```

> 仅支持 `WITH_TARGET=linx64v5-linux-musl` 这一个目标。

构建过程中，每完成一个步骤会在 `stamps/` 下创建对应的 stamp 文件。如果中途中断，
重新运行 `make` 会从最后一个完成的步骤恢复：

```
stamps/
├── build-llvm-musl          # LLVM/clang/lld
├── build-kernel-header      # Linux 内核头文件
├── build-musl               # musl libc
├── build-compiler-rt-musl   # compiler-rt
├── build-libcxx-musl        # libc++
├── build-libcxxabi-musl     # libc++abi
├── build-libunwind-musl     # libunwind
├── build-jemalloc           # jemalloc
└── build-tileopapi          # Linx-TileOP-API 头文件
```

从头重建：

```bash
make clean && make WITH_TARGET=linx64v5-linux-musl
```

### 1.4 验证工具链

```bash
export COMPILER_DIR=$(pwd)/output/linx_blockisa_llvm_musl/bin

# 版本检查
$COMPILER_DIR/clang --version
# 预期输出：
# clang version 15.0.4 (linx64v5-musl-local ...)
# Target: linx64v5-unknown-linux-musl
# Thread model: posix

# 安装树结构检查
ls $COMPILER_DIR/../
# 应看到: bin/  include/  lib/  libexec/  share/  sysroot/

# 关键二进制存在性检查
ls $COMPILER_DIR/{clang,clang++,ld.lld,llvm-objdump,llvm-ar,llvm-nm}
```

### 1.5（可选）打包

```bash
# macOS 需要先确保 gnu-tar 在 PATH 前面
export PATH="$(brew --prefix)/opt/gnu-tar/libexec/gnubin:$PATH"

make package
# 产物: output/linx_blockisa_llvm_musl.tar.gz
```

### 1.6 macOS 特有问题与修复

**问题 1：sancov 编译失败**

Apple clang 拒绝 `llvm/tools/sancov/sancov.cpp` 中的 initializer-list
（`chosen constructor is explicit in copy-initialization`）。

修复：将该文件中两处 `SpecialCaseList::createOrDie({{...}}, ...)` 改为
`SpecialCaseList::createOrDie(std::vector<std::string>{{...}}, ...)`，然后恢复构建：

```bash
ninja -C build/build-llvm-musl
```

**问题 2：内核头文件需要 GNU Make >= 4.0**

```bash
# 检查
make --version    # 如果版本 < 4.0
brew install make # 安装后用 gmake
```

---

## 2. 阶段二：编译 SuperNPUBench 算子

**仓库**：`SuperNPUBench/`
**产物**：ELF 二进制文件 (`.elf`) + 反汇编文件 (`.elf.diss`)
**预计耗时**：全量编译约 5-10 分钟

### 2.1 仓库结构

```
SuperNPUBench/
├── benchmark/
│   ├── one-level-arch/         # PTO ISA 单级 Tile 架构（本流程目标）
│   │   ├── kernels/            # header-only 算子实现
│   │   ├── test/
│   │   │   ├── common/         # 共享 Makefile.common, _start.s, benchmark.h
│   │   │   └── kernel/         # 按算子分目录，各含 compile.all + Makefile + src/
│   │   │       ├── matmul/
│   │   │       ├── broadcast/
│   │   │       ├── concat/
│   │   │       ├── gather/
│   │   │       ├── transpose/
│   │   │       ├── element_wise/
│   │   │       │   └── gelu/
│   │   │       ├── reduction/
│   │   │       │   ├── reducemax_col/
│   │   │       │   ├── reducemax_row/
│   │   │       │   ├── reducesum_col/
│   │   │       │   └── reducesum_row/
│   │   │       ├── control/
│   │   │       ├── fa/
│   │   │       └── sort/
│   │   └── compile_all.sh
│   └── two-level-arch/         # LinxISA 两级 Block ISA（本流程不涉及）
├── microbenchmark/             # 指令级微基准
│   ├── Makefile.common
│   ├── gen_cases.py            # 表驱动用例生成器
│   ├── common/
│   ├── cube/                   # CUBE 族 (BSTART.CUBE)
│   ├── vector/                 # TEPL 族 (BSTART.TEPL)
│   ├── memory/                 # TLSU 族 (BSTART.TLSU)
│   ├── scalar/                 # GPR 标量族 (BSTART.STD/FP)
│   └── compile_all.sh
├── architecture/              # ISA 参考文档
└── compile_all.sh             # 顶层脚本: two-level | one-level | all
```

### 2.2 环境设置

```bash
# 指向阶段一构建的编译器 bin 目录（必须设置）
export COMPILER_DIR=/path/to/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin

# 验证
$COMPILER_DIR/clang --version
# clang version 15.0.4 ... Target: linx64v5-unknown-linux-musl

cd SuperNPUBench
```

### 2.3 编译 benchmark/one-level-arch 全部算子

one-level-arch 包含 13 个算子类别，每个算子目录下有 `compile.all` 脚本，
内含多条 `make TESTCASE=...` 调用。

```bash
cd SuperNPUBench

# 方式 1：直接调用 one-level-arch 的编译脚本
bash benchmark/one-level-arch/compile_all.sh

# 方式 2：通过顶层脚本
bash compile_all.sh one-level
```

脚本会依次进入每个算子目录执行 `bash compile.all`，编译成功的 ELF 输出到：
`benchmark/one-level-arch/output/kernel/<operator>/elf/<category>/<name>.elf`

#### 单个算子 / 单条用例编译

```bash
# 进入算子目录
cd benchmark/one-level-arch/test/kernel/matmul

# 编译单条用例（生成 ELF）
make TESTCASE=matmul TYPE=HIF4_HIF4 VER=MX_NOGATHER M=256 N=2048 K=2048 tM=64 tN=64 tK=64

# 编译某个算子的全部用例
bash compile.all
```

#### 算子分类与 compile.all 位置

| 算子 | 目录 | compile.all 位置 |
|------|------|-------------------|
| matmul | `test/kernel/matmul/` | `test/kernel/matmul/compile.all` |
| broadcast | `test/kernel/broadcast/` | `test/kernel/broadcast/compile.all` |
| concat | `test/kernel/concat/` | `test/kernel/concat/compile.all` |
| gather | `test/kernel/gather/` | `test/kernel/gather/compile.all` |
| transpose | `test/kernel/transpose/` | `test/kernel/transpose/compile.all` |
| gelu | `test/kernel/element_wise/gelu/` | `test/kernel/element_wise/gelu/compile.all` |
| reducemax_col | `test/kernel/reduction/reducemax_col/` | `.../reducemax_col/compile.all` |
| reducemax_row | `test/kernel/reduction/reducemax_row/` | `.../reducemax_row/compile.all` |
| reducesum_col | `test/kernel/reduction/reducesum_col/` | `.../reducesum_col/compile.all` |
| reducesum_row | `test/kernel/reduction/reducesum_row/` | `.../reducesum_row/compile.all` |
| control | `test/kernel/control/` | `test/kernel/control/compile.all` |
| fa | `test/kernel/fa/` | `test/kernel/fa/compile.all` |
| sort | `test/kernel/sort/` | `test/kernel/sort/compile.all` |

### 2.4 编译 microbenchmark 全部用例

microbenchmark 包含 4 个 ISA 族共 293 条用例：

| 族 | Block 类型 | 用例数 | 覆盖 |
|----|-----------|--------|------|
| cube | BSTART.CUBE | 9 | TMATMUL / TMATMUL_BIAS / TMATMUL_MX / ACCCVT |
| vector | BSTART.TEPL | 135 | elementwise / tile-scalar / reduce / expand |
| memory | BSTART.TLSU | 25 | TLOAD / TSTORE / TMOV / MGATHER / MSCATTER (+mask, layout) |
| scalar | BSTART.STD/FP | 124 | int ALU / load-store / float / conversion × throughput+latency |

```bash
# 必须从 microbenchmark/ 目录运行（脚本用相对路径 cd 进入子目录）
cd microbenchmark

# 全部 4 个族
bash compile_all.sh all

# 单个族
bash compile_all.sh cube
bash compile_all.sh vector
bash compile_all.sh memory
bash compile_all.sh scalar
```

#### 单条用例编译

```bash
cd microbenchmark/scalar
make TESTCASE=add_i32_lat

cd ../cube
make TESTCASE=tmatmul_fp16_64x64x64
```

产物输出到：`output/microbenchmark/<family>/elf/<family>/<case>.elf`
（注意：路径相对于 SuperNPUBench 仓库根，因为 Makefile 的 ROOT 解析到仓库根）

#### 重新生成用例源码（可选）

```bash
cd microbenchmark
python3 gen_cases.py   # 重写 4 个族的 src/ 树 + compile.all
```

### 2.5 生成反汇编文件

`compile.all` 脚本默认只执行 `make`（默认目标 `all`），仅生成 ELF 二进制。
反汇编需要额外步骤。

#### 方式 1：单条用例的 `diss` 目标

```bash
cd benchmark/one-level-arch/test/kernel/matmul
make TESTCASE=matmul TYPE=HIF4_HIF4 VER=MX_NOGATHER M=256 N=2048 K=2048 tM=64 tN=64 tK=64 diss
# 生成: .../<name>.elf.diss
```

`diss` 目标内部执行：`llvm-objdump -dl <elf> > <elf>.diss`

#### 方式 2：批量生成所有 ELF 的反汇编（推荐）

```bash
cd SuperNPUBench

OBJDUMP=$COMPILER_DIR/llvm-objdump

# 为 one-level-arch 和 microbenchmark 的所有 ELF 生成反汇编
find benchmark/one-level-arch/output output/microbenchmark \
  -name "*.elf" -type f ! -name "*.diss" | while read -r elf; do
  "$OBJDUMP" -dl "$elf" > "${elf}.diss"
done

# 统计
echo "ELF:  $(find benchmark/one-level-arch/output output/microbenchmark -name '*.elf' | wc -l | tr -d ' ')"
echo "Diss: $(find benchmark/one-level-arch/output output/microbenchmark -name '*.diss' | wc -l | tr -d ' ')"
```

#### 反汇编文件格式

`.elf.diss` 文件内容示例：

```
/path/to/matmul_MASK_MASK_FP32....elf:  file format elf64-unknown

Disassembly of section .text:

000000000000112c4 <_start>:
       : 112c4: 00 00 00 00  ...
...
```

### 2.6 已知编译失败

以下算子/用例因工具链限制会编译失败，属已知问题，不影响流程：

| 算子/用例 | 失败原因 | 参考链接 |
|-----------|----------|----------|
| `fa` (fa_2d_unroll) | 编译器 Assertion 崩溃：`Reg != 0 && "LinxV5 CallingConv Fail!"`。X=1,Y=1 / X=2,Y=1 触发，避免 Ydim=1 | README Issue #6 |
| `sort` (topk) | `TEXPANDSCALAR` 标识符未在工具链中声明/暴露 | microbenchmark README VECTOR_SKIP |
| micro `tmatmul_acc` | 工具链 `matmul.ac` 后端崩溃 | microbenchmark README |
| micro vector 子集 | 部分 TEPL opcode 未暴露（TPRELU/TADDC/TRELU/TNEG/TNOT/TLOG/TPART*/TCOL* 等） | microbenchmark README |

> `compile_all.sh` 脚本不使用 `set -e`，单条失败不会中断整体编译。

### 2.7 构建系统参考

#### Makefile 目标

```bash
make TESTCASE=<case> all      # 编译生成 ELF（默认目标）
make TESTCASE=<case> diss     # 生成反汇编
make TESTCASE=<case> sim       # 在 QEMU 中运行（PLAT=linx）
make TESTCASE=<case> debug     # 调试模式
make clean                    # 清理当前算子的 .o 文件
make clean_all                # 清理所有输出
```

#### Makefile 参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `TESTCASE` | 测试用例名 | `matmul`, `fa_2d_unroll` |
| `TYPE` | 算子类型 (matmul) | `HIF4_HIF4`, `A16W4`, `MASK` |
| `VER` | 实现版本 | `MX_NOGATHER`, `MX_NOGATHER_REUSEA`, `BASE` |
| `MODE` | 算子模式 | `MASK_FP32`, `BF16x2_NOGATHER` |
| `M` / `N` / `K` | 矩阵维度 | `M=256 N=2048 K=2048` |
| `tM` / `tN` / `tK` | 分块大小 | `tM=64 tN=64 tK=64` |
| `COMPILER_DIR` | 编译器 bin 目录路径 | **必须设置** |
| `PLAT` | 平台 | `linx`（默认）, `cpu` |
| `baremetal` | 裸金属模式 | `on` / `off`（默认） |

#### 编译器关键选项

```
-mlxbc -fenable-matrix -O2
-mllvm -enable-all-vector-as-tilereg=true
-mllvm -linxv5-enable-HL-Inst-Opt=true
-mllvm -linxv5-enable-dim-opt=true
-mllvm -linxv5-enable-ldst-bridge=false
-mllvm -linxv5-enable-continuous-mem-opt=true
-mllvm -linxv5-enable-tile-clock-hand=false
-mllvm -linxv5-enable-simt-clock-hand=true
-mllvm -enable-misched=false
-std=c++20
```

### 2.8 验证编译产物

```bash
cd SuperNPUBench

# 统计 ELF 和反汇编文件数量
echo "one-level-arch  ELF: $(find benchmark/one-level-arch/output -name '*.elf' | wc -l | tr -d ' ')"
echo "one-level-arch  DISS: $(find benchmark/one-level-arch/output -name '*.diss' | wc -l | tr -d ' ')"
echo "microbenchmark  ELF: $(find output/microbenchmark -name '*.elf' | wc -l | tr -d ' ')"
echo "microbenchmark  DISS: $(find output/microbenchmark -name '*.diss' | wc -l | tr -d ' ')"

# 按族统计 microbenchmark
for fam in cube vector memory scalar; do
  echo "  $fam: $(find output/microbenchmark/$fam -name '*.elf' | wc -l | tr -d ' ') ELF"
done

# 验证一个 ELF 文件格式
file benchmark/one-level-arch/output/kernel/matmul/elf/kernel_matmul/matmul_MASK_MASK_FP32_M256_N256_K256_tM32_tN32_tK64.elf
# 预期: ... ELF 64-bit ... elf64-unknown
```

---

## 3. 阶段三：构建 SuperScalarModel 仿真器

**仓库**：`SuperScalarModel/`
**产物**：`bin/gfrun`（功能模型）+ `bin/gfsim`（周期时序模型）
**预计耗时**：约 3-8 分钟（取决于核数和是否 ccache）

### 3.1 仿真器概述

SuperScalarModel 包含两个核心二进制：

| 二进制 | 源码 | 作用 |
|--------|------|------|
| `bin/gfrun` | `emulator/main.cpp` | 功能模型——执行 BlockISA 程序，验证正确性 |
| `bin/gfsim` | `TimingSim/core/main.cpp` | 周期精确时序模型——逐周期推进，输出 PMU 统计 |

架构采用"主核 + 专用核"层次化模型：Block Control Core (BCC) 调度 Cube / Vector /
TMA 等专用引擎，以 Tile Register 为数据交互枢纽。

### 3.2 构建依赖

| 组件 | 要求 | macOS 安装 | Linux 安装 |
|------|------|-----------|------------|
| C++ 编译器 | GCC 8+ 或 Clang 10+ | Xcode (clang) | `apt install gcc g++` |
| CMake | >= 3.10 | `brew install cmake` | `apt install cmake` |
| Python | 3.8+ | `brew install python@3` | `apt install python3` |
| toml (Python 包) | 必须 | `pip3 install toml` | `pip3 install toml` |
| libelf | 必须 | `brew install libelf` | `apt install libelf-dev` |
| rapidjson | 已内置 | — | — |
| googletest | 仅 `--tests` 时需要 | 见 `tests/README.md` | 见 `tests/README.md` |
| ccache | 可选（加速重建） | `brew install ccache` | `apt install ccache` |

### 3.3 构建

```bash
cd SuperScalarModel

# 方式 1：推荐——配置 + 编译（一步到位）
python3 build.py all -j8

# 如果系统 python3 缺少 toml，使用项目自带的 venv：
.venv/bin/python3 build.py all -j8

# 全新重建（清理后重新配置编译）
python3 build.py all --clean -j8

# 方式 2：分步
python3 build.py configure        # 仅配置
python3 build.py build -j8         # 仅编译

# 仅编译单个目标
python3 build.py build --target gfrun -j8
python3 build.py build --target gfsim -j8

# 清理构建目录
python3 build.py clean
```

> **重要**：产物输出到**源码树**中，不是 `build/` 目录：
> - 可执行文件 → `bin/`
> - 库文件 → `lib/`
> - `build/` 仅保存 CMake 配置和中间产物

### 3.4 验证产物

```bash
ls -la bin/gfrun bin/gfsim
# 预期: 两个可执行文件，gfrun ~3.6MB，gfsim ~13MB

# 快速功能检查
bin/gfrun -f tests/prebuilt-elf/microbenchmark/scalar/add_i32_lat.elf 2>&1 | tail -5
# 预期输出末尾:
# Total Block number = 1031
# Total Inst number = 9256
# Suaccelss to Reach the End of Benchmark! R2 = 0
```

### 3.5 build.py 常用选项

| 选项 | 说明 |
|------|------|
| `-j, --parallel N` | 并行编译任务数 |
| `--clean` | 清理构建目录后重新配置 |
| `-t, --build-type TYPE` | CMake 构建类型：`Release`(默认) / `Debug` / `RelWithDebInfo` / `MinSizeRel` |
| `-O, --opt-level LEVEL` | 优化级别：`O0` / `O1` / `O2` / `O3`(默认) / `Os` |
| `-G, --generator GEN` | CMake 生成器，如 `Ninja` |
| `--tests` | 编译单元测试（需 googletest） |
| `--generic-soc` | 启用 generic SOC V3.1.1 路径（仅 x86） |
| `--generic-soc-new` | 启用 generic SOC V6.6.4 路径（仅 x86） |
| `--no-debug` | 禁用调试符号 |
| `--asan` | 启用 AddressSanitizer（建议配合 `-O O0`） |
| `--ubsan` | 启用 UndefinedBehaviorSanitizer |
| `--coverage` | 启用覆盖率插桩 |
| `--warnings-as-errors` | 警告视为错误（CI 默认 ON，本地默认 OFF） |
| `--verbose` | 详细 CMake 输出 |
| `-y, --yes` | 自动确认依赖安装提示 |

### 3.6 产物清单

| 路径 | 类型 | 说明 |
|------|------|------|
| `bin/gfrun` | 可执行文件 | 功能模型 |
| `bin/gfsim` | 可执行文件 | 周期时序模型 |
| `lib/libsoft_core.a` | 静态库 | emulator |
| `lib/libmodel_lib.a` | 静态库 | TimingSim |
| `lib/libisa_blockisa.a` | 静态库 | ISA 定义 |
| `lib/libsoftfloat.a` | 静态库 | 浮点支持 |
| `lib/libconfigs.a` | 静态库 | 配置 |
| `lib/libcommon_impl.a` | 静态库 | 通用工具 |
| `lib/libtools_lib.a` | 静态库 | 工具 |

### 3.7 手动 CMake 构建（替代方式）

```bash
mkdir build && cd build
cmake ..
cmake --build . -j8
# 产物同样输出到 bin/ 和 lib/
```

---

## 4. 阶段四：运行仿真器测试

**前提**：阶段二已生成 ELF 文件，阶段三已构建 `gfrun` / `gfsim`

### 4.1 gfrun — 功能模型（正确性验证）

`gfrun` 通过功能模拟器执行 BlockISA 二进制程序，验证执行正确性。

```bash
cd SuperScalarModel

# 基本运行
bin/gfrun -f /path/to/<program>.elf

# 指定起始/停止 PC
bin/gfrun -f <elf> -X <start_pc_hex> -r <stop_pc_hex>

# 限制最大块数 + 开启 trace
bin/gfrun -f <elf> -c <max_blocks> -t 1
```

#### 成功标志

运行成功时输出末尾包含：

```
Total Block number = <N>
Total Inst number = <M>
Suaccelss to Reach the End of Benchmark! R2 = 0
```

> `R2 = 0` 表示退出码为 0（通过）。注意源码中 "Suaccelss" 是原始拼写。

#### 运行 SuperNPUBench 编译的 ELF

```bash
# microbenchmark — 标量族
bin/gfrun -f ../SuperNPUBench/output/microbenchmark/scalar/elf/scalar/add_i32_lat.elf

# microbenchmark — Cube 族
bin/gfrun -f ../SuperNPUBench/output/microbenchmark/cube/elf/cube/tmatmul_fp16_64x64x64.elf

# microbenchmark — Memory 族
bin/gfrun -f ../SuperNPUBench/output/microbenchmark/memory/elf/memory/tload_fp16_16x16.elf

# benchmark/one-level-arch — matmul
bin/gfrun -f ../SuperNPUBench/benchmark/one-level-arch/output/kernel/matmul/elf/kernel_matmul/matmul_MASK_MASK_FP32_M256_N256_K256_tM32_tN32_tK64.elf
```

### 4.2 gfsim — 周期时序模型

`gfsim` 驱动 TimingSim 逐周期推进，输出 PMU 统计、Stall 统计和可选的可视化数据。

```bash
# 基本运行
bin/gfsim -f /path/to/<program>.elf

# trace 模式（1=普通，5=完整 PipeView）
bin/gfsim -f <input> -t 1
bin/gfsim -f <input> -t 5

# 运行时覆盖配置
bin/gfsim -f <input> -s core.bp_mode=0 core.soc_enable=true

# 纯 tile-op 算子（如 control/hashtable_lookup_simd）需要 single-tier 模式
bin/gfsim -f <elf> -s core.singleTierMode=true
```

> **重要**：纯 tile-op 算子（使用 TEPL 模板指令的算子，如
> `control/hashtable_lookup_simd`）在 gfsim 中**必须**加
> `-s core.singleTierMode=true`，否则引擎不工作，运行会死锁。
> `gfrun` 不需要此标志。

#### gfsim 输出的 PMU 统计

运行结束后输出统计区段，关键指标：

```
Total Cycles.....................................:      41399
Sim Total Cycles.................................:      41399
superScalar Tileop Total Cycles..................:      41399
  |--Cube Tileop Total Cycles....................:      30082
  |--Vector Tileop Total Cycles..................:          0
  |--TMA Tileop Total Cycles.....................:      11955
superScalar Run Tileop Total Cycles..............:      41305
```

| 指标 | 含义 |
|------|------|
| `Total Cycles` / `Sim Total Cycles` | 总仿真周期数 |
| `superScalar Tileop Total Cycles` | Tile 算子执行总周期 |
| `Cube/Vector/TMA Tileop Total Cycles` | 各引擎周期分解 |
| `superScalar Run Tileop Total Cycles` | 运行态周期（排除 idle） |
| `Average Outstanding Block Number` | 平均未完成块数 |
| 各类 `Stall` 统计 | ROB/Rename/BIsq/TileReg 满导致的停顿 |

#### PMU 统计注意事项

- `Cube/Vector/TMA Tileop Total Cycles` 子项统计的是**块数**而非周期数，不等于父项的分解
- `Real Read/Write TileReg Cnt` 语义因 PE 不同而异（Cube=请求数，Vector=操作数，TMA=RFB 请求数），不可跨 PE 比较
- 标签显示 `(512B)` 但硬件最大为 256B（`MAX_TILE_DATA_BYTE`），计算带宽需乘 256

### 4.3 可视化输出

#### PipeView（指令级流水线图）

```bash
bin/gfsim -f <input> -p 1 --pipefile my_trace    # 完整模式（块+组+指令）
bin/gfsim -f <input> -p 2 --pipefile my_trace    # 仅块级（输出更小）
bin/gfsim -f <input> -p 1 --pipe_filter_group    # 仅组级
```

产物：`<pipefile>.out`（Konata v0004 格式）。用 [Konata](https://github.com/shioyadan/Konata/releases) 桌面查看器打开。

#### SwimLane（块级执行甘特图）

```bash
bin/gfsim -f <input> --swimlane 1 --swimfile my_swim
```

产物：`<swimfile>.json`（Perfetto 格式）。在 [ui.perfetto.dev](https://ui.perfetto.dev) 打开（浏览器本地，不上传）。

| | PipeView | SwimLane |
|---|---|---|
| 粒度 | 每指令的流水线阶段 | 每块的执行窗口 |
| 适合 | 微架构调试（停顿、缓存、旁路） | 宏观并行度（PE 利用率、依赖链） |

### 4.4 集成自测（Pass List）

SuperScalarModel 内置集成回归测试 `scripts/ci_self_test.py`，运行 pass list 中列出的全部 ELF：

```bash
cd SuperScalarModel

# PR 级（轻量，每 ELF < 60s）
python3 scripts/ci_self_test.py --tool gfrun
python3 scripts/ci_self_test.py --tool gfsim

# 先编译再测试
python3 scripts/ci_self_test.py --tool gfsim --build

# 按子串过滤单条用例
python3 scripts/ci_self_test.py --tool gfrun --filter tmatmul_fp32

# 指定二进制路径（跳过编译）
python3 scripts/ci_self_test.py --tool gfsim --bin /path/to/gfsim

# 加大超时（秒，默认 600）
python3 scripts/ci_self_test.py --tool gfsim --filter tmatmul --timeout 1800

# Nightly 级（重量级，每 ELF >= 60s）
python3 scripts/ci_self_test.py --tool gfsim --pass-list tests/gfsim-pass-list-nightly.txt

# 两个工具一起
python3 scripts/ci_self_test.py --both --build --opt-level O0 --build-type Debug
```

Pass list 文件：

| 文件 | 级别 | 工具 |
|------|------|------|
| `tests/gfrun-pass-list.txt` | 轻量 | gfrun |
| `tests/gfrun-pass-list-nightly.txt` | 重量 | gfrun |
| `tests/gfsim-pass-list.txt` | 轻量 | gfsim |
| `tests/gfsim-pass-list-nightly.txt` | 重量 | gfsim |

### 4.5 简易测试示例

以下为一次完整的简易运行测试（"只测试简易"）：

```bash
cd SuperScalarModel
BENCH=../SuperNPUBench

echo "===== gfrun 功能测试 ====="

echo "--- micro/scalar/add_i32_lat ---"
bin/gfrun -f $BENCH/output/microbenchmark/scalar/elf/scalar/add_i32_lat.elf 2>&1 | tail -5

echo "--- micro/cube/tmatmul_fp16_64x64x64 ---"
bin/gfrun -f $BENCH/output/microbenchmark/cube/elf/cube/tmatmul_fp16_64x64x64.elf 2>&1 | tail -5

echo "--- micro/memory/tload_fp16_16x16 ---"
bin/gfrun -f $BENCH/output/microbenchmark/memory/elf/memory/tload_fp16_16x16.elf 2>&1 | tail -5

echo "--- one-level/matmul_MASK_FP32 ---"
bin/gfrun -f $BENCH/benchmark/one-level-arch/output/kernel/matmul/elf/kernel_matmul/matmul_MASK_MASK_FP32_M256_N256_K256_tM32_tN32_tK64.elf 2>&1 | tail -5

echo ""
echo "===== gfsim 时序测试 ====="

echo "--- micro/scalar/add_i32_lat ---"
bin/gfsim -f $BENCH/output/microbenchmark/scalar/elf/scalar/add_i32_lat.elf 2>&1 | grep "Total Cycles" | head -2

echo "--- one-level/matmul_MASK_FP32 ---"
bin/gfsim -f $BENCH/benchmark/one-level-arch/output/kernel/matmul/elf/kernel_matmul/matmul_MASK_MASK_FP32_M256_N256_K256_tM32_tN32_tK64.elf 2>&1 | grep -E "Total Cycles|Cube Tileop|TMA Tileop" | head -5
```

---

## 5. 一键全流程脚本

以下脚本从零执行完整流程。保存为 `run_full_flow.sh` 后 `bash run_full_flow.sh` 运行。

```bash
#!/bin/bash
set -e

# ===== 配置 =====
GH="${GITHUB_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
TC="$GH/linx-toolchain-build"
BENCH="$GH/SuperNPUBench"
SIM="$GH/SuperScalarModel"
JOBS="${JOBS:-8}"
export COMPILER_DIR="$TC/output/linx_blockisa_llvm_musl/bin"
OBJDUMP="$COMPILER_DIR/llvm-objdump"

# macOS 检测
if [[ "$(uname)" == "Darwin" ]]; then
  MAKE_CMD="gmake"
else
  MAKE_CMD="make"
fi

echo "=========================================="
echo "  从算子到芯片执行 — 端到端全流程"
echo "=========================================="

# ===== 阶段一：构建工具链 =====
echo ""
echo ">>> [1/4] 构建编译器工具链..."
cd "$TC"
$MAKE_CMD init-src
$MAKE_CMD WITH_TARGET=linx64v5-linux-musl
echo "  工具链版本: $($COMPILER_DIR/clang --version 2>&1 | head -1)"

# ===== 阶段二A：编译 one-level-arch =====
echo ""
echo ">>> [2/4] 编译 SuperNPUBench 算子..."
cd "$BENCH"
echo "  --- one-level-arch ---"
bash benchmark/one-level-arch/compile_all.sh

# ===== 阶段二B：编译 microbenchmark =====
echo "  --- microbenchmark ---"
cd "$BENCH/microbenchmark"
bash compile_all.sh all

# ===== 阶段二C：生成反汇编 =====
echo "  --- 生成反汇编文件 ---"
cd "$BENCH"
find benchmark/one-level-arch/output output/microbenchmark \
  -name "*.elf" -type f ! -name "*.diss" | while read -r elf; do
  "$OBJDUMP" -dl "$elf" > "${elf}.diss"
done
ELF_COUNT=$(find benchmark/one-level-arch/output output/microbenchmark -name "*.elf" | wc -l | tr -d ' ')
DISS_COUNT=$(find benchmark/one-level-arch/output output/microbenchmark -name "*.diss" | wc -l | tr -d ' ')
echo "  ELF: $ELF_COUNT | Diss: $DISS_COUNT"

# ===== 阶段三：构建仿真器 =====
echo ""
echo ">>> [3/4] 构建 SuperScalarModel 仿真器..."
cd "$SIM"
if [ -f .venv/bin/python3 ]; then
  .venv/bin/python3 build.py all -j "$JOBS"
else
  python3 build.py all -j "$JOBS"
fi
echo "  gfrun: $(ls -la bin/gfrun | awk '{print $5}') bytes"
echo "  gfsim: $(ls -la bin/gfsim | awk '{print $5}') bytes"

# ===== 阶段四：运行测试 =====
echo ""
echo ">>> [4/4] 运行仿真器测试（简易）..."
BENCH_PATH="$BENCH"

echo "  --- gfrun 功能测试 ---"
for elf in \
  "$BENCH_PATH/output/microbenchmark/scalar/elf/scalar/add_i32_lat.elf" \
  "$BENCH_PATH/output/microbenchmark/cube/elf/cube/tmatmul_fp16_64x64x64.elf" \
  "$BENCH_PATH/output/microbenchmark/memory/elf/memory/tload_fp16_16x16.elf" \
  "$BENCH_PATH/benchmark/one-level-arch/output/kernel/matmul/elf/kernel_matmul/matmul_MASK_MASK_FP32_M256_N256_K256_tM32_tN32_tK64.elf"
do
  name=$(basename "$elf")
  result=$(bin/gfrun -f "$elf" 2>&1 | grep -o "R2 = [0-9]*" | tail -1)
  echo "    $name → $result"
done

echo "  --- gfsim 时序测试 ---"
for elf in \
  "$BENCH_PATH/output/microbenchmark/scalar/elf/scalar/add_i32_lat.elf" \
  "$BENCH_PATH/benchmark/one-level-arch/output/kernel/matmul/elf/kernel_matmul/matmul_MASK_MASK_FP32_M256_N256_K256_tM32_tN32_tK64.elf"
do
  name=$(basename "$elf")
  cycles=$(bin/gfsim -f "$elf" 2>&1 | grep "Total Cycles\.\." | head -1 | grep -oE '[0-9]+$')
  echo "    $name → $cycles cycles"
done

echo ""
echo "=========================================="
echo "  全流程完成！"
echo "=========================================="
```

---

## 6. 验证基线（参考结果）

以下为一次完整流程的实际运行结果，可用于验证你的环境是否正确。

### 6.1 工具链

```
clang version 15.0.4 (linx64v5-musl-local cd22ce5f4ecd54939565d6906749f1ca3808622f)
Target: linx64v5-unknown-linux-musl
```

### 6.2 SuperNPUBench 编译产物

| 类别 | ELF 数 | .diss 数 | 说明 |
|------|--------|----------|------|
| one-level-arch | 52 | 52 | 11/13 算子成功（fa、sort 失败，已知问题） |
| microbenchmark | 293 | 293 | cube(9) + vector(135) + memory(25) + scalar(124) |
| **合计** | **345** | **345** | |

### 6.3 gfrun 功能测试结果

| ELF | Block 数 | Inst 数 | 退出码 |
|-----|----------|---------|--------|
| micro/scalar/add_i32_lat | 1031 | 9256 | R2=0 ✓ |
| micro/cube/tmatmul_fp16_64x64x64 | 8213 | 123226 | R2=0 ✓ |
| micro/memory/tload_fp16_16x16 | 272 | 3150 | R2=0 ✓ |
| one-level/matmul_MASK_FP32 | 1309 | 6165 | R2=0 ✓ |

### 6.4 gfsim 时序测试结果

| ELF | Total Cycles | Cube Cycles | Vector Cycles | TMA Cycles |
|-----|-------------|-------------|---------------|------------|
| micro/scalar/add_i32_lat | 3227 | 0 | 0 | 0 |
| one-level/matmul_MASK_FP32 | 41399 | 30082 | 0 | 11955 |

---

## 7. 故障排查

### 7.1 工具链构建

| 问题 | 原因 | 解决 |
|------|------|------|
| `GNU Make >= 4.0 is required` | macOS 自带 make 是 3.81 | `brew install make`，用 `gmake` |
| `tar: --format=gnu: Not found` (package 阶段) | macOS tar 不支持 gnu 格式 | `brew install gnu-tar`，加入 PATH |
| sancov.cpp 编译错误 `copy-initialization` | Apple clang 对 initializer-list 更严格 | 改 `createOrDie({{...}}` 为 `createOrDie(std::vector<std::string>{{...}}` |
| 构建中断后重新运行不恢复 | stamp 文件可能不完整 | 检查 `stamps/` 目录，删除不完整的 stamp 后重新 `make` |
| `COMPILER_DIR is not set` | 环境变量未设置 | `export COMPILER_DIR=.../linx_blockisa_llvm_musl/bin` |

### 7.2 SuperNPUBench 编译

| 问题 | 原因 | 解决 |
|------|------|------|
| `fa` 编译 Assertion 崩溃 | 编译器 Issue #6，Ydim=1 触发 | 避免 `Y=1` 的配置 |
| `sort` 编译 `TEXPANDSCALAR` 未声明 | 工具链未暴露该 opcode | 跳过，等工具链更新 |
| micro `tmatmul_acc` 编译失败 | 工具链 matmul.ac 后端 crash | 跳过 |
| micro compile_all.sh `cd: No such file` | 从错误目录运行 | 必须从 `microbenchmark/` 目录运行 |
| micro 重新编译无变化 | Makefile 无 clean 依赖，跳过已编译 | 先 `make clean_all` 或 `make TESTCASE=xxx clean` |
| 编译产物路径找不到 | Makefile 的 ROOT 解析到仓库根 | one-level 在 `benchmark/one-level-arch/output/`，micro 在 `output/microbenchmark/` |

### 7.3 SuperScalarModel 构建

| 问题 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: No module named 'toml'` | Python 缺 toml 包 | `pip3 install toml` 或用 `.venv/bin/python3` |
| `Could NOT find libelf` | libelf 未安装 | macOS: `brew install libelf`；Linux: `apt install libelf-dev` |
| `Could NOT find GTest` | googletest 未安装 | 仅 `--tests` 时需要，见 `tests/README.md` 安装到 `$GTEST_ROOT` |
| 链接警告 `ignoring duplicate libraries` | 重复链接同一库 | 无害，不影响功能 |
| 构建产物在 `build/` 而非 `bin/` | 用了手动 cmake | build.py 把产物输出到 `bin/` 和 `lib/`；手动 cmake 默认输出到 build 目录 |

### 7.4 仿真器运行

| 问题 | 原因 | 解决 |
|------|------|------|
| `ELF binary or Json file open fail` | 文件路径错误或不存在 | 检查 `-f` 后的路径 |
| gfsim 运行死锁/无输出 | 纯 tile-op 算子未开 single-tier | 加 `-s core.singleTierMode=true` |
| `R2 = 非0` | 程序执行结果不正确 | 检查算子实现或编译选项 |
| gfsim 运行很慢 | 周期精确仿真开销大 | 使用 gfrun 做功能验证；gfsim 用于时序分析 |
| macOS 无 `timeout` 命令 | 系统不带 | `brew install coreutils` 安装 `gtimeout`，或直接运行（用 shell 超时控制） |

---

## 8. 附录：目录结构与文件清单

### 8.1 linx-toolchain-build 目录结构

```
linx-toolchain-build/
├── Makefile                     # 顶层构建入口
├── README.md
├── src/                         # 组件源码（make init-src 生成）
│   ├── llvm-project/            # LLVM 15 (分支 dev-llvm15_56)
│   ├── musl/                    # musl libc (分支 linx)
│   ├── jemalloc/                # jemalloc (分支 linx)
│   ├── linux-linxisa/           # Linux 内核 (分支 main)
│   └── Linx-TileOP-API/         # Tile 算子 API 头文件 (分支 linx)
├── build/                       # 构建中间产物
│   ├── build-llvm-musl/
│   ├── build-musl/
│   ├── build-compiler-rt-musl/
│   ├── build-libcxx-musl/
│   ├── build-libcxxabi-musl/
│   ├── build-libunwind-musl/
│   ├── build-jemalloc/
│   ├── build-kernel-header/
│   └── build-tileopapi/
├── stamps/                      # 构建步骤 stamp 文件
├── output/                      # 最终产物
│   └── linx_blockisa_llvm_musl/
│       ├── bin/                 # clang, clang++, ld.lld, llvm-objdump, llvm-ar, ...
│       ├── lib/                 # clang runtime, libc++, ...
│       ├── include/
│       ├── libexec/
│       ├── share/
│       └── sysroot/             # musl + kernel headers + runtime libs
│           └── usr/
└── scripts/
```

### 8.2 SuperNPUBench 产物目录结构

```
SuperNPUBench/
├── benchmark/one-level-arch/output/
│   └── kernel/
│       ├── matmul/elf/kernel_matmul/
│       │   ├── matmul_HIF4_HIF4_MX_NOGATHER_B1_M256_N2048_K2048_tM64_tN64_tK64.elf
│       │   ├── matmul_HIF4_HIF4_MX_NOGATHER_B1_M256_N2048_K2048_tM64_tN64_tK64.elf.diss
│       │   ├── matmul_MASK_MASK_FP32_M256_N256_K256_tM32_tN32_tK64.elf
│       │   ├── matmul_MASK_MASK_FP32_M256_N256_K256_tM32_tN32_tK64.elf.diss
│       │   └── ...
│       ├── broadcast/elf/kernel_broadcast/
│       ├── concat/elf/kernel_concat/
│       ├── gather/elf/kernel_gather/
│       ├── transpose/elf/kernel_transpose/
│       ├── element_wise/gelu/elf/kernel_element_wise_gelu/
│       ├── reduction/reducemax_col/elf/kernel_reduction_reducemax_col/
│       ├── reduction/reducemax_row/elf/kernel_reduction_reducemax_row/
│       ├── reduction/reducesum_col/elf/kernel_reduction_reducesum_col/
│       ├── reduction/reducesum_row/elf/kernel_reduction_reducesum_row/
│       └── control/elf/kernel_control/
└── output/microbenchmark/
    ├── cube/elf/cube/
    │   ├── tmatmul_fp16_64x64x64.elf
    │   ├── tmatmul_fp16_64x64x64.elf.diss
    │   └── ...
    ├── vector/elf/vector/
    │   ├── tadd_fp16_16x16.elf
    │   ├── tadd_fp16_16x16.elf.diss
    │   └── ...
    ├── memory/elf/memory/
    │   ├── tload_fp16_16x16.elf
    │   ├── tload_fp16_16x16.elf.diss
    │   └── ...
    └── scalar/elf/scalar/
        ├── add_i32_lat.elf
        ├── add_i32_lat.elf.diss
        └── ...
```

### 8.3 SuperScalarModel 目录结构

```
SuperScalarModel/
├── CMakeLists.txt              # 根 CMake 入口
├── build.py                    # 推荐构建封装
├── opencode.json               # opencode 配置
├── .venv/                      # Python 虚拟环境（含 toml）
├── bin/                        # 可执行产物（gitignored）
│   ├── gfrun                   # 功能模型
│   └── gfsim                   # 周期时序模型
├── lib/                        # 静态库产物（gitignored）
│   ├── libsoft_core.a
│   ├── libmodel_lib.a
│   ├── libisa_blockisa.a
│   ├── libsoftfloat.a
│   ├── libconfigs.a
│   ├── libcommon_impl.a
│   └── libtools_lib.a
├── build/                      # CMake 配置 + 中间产物（gitignored）
├── emulator/                   # 功能模型源码 (gfrun)
├── TimingSim/                  # 周期时序模型源码 (gfsim)
│   ├── core/                   # 顶层核心
│   ├── frontend/               # 前端（IFU/BCtrl）
│   ├── scalar_pe/              # 标量 PE
│   ├── pe/                     # PE 基类/Cube/Vector
│   ├── group/                  # Group 指令协调层
│   ├── memory/                 # LSU/L1/L2/SOC
│   ├── infra/                   # SimSys/队列/延迟
│   ├── debug/DFX/              # PipeView/验证
│   └── trace/                  # 指令 trace
├── isa/                        # BlockISA 指令定义/编解码
├── refmodel/                   # 参考模型（差分测试）
├── softfloat/                  # 浮点支持
├── common/                     # 通用工具
├── configs/                    # TOML 配置文件
├── tests/                      # 测试 + prebuilt-elf
│   ├── prebuilt-elf/           # 预编译 ELF（CI 用）
│   ├── gfrun-pass-list.txt
│   ├── gfsim-pass-list.txt
│   ├── gfrun-pass-list-nightly.txt
│   └── gfsim-pass-list-nightly.txt
├── scripts/                    # 脚本（ci_self_test.py, trace, 覆盖率等）
├── modelSpec/                  # 模型架构设计文档
├── archSpec/                   # 架构级设计文档
├── third_party/               # 第三方依赖（rapidjson）
└── tools/                      # 工具
```

### 8.4 关键路径速查

```
# 工具链
linx-toolchain-build/output/linx_blockisa_llvm_musl/bin/clang

# one-level-arch ELF
SuperNPUBench/benchmark/one-level-arch/output/kernel/<op>/elf/<category>/<name>.elf

# microbenchmark ELF
SuperNPUBench/output/microbenchmark/<family>/elf/<family>/<case>.elf

# 反汇编（与 ELF 同目录同名，加 .diss 后缀）
<name>.elf.diss

# 仿真器
SuperScalarModel/bin/gfrun
SuperScalarModel/bin/gfsim

# 集成测试
SuperScalarModel/scripts/ci_self_test.py
SuperScalarModel/tests/gfrun-pass-list.txt
SuperScalarModel/tests/gfsim-pass-list.txt
```

---

## 相关链接

- [Linx ISA 官网](https://linxisa.github.io/linx-isa/)
- [PTO ISA 文档](https://pto-isa.github.io/docs/isa/tile/)
- [Konata 查看器（PipeView）](https://github.com/shioyadan/Konata/releases)
- [Perfetto 查看器（SwimLane）](https://ui.perfetto.dev)
