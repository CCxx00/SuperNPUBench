# PTO C++ Programming Guide

> **Version**: 2026.07 | **ISA**: PTO-ISA BlockISA v0.55 | **Toolchain**: linx_blockisa_llvm_musl (clang-15)

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Programming Model](#2-programming-model)
- [3. C++ Programming Interface](#3-c-programming-interface)
- [4. Writing Kernels](#4-writing-kernels)
- [5. Compilation & Toolchain](#5-compilation--toolchain)
- [6. Best Practices & Optimization](#6-best-practices--optimization)
- [7. Appendix](#7-appendix)

---

## 1. Introduction

### 1.1 What Is PTO-ISA?

PTO-ISA (Parallel Tile Operator ISA) is a block-structured Instruction Set
Architecture designed for tile-programming on NPU hardware. It organizes
computation into **block instructions** that operate on **tile registers** —
fixed-size 2D data blocks resident in on-chip register files — rather than
individual scalars.

The programming model is exposed to C++ via **Linx-TileOP-API**, a header-only
template library that maps high-level tile operations (`TADD`, `TMATMUL`,
`TLOAD`, etc.) to inline-assembly block instructions (`BSTART.TEPL`,
`BSTART.TLSU`, `BSTART.CUBE`).

### 1.2 Target Architecture

The **DavinciOO v4 core** consists of 4 Processing Elements (PEs):

| Component | Per-PE | Per-Core (4 PE) |
|-----------|--------|-----------------|
| OoO front-end | independent | — |
| CELL register file | 2048 × 128 B = 256 KB | 1 MB |
| TileReg namespace | 4 queues × 16 entries = 64 | 256 |
| TLSU (memory unit) | private | — |
| CUBE (matrix unit) | private | — |
| TEPL (elementwise unit) | private | — |

Each PE executes tile/block intrinsics as **thread-local** instructions — no
implicit cross-PE register access. Global memory is reachable only through TLSU
blocks (`TLOAD`, `TSTORE`, `MGATHER`, `MSCATTER`).

### 1.3 Document Scope

This guide covers:
- The C++ tile-programming API (`pto_tileop.hpp`)
- Kernel writing patterns (from SuperNPUBench)
- Compilation with the Linx toolchain

For ISA-level encoding and hardware microarchitecture, see the DavinciOO ISA
intrinsic documentation.

---

## 2. Programming Model

### 2.1 Tile-Centric Execution

Unlike scalar ISAs where each instruction processes one element, PTO-ISA
**block instructions** process entire tiles in one operation:

```cpp
// One TADD call adds two 16×16 tiles (256 elements) in a single block instruction
Tile<Vec, half, 16, 16> a, b, c;
TADD(c, a, b);   // c[i][j] = a[i][j] + b[i][j]  for all 256 elements
```

### 2.2 Memory Hierarchy

```
Global Memory (DRAM)
    ↕  TLOAD / TSTORE / MGATHER / MSCATTER  (TLSU blocks)
Tile Register Files (on-chip, per-PE)
    ├── Location::Vec   — general-purpose elementwise tiles
    ├── Location::Left  — GEMM A-operand tiles (boxed, 512B fractal)
    ├── Location::Right — GEMM B-operand tiles (boxed, 512B fractal)
    └── Location::Acc   — accumulator tiles (boxed, 1024B fractal, FP32)
```

There is **no pointer** to tile registers. Data must be explicitly moved between
global memory and tile registers via `TCOPYIN`/`TCOPYOUT` (or `TLOAD`/`TSTORE`).

### 2.3 Tile Register Namespace

Each PE has 4 independent register queues, each with 16 entries:

| Queue | Name | Typical Use |
|-------|------|-------------|
| `T` | `T#1..T#16` | General tile result / temporary |
| `U` | `U#1..U#16` | Second general stream (separate lifetime) |
| `M` | `M#1..M#16` | Extra stream (mask / index / data movement) |
| `N` | `N#1..N#16` | Extra stream (isolated lifetime) |

Queues use **relative indexing**: `#1` = newest live value, `#2` = one older.
When a new tile is produced, it appends to the queue tail. When a source is
consumed without `.reuse`, it may be released.

In C++, the compiler manages register allocation automatically — the programmer
declares C++ `Tile` variables and the lowering pass maps them to TReg queues.

### 2.4 Tile Size Classes

| `imm4` | Tile Bytes | CELLs | Max Live Tiles (capacity-limited) |
|--------|-----------|-------|-----------------------------------|
| 3 | 128 B | 1 | 2048 |
| 4 | 256 B | 2 | 1024 |
| 5 | 512 B | 4 | 512 |
| 6 | 1 KB | 8 | 256 |
| 7 | 2 KB | 16 | 128 |
| 8 | 4 KB | 32 | 64 |
| 9 | 8 KB | 64 | 32 |

Active profile restricts tile allocation to **128 B – 8 KB** (`imm4 = 3..9`).
Total live tile payload per PE must not exceed **256 KB** (2048 CELLs).

### 2.5 Block Instruction Structure

Each tile operation lowers to a **block** described by a header chain:

```
BSTART.<family> <opcode>, <dtype>    ← select block class + opcode + main dtype
B.DATR  <dtype_ext>, <pad>, ...      ← data attributes (optional)
B.DIM   <reg>, <imm>, ->LBx          ← dimensions / loop bounds
B.IOT   <src1>, <src2>, last, -><dst><size>  ← tile operand binding
BSTOP                               ← block end
```

Families:
- **TEPL** — elementwise, tile-scalar, reduce, expand, compare, select
- **TLSU** — TLOAD, TSTORE, MGATHER, MSCATTER (memory ↔ tile)
- **CUBE** — TMATMUL, TGEMV, ACCCVT (matrix compute)

In C++, these are emitted automatically by the template intrinsics — the
programmer never writes raw assembly.

### 2.6 Data Types

| Category | Types |
|----------|-------|
| Float | `FP64`, `FP32`, `FP16`, `BF16`, `HiF8`, `e4m3`, `e5m2`, `HiF4x2` |
| Signed int | `S64`, `S32`, `S16`, `S8`, `S4x2` |
| Unsigned int | `U64`, `U32`, `U16`, `U8`, `U4x2` |

In C++, these map to: `float`, `__half`, `__bf16`, `int8_t`, `int16_t`,
`int32_t`, `uint8_t`, `uint16_t`, `uint32_t`, `__fp8_e4m3`, etc.

---

## 3. C++ Programming Interface

### 3.1 Include Strategy

The single entry point header:

```cpp
#include <common/pto_tileop.hpp>
using namespace pto;
```

This transitively includes:
1. `pto_tile.hpp` — Tile / GlobalTensor type system + concepts
2. `tileop_api.hpp` — public API wrappers (MATMUL, TADD, TLOAD, ...)
3. `global_iterator.hpp` — DRAM tile-stepping iterator
4. `tile_tensor_impl.hpp` — out-of-line Tile constructors
5. `debug_utils.hpp` — debug helpers

Backend is selected by defining exactly one of: `__linx`, `__ARM_FEATURE_SME`,
or `__cpu_sim__` at compile time.

### 3.2 Tile Types

#### Tile

The on-chip tile type, statically shaped:

```cpp
template <Location Loc, typename DType, int Rows, int Cols,
          BLayout BFractal = BLayout::RowMajor,
          int ValidRow = Rows, int ValidCol = Cols,
          SLayout SFractal = SLayout::NoneBox,
          int SFractalSize = 512,
          PadValue PadVal = PadValue::Null>
struct Tile;
```

| Parameter | Meaning |
|-----------|---------|
| `Loc` | Pipeline stage: `Vec`, `Left`, `Right`, `Acc`, `Mat`, `Bias`, `Scaling` |
| `DType` | Element type (`half`, `float`, `int32_t`, ...) |
| `Rows`, `Cols` | Compile-time tile footprint |
| `ValidRow`, `ValidCol` | Active (unpadded) extent; `-1` = runtime dynamic |
| `BFractal` | Block-level layout: `RowMajor` or `ColMajor` |
| `SFractal` | Inner fractal layout: `NoneBox`, `RowMajor`, `ColMajor` |
| `SFractalSize` | `512` for input tiles, `1024` for accumulator tiles |

#### Convenience Aliases

```cpp
// GEMM A-operand: ColMajor outer, RowMajor fractal, 512B block
template <typename E, int R, int C, int VR = R, int VC = C>
using TileLeft  = Tile<Location::Left,  E, R, C, BLayout::ColMajor, VR, VC, SLayout::RowMajor, 512>;

// GEMM B-operand: RowMajor outer, ColMajor fractal, 512B block
template <typename E, int R, int C, int VR = R, int VC = C>
using TileRight = Tile<Location::Right, E, R, C, BLayout::RowMajor, VR, VC, SLayout::ColMajor, 512>;

// GEMM accumulator: ColMajor outer, RowMajor fractal, 1024B block (FP32)
template <typename E, int R, int C, int VR = R, int VC = C>
using TileAcc   = Tile<Location::Acc, E, R, C, BLayout::ColMajor, VR, VC, SLayout::RowMajor, 1024>;
```

#### General-Purpose Tile (elementwise / reduction / load-store)

```cpp
Tile<Location::Vec, half, 16, 16, BLayout::RowMajor> a;
Tile<Location::Vec, float, 16, 16, BLayout::ColMajor> b;
```

#### Constructors

```cpp
TileLeft<half, 128, 128> a;                    // default (uninitialized)
TileLeft<half, 128, 128> a(0.0_h);              // fill with scalar
TileLeft<half, 128, 128, -1, -1> a(96, 64);    // dynamic valid extents
TileLeft<half, 128, 128, -1, -1> a(0.0_h, 96, 64);  // fill + dynamic extents
TileAcc<float, 128, 128, -1, -1> c(0.0f, 96, 64);   // accumulator fill + dynamic
```

### 3.3 Global Memory Types

#### GlobalTensor

```cpp
template <typename DType, typename Shape, typename Stride,
           Layout Layout_ = Layout::ND>
struct GlobalTensor;
```

#### global_tensor (convenience wrapper)

```cpp
using GM = global_tensor<half, RowMajor<256, 256>>;
GM gA(dram_ptr);                    // both dims static
GM gA(dram_ptr, dynamic_cols);      // one dim dynamic
GM gA(dram_ptr, dynamic_rows, dynamic_cols);  // both dynamic
```

#### global_iterator (tile-stepping)

```cpp
using GM = global_tensor<half, RowMajor<256, 256>>;
GM gA(dram_ptr);
global_iterator<GM, TileLeft<half, 128, 128>> it(gA.data());

for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
        TileLeft<half, 128, 128> a;
        auto view = it(i, j);       // returns a GlobalTensor view of one 128×128 tile
        TCOPYIN(a, view);           // load that tile
    }
```

### 3.4 Tile Operations — C++ API Reference

#### Memory Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `TCOPYIN` | `(tile& dst, gm& src)` | Load tile from global memory |
| `TCOPYOUT` | `(gm& dst, tile& src)` | Store tile to global memory |
| `TCOPY` | `(tile& dst, tile& src)` | Tile-to-tile copy |
| `MGATHER` | `(tile& dst, gm& src, tile& offsets)` | Gather by per-element byte offsets |
| `MSCATTER` | `(gm& dst, tile& src, tile& offsets)` | Scatter by per-element byte offsets |

> `TLOAD`/`TSTORE` are aliases for `TCOPYIN`/`TCOPYOUT` in the PTO programming model.

#### Matrix Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `MATMUL` | `(C& dst, A& src0, B& src1)` | `dst = src0 × src1` |
| `MATMACC` | `(C& dst, A& src0, B& src1)` | `dst += src0 × src1` (fused MAC) |
| `MATMULMX` | `(C&, A&, AX&, B&, BX&)` | MX mixed-precision (A+B scaling) |
| `MATMULMXB` | `(C&, A&, B&, BX&)` | MX mixed-precision (B-only scaling) |
| `ACCCVT` | `(tile& dstVec, tile& srcAcc)` | Accumulator → Vec convert |

`MATMUL` overwrites the accumulator; `MATMACC` accumulates into it. A and B
must be `TileLeft`/`TileRight` (boxed fractal layout); C must be `TileAcc`.

#### Elementwise Binary

| Function | Description |
|----------|-------------|
| `TADD(dst, a, b)` | `dst = a + b` |
| `TSUB(dst, a, b)` | `dst = a - b` |
| `TMUL(dst, a, b)` | `dst = a * b` |
| `TDIV(dst, a, b)` | `dst = a / b` |
| `TMAX(dst, a, b)` | `dst = max(a, b)` |
| `TMIN(dst, a, b)` | `dst = min(a, b)` |
| `TAND(dst, a, b)` | Bitwise AND |
| `TOR(dst, a, b)` | Bitwise OR |
| `TXOR(dst, a, b)` | Bitwise XOR |
| `TCMP(dst, a, b)` | Compare (produces mask) |

#### Elementwise Unary

| Function | Description |
|----------|-------------|
| `TAbs(dst, src)` | Absolute value |
| `TExp(dst, src)` | Exponential (e^x) |
| `TLog(dst, src)` | Natural logarithm |
| `TSqrt(dst, src)` | Square root |
| `TRSqrt(dst, src)` | Reciprocal square root (1/√x) |
| `TRecip(dst, src)` | Reciprocal (1/x) |
| `TNeg(dst, src)` | Negation |
| `TNot(dst, src)` | Bitwise NOT |
| `TCvt(dst, src)` | Type/location conversion |
| `TCast(dst, src)` | Bit-cast reshape (same numel) |

#### Tile-Scalar Operations

| Function | Description |
|----------|-------------|
| `TADDs(dst, src, scalar)` | `dst = src + scalar` |
| `TSUBs(dst, src, scalar)` | `dst = src - scalar` |
| `TMULs(dst, src, scalar)` | `dst = src * scalar` |
| `TDIVs(dst, src, scalar)` | `dst = src / scalar` |
| `TMAXs(dst, src, scalar)` | `dst = max(src, scalar)` |
| `TMINs(dst, src, scalar)` | `dst = min(src, scalar)` |

#### Broadcast / Fill

| Function | Description |
|----------|-------------|
| `TEXPANDS(tile, scalar)` | Fill entire tile with scalar value |
| `TCI(tile, scalar)` | Constant-inject ramp `[base, base+1, ...]` |
| `TROWEXPANDMUL(dst, mat, vec)` | Row-broadcast multiply (each row × corresponding scalar) |
| `TCOLEXPANDMUL(dst, mat, vec)` | Column-broadcast multiply |
| `TROWEXPANDADD`, `TCOLEXPANDADD`, ... | Broadcast add/sub/div/max/min variants |

#### Reductions

| Function | Description |
|----------|-------------|
| `TROWSUM(dst, src)` | Row-wise sum → 1-element-per-row vector |
| `TCOLSUM(dst, src)` | Column-wise sum → 1-element-per-col vector |
| `TROWMAX(dst, src)` | Row-wise max |
| `TCOLMAX(dst, src)` | Column-wise max |
| `TROWMIN(dst, src)` | Row-wise min |
| `TCOLMIN(dst, src)` | Column-wise min |

#### Shift / Special

| Function | Description |
|----------|-------------|
| `TSHL(dst, src, shift)` | Left shift |
| `TSHR(dst, src, shift)` | Right shift |
| `TTRANS(dst, src)` | Hardware 2D transpose |
| `TRESHAPE(dst, src)` | Reshape (same numel) |
| `TEXTRACT(dst, src)` | Extract sub-tile |
| `TINSERT(dst, src)` | Insert sub-tile |
| `TFILLPAD(dst, src)` | Fill padding region |
| `TCONCAT(dst, src0, src1)` | Concatenate two tiles |
| `TSELECT(dst, mask, a, b)` | Select by mask |

### 3.5 Concepts and Constraints

```cpp
template <typename T> concept is_tile_data_v;    // matches Tile<...>
template <typename T> concept is_global_data_v;  // matches GlobalTensor / global_tensor
template <typename T> concept is_boxed_data_v;   // boxed (fractal) tile
```

These are used as C++20 template constraints on all API functions. The compiler
will produce a clear error if you pass the wrong type.

---

## 4. Writing Kernels

### 4.1 Kernel Structure

A PTO kernel is a C++ template function. Shapes are compile-time template
parameters enabling full unrolling:

```cpp
template <typename dtype, int gM, int gN, int gK,
          int tM, int tN, int tK>
void my_kernel(float *out_ptr, dtype *a_ptr, dtype *b_ptr) {
    using namespace pto;
    // ... tile declarations and operations ...
}
```

### 4.2 Example: Matrix Multiply

```cpp
#include <common/pto_tileop.hpp>
using namespace pto;

template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK>
void matmul(float *c_ptr, dtype *a_ptr, dtype *b_ptr) {
    // Global memory descriptors
    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gK, gN>>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;

    // Tile types
    using TileA   = TileLeft<dtype, tM, tK>;
    using TileB   = TileRight<dtype, tK, tN>;
    using TileAcc = TileAcc<float, tM, tN>;

    gmA gA(a_ptr);
    gmB gB(b_ptr);
    gmC gC(c_ptr);

    global_iterator<gmA, TileA> itA(gA.data());
    global_iterator<gmB, TileB> itB(gB.data());
    global_iterator<gmC, TileAcc> itC(gC.data());

    constexpr int Mb = gM / tM, Nb = gN / tN, Kb = gK / tK;

    for (int i = 0; i < Mb; ++i) {
        for (int j = 0; j < Nb; ++j) {
            TileAcc acc(0.0f);                    // zero-initialize accumulator
            for (int k = 0; k < Kb; ++k) {
                TileA tA;
                TileB tB;
                TCOPYIN(tA, itA(i, k));           // load A tile
                TCOPYIN(tB, itB(k, j));           // load B tile
                if (k == 0)
                    MATMUL(acc, tA, tB);           // first: dst = A × B
                else
                    MATMACC(acc, tA, tB);          // subsequent: dst += A × B
            }
            // Convert Acc → Vec for store
            Tile<Vec, float, tM, tN, BLayout::RowMajor> out;
            ACCCVT(out, acc);
            TCOPYOUT(itC(i, j), out);
        }
    }
}
```

### 4.3 Example: Elementwise + Reduction

```cpp
// Column-wise reduce-sum of a matrix
template <typename dtype, int gM, int gN, int tM, int tN>
void reducesum(dtype *out_ptr, dtype *in_ptr) {
    using namespace pto;

    using gmIn  = global_tensor<dtype, RowMajor<gM, gN>>;
    using gmOut = global_tensor<dtype, RowMajor<1, gN>>;

    using TileData = Tile<Vec, dtype, tM, tN, BLayout::RowMajor>;
    using TileSum  = Tile<Vec, dtype, 1, tN, BLayout::RowMajor>;

    gmIn gIn(in_ptr);
    gmOut gOut(out_ptr);

    global_iterator<gmIn, TileData> itIn(gIn.data());
    global_iterator<gmOut, TileSum> itOut(gOut.data());

    constexpr int Mb = gM / tM, Nb = gN / tN;

    for (int j = 0; j < Nb; ++j) {
        TileSum sum;
        TEXPANDS(sum, static_cast<dtype>(0));     // fill with zero

        for (int i = 0; i < Mb; ++i) {
            TileData data;
            TCOPYIN(data, itIn(i, j));

            TileSum partial;
            TCOLSUM(partial, data);                // column-wise sum
            TADD(sum, sum, partial);               // accumulate
        }
        TCOPYOUT(itOut(0, j), sum);
    }
}
```

### 4.4 Example: Quantization (absmax + scale)

```cpp
// Per-token FP8 quantization: compute scale = max|x| / max_val, then x / scale
template <int M, int Npc, int TileM = 16>
void per_token_cast(__bf16 *x, int hidden, float *out_sf, __bf16 *out,
                    float max_val, float clamp_min) {
    using namespace pto;
    using tile_x    = Tile<Vec, __bf16, TileM, Npc, BLayout::RowMajor>;
    using tile_f    = Tile<Vec, float,   TileM, Npc, BLayout::RowMajor>;
    using tile_amax = Tile<Vec, float, TileM, 8, BLayout::RowMajor, TileM, 1>;

    tile_x xq;
    tile_f xf, neg, outf;
    tile_amax pos, negm, amax, sf, sfinv, inv;

    // 1) Load and convert bf16 → fp32
    TCOPYIN(xq, /* global view */);
    TCVT(xf, xq);

    // 2) absmax = max(max(x), max(-x))   (TABS/TNEG unavailable → emulate)
    TMULs(neg, xf, -1.0f);
    TROWMAX(pos, xf);
    TROWMAX(negm, neg);
    TMAX(amax, pos, negm);

    // 3) Clamp and compute scale
    TMAXs(amax, amax, clamp_min);
    TMULs(sf, amax, 1.0f / max_val);
    TRECIP(inv, amax);
    TMULs(sfinv, inv, max_val);

    // 4) Scale and convert back
    TROWEXPANDMUL(outf, xf, sfinv);
    TCVT(xq, outf);
    TSTORE(/* scale global */, sf);
    TSTORE(/* data global */, xq);
}
```

### 4.5 Tail / Boundary Handling

For dimensions not evenly divisible by tile size, use the `ValidRow`/`ValidCol`
template parameters:

```cpp
constexpr int rmd_M = gM % tM;  // remainder rows
constexpr int rmd_N = gN % tN;  // remainder cols

// Full tile (no remainder)
using TileData     = Tile<Vec, dtype, tM, tN, BLayout::RowMajor>;

// Row-tail tile (fewer valid rows)
using TileDataRow  = Tile<Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, tN>;

// Col-tail tile (fewer valid cols)
using TileDataCol  = Tile<Vec, dtype, tM, tN, BLayout::RowMajor, tM, rmd_N>;

// Corner tile
using TileDataCor  = Tile<Vec, dtype, tM, tN, BLayout::RowMajor, rmd_M, rmd_N>;

// In the kernel body:
for (int i = 0; i < Mb; ++i) {
    TileData data;
    // ... process full tile ...
}
if constexpr (rmd_M > 0) {
    TileDataRow data;    // valid region = rmd_M × tN
    // ... process tail ...
}
```

### 4.6 Test Driver Pattern

```cpp
#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "fileop.h"
#include "matmul/matmul.hpp"          // kernel header

#define globM 256    // overridable via -DglobM=...
#define globN 256
#define globK 256
#define tilM  16
#define tilN  16
#define tilK  16

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4096

int main() {
    // 4KB-aligned stack buffers (required for DMA tile loads)
    uint8_t a_buf[globM * globK * sizeof(__half) + 2 * ALIGN];
    __half *a = (__half *)(((uint64_t)a_buf & ALIGN_MASK) + ALIGN);
    // ... similarly for b, c ...

    BENCHSTART;
    matmul<__half, globM, globN, globK, tilM, tilN, tilK>(c, a, b);
    BENCHEND;

    return 0;
}
```

---

## 5. Compilation & Toolchain

### 5.1 Toolchain

Build the Linx toolchain from `linx-toolchain-build`:

```bash
cd linx-toolchain-build
make init-src
make WITH_TARGET=linx64v5-linux-musl
export COMPILER_DIR=$(pwd)/output/linx_blockisa_llvm_musl/bin
```

### 5.2 Compiler Flags

```bash
$COMPILER_DIR/clang++ \
    -mlxbc -fenable-matrix -O2 \
    -mllvm -enable-all-vector-as-tilereg=true \
    -mllvm -linxv5-enable-HL-Inst-Opt=true \
    -mllvm -linxv5-enable-dim-opt=true \
    -mllvm -linxv5-enable-ldst-bridge=false \
    -mllvm -linxv5-enable-continuous-mem-opt=true \
    -mllvm -linxv5-enable-tile-clock-hand=false \
    -mllvm -linxv5-enable-simt-clock-hand=true \
    -mllvm -enable-misched=false \
    -std=c++20 \
    -D__linx -DENABLE_TENSOR_INSTR \
    -I<repo>/benchmark/one-level-arch/include \
    -I<repo>/benchmark/one-level-arch/test/common \
    -I<repo>/benchmark/one-level-arch/kernels \
    kernel.cpp -nostartfiles _start.s -o kernel.elf
```

| Flag | Purpose |
|------|---------|
| `-mlxbc` | Enable BlockISA code generation |
| `-fenable-matrix` | Enable matrix/tile intrinsics |
| `-std=c++20` | Required for concepts and `[[maybe_unused]]` |
| `-D__linx` | Select Linx backend (jcore/ inline asm) |
| `-DENABLE_TENSOR_INSTR` | Enable tensor instruction definitions |
| `-O2` | Optimization level |
| `-mllvm -enable-all-vector-as-tilereg=true` | Treat all vectors as tile registers |

### 5.3 Makefile Build System

```bash
cd benchmark/one-level-arch/test/kernel/matmul
make TESTCASE=matmul TYPE=MASK MODE=MASK_FP32 M=256 N=256 K=256 tM=16 tN=16 tK=16
make TESTCASE=matmul ... diss    # also generate disassembly
```

### 5.4 Running on Simulator

```bash
# Functional model (correctness)
bin/gfrun -f kernel.elf

# Cycle-accurate timing model
bin/gfsim -f kernel.elf
```

---

## 6. Best Practices & Optimization

### 6.1 Register Pressure Management

- Prefer `MATMACC` (fused accumulate) over separate `TMUL` + `TADD`
- Use `TileAcc<float>` for accumulation — FP32 accumulator avoids precision loss
- Keep live tile count within the 64-entry naming window and 256 KB capacity
- Distribute long-lived tiles across T/U/M/N queues to avoid window overflow

### 6.2 Tile Reuse

- Load A tiles once and reuse across N-dimension iterations (A-tile reuse)
- Use `TCOPY` to duplicate tiles when the source is needed by multiple consumers
- Minimize `TLOAD`/`TSTORE` — register-resident compute is much faster than DMA

### 6.3 Tail Handling

- Use `if constexpr (rmd > 0)` to elide empty tail paths at compile time
- Declare dedicated `ValidRow`/`ValidCol` tile types for each tail quadrant
- The 4-quadrant pattern: full / col-tail / row-tail / corner

### 6.4 Layout Selection

- GEMM operands must use `TileLeft`/`TileRight` (boxed fractal, 512B)
- Accumulators must use `TileAcc` (boxed fractal, 1024B, FP32)
- Elementwise/reduction/load-store: plain `Tile<Vec, ...>` with `NoneBox`
- Match global memory layout (`RowMajor`/`ColMajor`) to avoid implicit transposes

### 6.5 Alignment

- Global memory buffers must be **4 KB aligned** for DMA tile loads
- Tile columns must be 32-byte aligned (e.g. `Npc % 16 == 0` for bf16)
- Use the `ALIGN_MASK + ALIGN` idiom for stack buffers

### 6.6 Static Shapes

- Pass all matrix dimensions and tile sizes as **template parameters**
- This enables full loop unrolling and compile-time register allocation
- Use `constexpr` arithmetic for derived values (`Mb = gM / tM`, `rmd = gM % tM`)

### 6.7 Known Toolchain Limitations

| Limitation | Workaround |
|------------|------------|
| `TABS` not exposed | Emulate: `TMULs(neg, x, -1); TROWMAX(pos, x); TROWMAX(neg, neg); TMAX(absmax, pos, neg);` |
| `TNEG` not exposed | `TMULs(dst, src, -1.0)` |
| `TEXPANDSCALAR` not in wrapper API | Use `TEXPANDSCALAR_Impl()` directly (include `jcore/TExpandScalar.hpp`) |
| `TMATMUL_ACC` may crash on tile spill | Use fresh `TMATMUL` + explicit `TADD` accumulation if register pressure is high |
| Broadcast ops (`TROWEXPAND*`) static_assert bug under `__linx` | Use `_TEPL` inline-asm variants from `template_asm.h` |

---

## 7. Appendix

### 7.1 Complete Operation Index

#### Matrix (CUBE family)

| Operation | Signature | Description |
|-----------|-----------|-------------|
| `MATMUL` | `(C&, A&, B&)` | `C = A × B` |
| `MATMACC` | `(C&, A&, B&)` | `C += A × B` |
| `MATMULMX` | `(C&, A&, AX&, B&, BX&)` | MX GEMM (A+B scale) |
| `MATMULMXB` | `(C&, A&, B&, BX&)` | MX GEMM (B-only scale) |
| `MATMACCMX` | `(C&, A&, AX&, B&, BX&)` | Fused MAC MX (A+B) |
| `MATMACCMXB` | `(C&, A&, B&, BX&)` | Fused MAC MX (B-only) |
| `ACCCVT` | `(tileVec&, tileAcc&)` | Acc → Vec convert |

#### Memory (TLSU family)

| Operation | Description |
|-----------|-------------|
| `TCOPYIN` / `TLOAD` | Global → Tile |
| `TCOPYOUT` / `TSTORE` | Tile → Global |
| `TCOPY` | Tile → Tile |
| `MGATHER` | Scatter-load by offsets |
| `MSCATTER` | Scatter-store by offsets |

#### Elementwise Binary (TEPL family)

`TADD`, `TSUB`, `TMUL`, `TDIV`, `TMAX`, `TMIN`, `TAND`, `TOR`, `TXOR`,
`TCMP`, `TADDS`, `TSUBS`, `TMULS`, `TDIVS`, `TMAXS`, `TMINS`,
`TADDC`, `TSUBC` (scalar operand variants)

#### Elementwise Unary

`TAbs`, `TExp`, `TLog`, `TSqrt`, `TRSqrt`, `TRecip`, `TNeg`, `TNot`,
`TCvt`, `TCast`, `TReshape`, `TTrans`, `TSelect`

#### Broadcast / Fill

`TEXPANDS`, `TCI`, `TROWEXPAND{ADD,SUB,MUL,DIV,MAX,MIN,EXPDIF}`,
`TCOLEXPAND{ADD,SUB,MUL,DIV,MAX,MIN,EXPDIF}`, `TCONCAT`

#### Reductions

`TROWSUM`, `TCOLSUM`, `TROWMAX`, `TCOLMAX`, `TROWMIN`, `TCOLMIN`,
`TROWARGMAX`, `TROWARGMIN`, `TCOLARGMAX`, `TCOLARGMIN`,
`TPARTADD`, `TPARTMUL`, `TPARTMAX`, `TPARTMIN`

#### Shift / Extract / Special

`TSHL`, `TSHR`, `TSLL`, `TSRL`, `TSHLS`, `TSHRS`,
`TEXTRACT`, `TINSERT`, `TFILLPAD`, `TASSEMBLE`,
`TSEL`, `TSELS`, `THISTOGRAM`, `TMRGSORT`, `TIMG2COL`

### 7.2 Location Enum

```cpp
enum class Location {
    Vec,      // General-purpose elementwise tile
    Mat,      // Matrix tile (L1)
    Left,     // GEMM A-operand (L0A)
    Right,    // GEMM B-operand (L0B)
    Acc,      // Accumulator (L0C, FP32)
    Bias,     // Bias tile
    Scaling   // Scaling factor tile
};
```

### 7.3 Layout Enums

```cpp
enum class BLayout  { RowMajor, ColMajor };           // Block-level (outer)
enum class SLayout  { NoneBox, RowMajor, ColMajor };   // Sub-/fractal (inner)
enum class PadValue { Zero=0, Max=1, Min=2, Null=3 };
enum class CmpMode  { EQ, NE, GT, LT, GE, LE };
```

### 7.4 Tile Layout Cheat Sheet

| Tile Alias | Location | BFractal | SFractal | SFractalSize | Use |
|------------|----------|----------|----------|---------------|-----|
| `TileLeft` | Left | ColMajor | RowMajor | 512 | GEMM A |
| `TileRight` | Right | RowMajor | ColMajor | 512 | GEMM B |
| `TileAcc` | Acc | ColMajor | RowMajor | 1024 | GEMM C (FP32) |
| `Tile<Vec, ..., NoneBox>` | Vec | any | NoneBox | 512 | Elementwise |

### 7.5 Capacity Quick Reference

```
Per-PE:  256 KB CELL register file (2048 × 128 B)
         64 named TReg entries (4 queues × 16)
Per-core: 1 MB aggregate (4 PE)

Tile size: 128 B – 8 KB (imm4 = 3..9)
CELL:     128 B minimum granularity

Constraint: sum(live tile bytes) ≤ 256 KB AND live named entries ≤ 64
```

### 7.6 File Organization

```
benchmark/one-level-arch/
├── include/common/pto_tileop.hpp    ← Public API (include this)
├── kernels/                          ← Header-only kernel implementations
│   ├── matmul/matmul.hpp
│   ├── fa/sfa_pto.hpp
│   ├── reduction/reducesum_colvec_pto.hpp
│   ├── transpose/transpose_pto.hpp
│   └── deepseek/                     ← 19 migrated kernels
├── test/kernel/                      ← Test drivers + build system
│   ├── common/Makefile.common        ← Shared build rules
│   ├── matmul/{Makefile, compile.all, src/}
│   ├── fa/{Makefile, compile.all, src/}
│   └── deepseek/{Makefile, compile.all, src/}
└── compile_all.sh                    ← Top-level: compiles all operators
```
