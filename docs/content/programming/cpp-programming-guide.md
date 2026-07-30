# C++ Tile Kernel Programming Guide

This chapter is the compact, end-to-end reference for writing a kernel in the
current SuperNPUBench one-level programming profile. Use the focused chapters
for full operation semantics and worked algorithms.

## Programming surface

A kernel is a C++20 function over typed global tensor views and statically
shaped tiles. Scalar C++ chooses coordinates, loop bounds, and specializations.
Tile operations carry the tensor data path.

```cpp
template <int Rows, int Cols, int TileRows, int TileCols>
void add(float* out, float* lhs, float* rhs) {
    using Global = global_tensor<float, RowMajor<Rows, Cols>>;
    using Tile = LocalTile<float, TileRows, TileCols>;
    using Iterator = global_iterator<Global, Tile>;

    Iterator out_tiles(out);
    Iterator lhs_tiles(lhs);
    Iterator rhs_tiles(rhs);

    for (int row = 0; row < Rows / TileRows; ++row) {
        for (int col = 0; col < Cols / TileCols; ++col) {
            Tile a;
            Tile b;
            Tile c;
            TLOAD(a, lhs_tiles(row, col));
            TLOAD(b, rhs_tiles(row, col));
            TADD(c, a, b);
            TSTORE(out_tiles(row, col), c);
        }
    }
}
```

The current headers use `pto` as the literal C++ namespace. Reader-facing
examples use the public tile aliases so backend-specific implementation types
do not obscure the dataflow.

## Tiles and tensor views

`global_tensor` describes element type, rank, shape, and layout; it does not
copy data. `global_iterator` maps a compile-time tile shape to a runtime tile
coordinate. A local tile belongs to one execution thread. A shared tile is
visible to the participating threads in a block.

The active FlashAttention and matrix-multiplication cases keep every
materialized tile at or below **4 KiB**. Small vector-style kernels may use an
exact **128-byte** tile. Compute capacity from the physical tile shape, not
only the valid region:

```cpp
static_assert(TileRows * TileCols * sizeof(float) <= 4 * 1024);
```

Valid rows and columns may be smaller than physical rows and columns. Use the
valid region for tails; padding lanes must not change observable results.

## Operation form

Tile operations follow destination-first C++ syntax:

```cpp
TLOAD(destination_tile, source_region);
TADD(destination_tile, lhs_tile, rhs_tile);
TMATMUL_ACC(accumulator_tile, lhs_matrix_tile, rhs_matrix_tile);
TSTORE(destination_region, source_tile);
```

`TLOAD` and `TSTORE` are the canonical memory-operation names. Do not use the
superseded copy-in or copy-out spellings in new kernels.

Every write defines a new version of the destination tile ID. The scheduler
uses tile IDs and versions to preserve dependencies while issuing independent
tile operations out of order. Kernels do not insert event objects or explicit
tile synchronization calls.

## Blocks and threads

Use `get_block_idx()` to choose a global tensor partition and
`get_thread_idx()` to choose a disjoint fragment within that partition:

```cpp
constexpr uint32_t kThreadsPerBlock = 4;
const uint32_t block = get_block_idx();
const uint32_t thread = get_thread_idx();
const uint32_t tile_row = block * kThreadsPerBlock + thread;
```

All threads enter at the same kernel function. Branches may select different
work, but convergent code must not read a shared value until every producer
required by that operation has made its version visible.

## Matrix multiplication

Keep the left operand and accumulator local. The right operand may be loaded
into shared tile storage when all threads reuse the same panel:

```cpp
LocalTile<half, kM, kK> left;
SharedTile<half, kK, kN> right;
LocalTile<float, kM, kN> accum;

TLOAD(left, a_tiles(thread, k_panel));
TLOAD(right, b_tiles(k_panel, n_panel));
TMATMUL_ACC(accum, left, right);
```

Choose `kM`, `kN`, and `kK` so each physical tile respects the 4 KiB profile
limit and the source dimensions are divisible or represented with valid-region
metadata.

## Build the current toolchain

The benchmark build uses the complete LLVM + musl toolchain produced by
`LinxISA/linx-toolchain-build`. That repository pins the compiler, C++ runtime,
sysroot, and tile headers as one install tree.

```console
git clone https://github.com/LinxISA/linx-toolchain-build.git
cd linx-toolchain-build
make init-src
make WITH_TARGET=linx64v5-linux-musl
export COMPILER_DIR="$PWD/output/linx_blockisa_llvm_musl/bin"
"$COMPILER_DIR/clang" --version
```

The current build chain tracks LLVM branch `dev-llvm15_56` and targets
`linx64v5-unknown-linux-musl`. Use `gmake` on macOS when GNU Make 4 or newer is
not the default.

## Compile and inspect a kernel

```console
cd benchmark/one-level-arch/test/kernel/fa
make TESTCASE=fa_2d_unroll Sq=256 Skv=512 Tm=8 Tk=16 X=1 Y=2
make TESTCASE=fa_2d_unroll Sq=256 Skv=512 Tm=8 Tk=16 X=1 Y=2 diss
```

Generated ELF files and disassembly are written below
`benchmark/one-level-arch/output/`. The [benchmark catalog](../benchmarks/index.md)
lists every active command and links it to the source reached by the build.

## Continue reading

- [Compile-time shapes and templates](../cpp/static-programming.md)
- [Multidimensional tiling](../tutorials/multidimensional-tiling.md)
- [Fine-grained 128-byte tiles](../tutorials/fine-grained-tiles.md)
- [Superscalar tile dependencies](../model/execution.md)
- [Intrinsic reference](../intrinsics/index.md)
- [FlashAttention guide](../tutorials/flash-attention.md)
