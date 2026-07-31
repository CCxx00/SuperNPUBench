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
            auto lhs_view = lhs_tiles(row, col);
            auto rhs_view = rhs_tiles(row, col);
            auto out_view = out_tiles(row, col);
            TLOAD(a, lhs_view);
            TLOAD(b, rhs_view);
            TADD(c, a, b);
            TSTORE(out_view, c);
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

auto left_view = a_tiles(thread, k_panel);
auto right_view = b_tiles(k_panel, n_panel);
TLOAD(left, left_view);
TLOAD(right, right_view);
TMATMUL_ACC(accum, left, right);
```

Choose `kM`, `kN`, and `kK` so each physical tile respects the 4 KiB profile
limit and the source dimensions are divisible or represented with valid-region
metadata.

## Build the current toolchain

Use the LLVM 23 sources pinned by the `linx-isa` superproject. Configure the
LinxISA experimental target and keep the compiler, QEMU, and LinxCoreModel on
the same superproject baseline:

```console
export LINX_ROOT=/path/to/linx-isa
cmake -S "$LINX_ROOT/compiler/llvm/llvm" \
  -B "$LINX_ROOT/compiler/llvm/build-linxisa-clang" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=LinxISA \
  -DLLVM_TARGETS_TO_BUILD=""
cmake --build "$LINX_ROOT/compiler/llvm/build-linxisa-clang" \
  --target clang lld llvm-objdump llvm-objcopy llvm-readobj
export COMPILER_DIR="$LINX_ROOT/compiler/llvm/build-linxisa-clang/bin"
"$COMPILER_DIR/clang" --version
```

See [Set Up the Current Toolchain](../start/toolchain.md) for the complete
compiler, QEMU, and model workflow.

## Compile and inspect a kernel

```console
cd benchmark/one-level-arch/test/kernel/fa
make -n TESTCASE=fa_2d_unroll Sq=256 Skv=512 Tm=8 Tk=16 X=1 Y=2 \
  PLAT=linx COMPILER_DIR="$COMPILER_DIR"
```

The [benchmark catalog](../benchmarks/index.md) lists every active one-level
manifest command and links it to the source reached by the build. Current
compiler and model validation uses the promoted cases described in
[Build, Run, and Inspect Generated Code](../start/build-run.md).

## Continue reading

- [Compile-time shapes and templates](../cpp/static-programming.md)
- [Multidimensional tiling](../tutorials/multidimensional-tiling.md)
- [Fine-grained 128-byte tiles](../tutorials/fine-grained-tiles.md)
- [Superscalar tile dependencies](../model/execution.md)
- [Intrinsic reference](../intrinsics/index.md)
- [FlashAttention guide](../tutorials/flash-attention.md)
