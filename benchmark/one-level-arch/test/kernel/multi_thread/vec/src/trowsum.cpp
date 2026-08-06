#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "fileop.h"

#include <cstdint>

using namespace pto;

#ifndef TileRows
#define kTileRows 16
#else
#define kTileRows TileRows
#endif

#ifndef TileCols
#define kTileCols 16
#else
#define kTileCols TileCols
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4 * 1024

template <int Rows, int Cols>
void trowsum_multithread(float *out_ptr, float *in_ptr) {
    constexpr int kTileByteLimit = 4 * 1024;

    static_assert(Rows * Cols * sizeof(float) <= kTileByteLimit,
                  "each PE input tile must not exceed 4 KiB");
    static_assert(Rows * 8 * sizeof(float) <= kTileByteLimit,
                  "each PE row-sum tile must not exceed 4 KiB");

    using tileIn =
        Tile<Location::Vec, float, Rows, Cols, BLayout::RowMajor>;
    // TROWSUM produces one value per row. Keep 8 physical columns for Tile
    // alignment while exposing only column 0 as valid data.
    using tileSum =
        Tile<Location::Vec, float, Rows, 8, BLayout::RowMajor, Rows, 1>;
    using gmIn = global_tensor<float, RowMajor<Rows, Cols>>;
    using gmOut = global_tensor<float, RowMajor<Rows, 1>>;
    using itIn = global_iterator<gmIn, tileIn>;
    using itOut = global_iterator<gmOut, tileSum>;

    const uint32_t tid = get_thread_idx();
    const uint32_t in_offset = tid * Rows * Cols;
    const uint32_t out_offset = tid * Rows;

    itIn in_iter(in_ptr + in_offset);
    itOut out_iter(out_ptr + out_offset);

    tileIn tIn;
    tileSum tSum;
    auto src = in_iter(0, 0);
    auto dst = out_iter(0, 0);
    TLOAD(tIn, src);
    TROWSUM(tSum, tIn);
    TSTORE(dst, tSum);
}

int main() {
    static_assert(kTileRows % 4 == 0,
                  "TileRows must be divisible by the four PE threads");

    float in_buf[kTileRows * kTileCols + 2 * ALIGN];
    float out_buf[kTileRows + 2 * ALIGN];
    float *in = (float *)(((uint64_t)in_buf & ALIGN_MASK) + ALIGN);
    float *out = (float *)(((uint64_t)out_buf & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
#define SRC_PATH CHK_DIR "/src.bin"
    readBinaryFile(
        SRC_PATH, (uint8_t *)in,
        kTileRows * kTileCols * sizeof(float));
#else
    for (int i = 0; i < kTileRows * kTileCols; ++i) {
        in[i] = 1.0f;
    }
#endif

    BENCHSTART;
    trowsum_multithread<kTileRows / 4, kTileCols>(out, in);
    BENCHEND;

#ifdef RES_CHECK
#define OUT_PATH CHK_DIR "/trowsum_out.bin"
    writeBinaryFile(
        OUT_PATH, (uint8_t *)out, kTileRows * sizeof(float));
#endif

    return 0;
}
