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
void tadd_accum_unit(float* out_ptr) {
    using tileT = Tile<Location::Vec, float, Rows, Cols, BLayout::ColMajor>;
    using gmOut = global_tensor<float, RowMajor<Rows, Cols>>;
    using itOut = global_iterator<gmOut, tileT>;

    tileT tO;
    tileT tPV;
    TEXPANDS(tO, 0.0f);
    TEXPANDS(tPV, 0.25f);
    TADD(tO, tO, tPV);
    TADD(tO, tO, tPV);
    TADD(tO, tO, tPV);
    TADD(tO, tO, tPV);

    itOut out_iter(out_ptr);
    auto dst = out_iter(0, 0);
    TSTORE(dst, tO);
}

int main() {
    float out_buf[kTileRows * kTileCols + 2 * ALIGN];
    float* out = (float*)(((uint64_t)out_buf & ALIGN_MASK) + ALIGN);

    BENCHSTART;
    tadd_accum_unit<kTileRows, kTileCols>(out);
    BENCHEND;

#ifdef RES_CHECK
#define OUT_PATH CHK_DIR "/tadd_accum_out.bin"
    writeBinaryFile(OUT_PATH, (uint8_t*)out, kTileRows * kTileCols * sizeof(float));
#endif

    return 0;
}
