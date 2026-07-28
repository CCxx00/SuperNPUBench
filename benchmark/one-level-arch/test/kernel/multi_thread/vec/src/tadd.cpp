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
void vec_multithread(float* out_ptr, float* a_ptr, float* b_ptr) {
    using tileT = Tile<Location::Vec, float, Rows, Cols, BLayout::RowMajor>;
    using gmIn = global_tensor<float, RowMajor<Rows, Cols>>;
    using gmOut = global_tensor<float, RowMajor<Rows, Cols>>;
    using itIn = global_iterator<gmIn, tileT>;
    using itOut = global_iterator<gmOut, tileT>;
    uint32_t tid = get_thread_id();
    uint32_t gm_offset = tid * Rows*Cols;

    tileT tA;
    tileT tB;
    tileT tC;

    itIn a_iter(a_ptr+gm_offset);
    itIn b_iter(b_ptr+gm_offset);
    auto src_a = a_iter(0, 0);
    auto src_b = b_iter(0, 0);
    TLOAD(tA, src_a);
    TLOAD(tB, src_b);
    TADD(tC, tA, tB);

    itOut out_iter(out_ptr+gm_offset);
    auto dst = out_iter(0, 0);
    TCOPYOUT(dst, tC);
}

int main() {
    float a_buf[kTileRows * kTileCols + 2 * ALIGN];
    float b_buf[kTileRows * kTileCols + 2 * ALIGN];
    float out_buf[kTileRows * kTileCols + 2 * ALIGN];
    float* a = (float*)(((uint64_t)a_buf & ALIGN_MASK) + ALIGN);
    float* b = (float*)(((uint64_t)b_buf & ALIGN_MASK) + ALIGN);
    float* out = (float*)(((uint64_t)out_buf & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
#define SRC_A_PATH CHK_DIR "/src_a.bin"
#define SRC_B_PATH CHK_DIR "/src_b.bin"
    readBinaryFile(SRC_A_PATH, (uint8_t*)a, kTileRows * kTileCols * sizeof(float));
    readBinaryFile(SRC_B_PATH, (uint8_t*)b, kTileRows * kTileCols * sizeof(float));
#else
    for (int i = 0; i < kTileRows * kTileCols; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
#endif

    BENCHSTART;
    vec_multithread<kTileRows/4, kTileCols>(out, a, b);
    BENCHEND;

#ifdef RES_CHECK
#define OUT_PATH CHK_DIR "/vec_out.bin"
    writeBinaryFile(OUT_PATH, (uint8_t*)out, kTileRows * kTileCols * sizeof(float));
#endif

    return 0;
}
