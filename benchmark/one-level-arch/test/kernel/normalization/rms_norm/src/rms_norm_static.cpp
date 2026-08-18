#include <common/pto_tileop.hpp>

#include <cstdint>

#include "fileop.h"
#include "normalization/rms_norm/rms_norm_pto.hpp"

#ifndef DType
#define DType __half
#endif

#ifndef EPS
#define EPS 1e-6f
#endif

// Same as dynamic rms_norm.cpp tiling_info {16,512,2,512}
#ifndef G_A
#define G_A 16
#endif
#ifndef G_R
#define G_R 512
#endif
#ifndef TILE_A
#define TILE_A 2
#endif
#ifndef TILE_R
#define TILE_R 512
#endif

int main() {
    using dtype = DType;

    dtype input_buf[G_A * G_R];
    dtype output_buf[G_A * G_R];
    dtype *input = input_buf;
    dtype *output = output_buf;

#ifdef RES_CHECK
#ifndef CHK_DIR
#error "CHK_DIR must be set when RES_CHECK is enabled"
#endif
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t *)input,
                   static_cast<size_t>(G_A) * G_R * sizeof(dtype));
#endif

    rms_norm<dtype, G_A, G_R, TILE_A, TILE_R>(input, output, EPS);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t *)output,
                    static_cast<size_t>(G_A) * G_R * sizeof(dtype));
#endif
}
