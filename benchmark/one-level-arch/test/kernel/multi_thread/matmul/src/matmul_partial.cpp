#include <common/pto_tileop.hpp>
#include <cstdint>
#include "benchmark.h"
#include "fileop.h"

using namespace pto;

#ifndef globM
#define globM 256
#endif

#ifndef globN
#define globN 256
#endif

#ifndef globK
#define globK 256
#endif

#ifndef tilM
#define tilM 16
#endif

#ifndef tilN
#define tilN 32
#endif

#ifndef tilK
#define tilK 32
#endif

#ifndef Batch
#define Batch 1
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN (4 * 1024)

template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_partial_multithread(float *c_ptr, dtype *a_ptr, dtype *b_ptr) {
    constexpr int kTileByteLimit = 4 * 1024;

    static_assert(gM % tM == 0);
    static_assert(gN % tN == 0);
    static_assert(gK % tK == 0);
    static_assert(tM * tK * sizeof(dtype) <= kTileByteLimit);
    static_assert(tM * tN * sizeof(float) <= kTileByteLimit);
    static_assert(tK * tN * sizeof(dtype) <= kTileByteLimit);

    const uint32_t tid = get_thread_idx();
    a_ptr += tid * gM * gK;
    c_ptr += tid * gM * gN;

    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gK, gN>>;
    using tileA = TileLeft<dtype, tM, tK>;
    using tileCAcc = TileAcc<float, tM, tN>;
    using tileB = TileRight<dtype, tK, tN>;
    using itA = global_iterator<gmA, tileA>;
    using itB = global_iterator<gmB, tileB>;

    itA gIterA(a_ptr);
    itB gIterB(b_ptr);

    constexpr int Mb = gM / tM;
    constexpr int Nb = gN / tN;
    constexpr int Kb = gK / tK;

    for (int i = 0; i < Mb; ++i) {
        for (int j = 0; j < Nb; ++j) {
            tileCAcc tCAcc;
            for (int k = 0; k < Kb; ++k) {
                tileA tA;
                tileB tB;
                auto gA = gIterA(i, k);
                auto gB = gIterB(k, j);
                TLOAD(tA, gA);
                TLOAD(tB, gB);
                if (k == 0) {
                    TMATMUL(tCAcc, tA, tB);
                } else {
                    TMATMUL_ACC(tCAcc, tA, tB);
                }
            }
            // Partial variant: keep the result in ACC and intentionally skip
            // conversion and storage to global memory.
        }
    }
}

int main() {
    using dtype = float;
    constexpr int kPeNum = 4;
    static_assert(globM % kPeNum == 0);
    constexpr int M = globM / kPeNum;

    dtype src0p[Batch * globM * globK + 2 * ALIGN];
    dtype src1p[Batch * globK * globN + 2 * ALIGN];
    float dstp[Batch * globM * globN + 2 * ALIGN];
    dtype *src0 = (dtype *)(((uint64_t)src0p & ALIGN_MASK) + ALIGN);
    dtype *src1 = (dtype *)(((uint64_t)src1p & ALIGN_MASK) + ALIGN);
    float *dst = (float *)(((uint64_t)dstp & ALIGN_MASK) + ALIGN);

    for (int i = 0; i < Batch * globM * globK; ++i) {
        src0[i] = 1.0f;
    }
    for (int i = 0; i < Batch * globK * globN; ++i) {
        src1[i] = 2.0f;
    }

    BENCHSTART;
    for (int b = 0; b < Batch; ++b) {
        matmul_partial_multithread<dtype, M, globN, globK, tilM, tilN, tilK>(
            dst + b * globM * globN,
            src0 + b * globM * globK,
            src1 + b * globK * globN);
    }
    BENCHEND;
    return 0;
}
