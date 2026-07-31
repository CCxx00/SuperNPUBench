#pragma once

#include <common/pto_tileop.hpp>

namespace pto {

template <typename T, int Rows, int Cols,
          BLayout Layout = BLayout::RowMajor, int ValidRows = Rows,
          int ValidCols = Cols>
using LocalTile =
    Tile<Location::Vec, T, Rows, Cols, Layout, ValidRows, ValidCols>;

template <typename T, int Rows, int Cols, int ValidRows = Rows,
          int ValidCols = Cols>
using MatrixLeftTile =
    LocalTile<T, Rows, Cols, BLayout::RowMajor, ValidRows, ValidCols>;

template <typename T, int Rows, int Cols, int ValidRows = Rows,
          int ValidCols = Cols>
using MatrixRightTile =
    LocalTile<T, Rows, Cols, BLayout::RowMajor, ValidRows, ValidCols>;

template <typename T, int Rows, int Cols, int ValidRows = Rows,
          int ValidCols = Cols>
using AccumulatorTile =
    LocalTile<T, Rows, Cols, BLayout::RowMajor, ValidRows, ValidCols>;

// Shared tile types are part of the programming contract, but the current
// compiler lane does not lower shared tile operations yet.
template <typename T, int Rows_, int Cols_> struct SharedTile {
  using DType = T;
  static constexpr int Rows = Rows_;
  static constexpr int Cols = Cols_;
};

enum class SharedMove { Insert, Broadcast };

using ::global_iterator;
using ::TADD;
using ::TADDS;
using ::TCVT;
using ::TLOAD;
using ::TMAXS;
using ::TMULS;
using ::TROWSUM;
using ::TSTORE;

template <typename Dst, typename Src> inline void TMOV(Dst &dst, Src &src) {
  ::TCOPY(dst, src);
}

template <typename Dst, typename Lhs, typename Rhs>
inline void TMATMUL(Dst &dst, Lhs &lhs, Rhs &rhs) {
  ::MATMUL(dst, lhs, rhs);
}

template <typename Dst, typename Previous, typename Lhs, typename Rhs>
inline void TMATMUL_ACC(Dst &dst, Previous &previous, Lhs &lhs, Rhs &rhs) {
  ::TCOPY(dst, previous);
  ::MATMACC(dst, lhs, rhs);
}

} // namespace pto
