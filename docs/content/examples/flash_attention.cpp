#include <pto_kernel/tile.hpp>

using namespace pto;

namespace {

constexpr int kSequence = 16;
constexpr int kHead = 16;
constexpr int kQueryRows = 16;
constexpr int kKeyRows = 16;

using QueryTile = MatrixLeftTile<int, kQueryRows, kHead>;
using KeyTile = MatrixRightTile<int, kHead, kKeyRows>;
using ValueTile = MatrixRightTile<int, kKeyRows, kHead>;
using ScoreTile = AccumulatorTile<int, kQueryRows, kKeyRows>;
using ScoreValues = LocalTile<int, kQueryRows, kKeyRows>;
using ScoreLeft = MatrixLeftTile<int, kQueryRows, kKeyRows>;
using OutputTile = AccumulatorTile<int, kQueryRows, kHead>;
using OutputValues = LocalTile<int, kQueryRows, kHead>;

using Queries = global_tensor<int, RowMajor<kSequence, kHead>>;
using Keys = global_tensor<int, ColMajor<kHead, kSequence>>;
using Values = global_tensor<int, ColMajor<kSequence, kHead>>;
using Output = global_tensor<int, RowMajor<kSequence, kHead>>;

} // namespace

extern "C" void flash_attention_probe(int *out, int *q, int *k, int *v) {
  global_iterator<Queries, QueryTile> query_tiles(q);
  global_iterator<Keys, KeyTile> key_tiles(k);
  global_iterator<Values, ValueTile> value_tiles(v);
  global_iterator<Output, OutputValues> output_tiles(out);

  for (int query_block = 0; query_block < kSequence / kQueryRows;
       ++query_block) {
    QueryTile query;
    OutputValues running;
    auto query_view = query_tiles(query_block, 0);
    TLOAD(query, query_view);

    for (int key_block = 0; key_block < kSequence / kKeyRows; ++key_block) {
      KeyTile key;
      ValueTile value;
      ScoreTile score_acc;
      ScoreValues score;
      ScoreLeft score_left;
      OutputTile output_acc;
      OutputValues contribution;
      auto key_view = key_tiles(0, key_block);
      auto value_view = value_tiles(key_block, 0);

      TLOAD(key, key_view);
      TLOAD(value, value_view);
      TMATMUL(score_acc, query, key);
      TCVT(score, score_acc);
      TCVT(score_left, score);
      TMATMUL(output_acc, score_left, value);
      TCVT(contribution, output_acc);

      if (key_block == 0) {
        TMOV(running, contribution);
      } else {
        TADD(running, running, contribution);
      }
    }

    auto output_view = output_tiles(query_block, 0);
    TSTORE(output_view, running);
  }
}
