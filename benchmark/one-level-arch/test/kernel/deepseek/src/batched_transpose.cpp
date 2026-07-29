#include <common/pto_tileop.hpp>
#include <cstdint>
#include "deepseek/transpose/batched_transpose_pto.hpp"
using namespace supernpu::tile_isa;
int main() {
    [[maybe_unused]] auto fn = &batched_transpose<float, 2, 16, 16>;
    return 0;
}
