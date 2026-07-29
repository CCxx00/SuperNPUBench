#include <common/pto_tileop.hpp>
#include <cstdint>
#include "deepseek/moe/inplace_unique_group_indices_pto.hpp"
using namespace supernpu::tile_isa;
int main() {
    [[maybe_unused]] auto fn = &inplace_unique_group_indices<16, 8, 8>;
    return 0;
}
