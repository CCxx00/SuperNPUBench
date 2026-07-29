#include <common/pto_tileop.hpp>
#include <cstdint>
#include "deepseek/moe/group_count_aux_fi_pto.hpp"
using namespace supernpu::tile_isa;
int main() {
    [[maybe_unused]] auto fn = &group_count<16, 8, 32>;
    return 0;
}
