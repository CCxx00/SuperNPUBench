#include <common/pto_tileop.hpp>
#include <cstdint>
#include "deepseek/moe/mask_indices_by_tp_pto.hpp"
using namespace supernpu::tile_isa;
int main() {
    [[maybe_unused]] auto fn = &mask_indices_by_tp<16, 8>;
    return 0;
}
