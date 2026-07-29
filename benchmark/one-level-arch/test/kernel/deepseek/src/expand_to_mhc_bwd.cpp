#include <common/pto_tileop.hpp>
#include <cstdint>
#include "deepseek/mhc/expand_to_mhc_bwd_pto.hpp"
using namespace supernpu::tile_isa;
int main() {
    [[maybe_unused]] auto fn = &expand_to_mhc_bwd<16, 64, 16, 64>;
    return 0;
}
