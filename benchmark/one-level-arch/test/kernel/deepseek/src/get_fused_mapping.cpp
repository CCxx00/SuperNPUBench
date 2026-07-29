#include <common/pto_tileop.hpp>
#include <cstdint>
#include "deepseek/moe/get_fused_mapping_pto.hpp"
using namespace supernpu::tile_isa;
int main() {
    [[maybe_unused]] auto fn = &get_fused_mapping<16, 8, 32>;
    return 0;
}
