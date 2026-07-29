#include <common/pto_tileop.hpp>
#include <cstdint>
#include "deepseek/moe/expand_to_fused_pto.hpp"
using namespace supernpu::tile_isa;
int main() {
    [[maybe_unused]] auto fn = &expand_to_fused<__bf16, 16, 64, 4, 32, 64>;
    return 0;
}
