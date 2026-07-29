#include <common/pto_tileop.hpp>
#include <cstdint>
#include "deepseek/quant/per_token_cast_pto.hpp"
using namespace supernpu::tile_isa;
int main() {
    [[maybe_unused]] auto fn = &per_token_cast<16, 16>;
    return 0;
}
