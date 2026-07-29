#include <common/pto_tileop.hpp>
#include <cstdint>
#include "deepseek/quant/per_token_cast_pto.hpp"
using namespace supernpu::tile_isa;
int main() {
    [[maybe_unused]] auto fn = &per_channel_cast<16, 32, 32>;
    return 0;
}
