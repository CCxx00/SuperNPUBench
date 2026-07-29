#include <common/pto_tileop.hpp>
#include <cstdint>
#include "deepseek/quant/cast_back_pto.hpp"
using namespace supernpu::tile_isa;
int main() {
    [[maybe_unused]] auto fn = &cast_back_per_token<__bf16, 16, 16>;
    return 0;
}
