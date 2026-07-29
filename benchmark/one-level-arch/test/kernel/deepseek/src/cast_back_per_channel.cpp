#include <common/pto_tileop.hpp>
#include <cstdint>
#include "deepseek/quant/cast_back_pto.hpp"
using namespace supernpu::tile_isa;
int main() {
    [[maybe_unused]] auto fn = &cast_back_per_channel<__bf16, 16, 32, 16, 32>;
    return 0;
}
