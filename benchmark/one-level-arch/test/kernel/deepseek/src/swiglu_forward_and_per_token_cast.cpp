#include <common/pto_tileop.hpp>
#include <cstdint>
#include "deepseek/quant/swiglu_fused_cast_pto.hpp"
using namespace supernpu::tile_isa;
int main() {
    [[maybe_unused]] auto fn = &swiglu_forward_and_per_token_cast<16, 16, 16>;
    return 0;
}
