#include <common/pto_tileop.hpp>
#include <cstdint>
#include "deepseek/engram/fused_weight_pto.hpp"
using namespace supernpu::tile_isa;
int main() {
    [[maybe_unused]] auto fn = &fused_weight<2, 64, 64>;
    return 0;
}
