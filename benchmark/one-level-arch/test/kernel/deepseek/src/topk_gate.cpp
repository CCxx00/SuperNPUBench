#include <common/pto_tileop.hpp>
#include <cstdint>
#include "deepseek/moe/topk_gate_pto.hpp"
using namespace supernpu::tile_isa;
int main() {
    [[maybe_unused]] auto fn = &topk_gate<16, 32, 4>;
    return 0;
}
