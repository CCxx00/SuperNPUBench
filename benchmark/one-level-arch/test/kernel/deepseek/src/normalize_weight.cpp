#include <common/pto_tileop.hpp>
#include <cstdint>
#include "deepseek/moe/normalize_weight_pto.hpp"
using namespace supernpu::tile_isa;
int main() {
    [[maybe_unused]] auto fn = &normalize_weight<16, 8>;
    return 0;
}
