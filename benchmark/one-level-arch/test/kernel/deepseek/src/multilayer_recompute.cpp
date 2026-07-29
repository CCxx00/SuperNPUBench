#include <common/pto_tileop.hpp>
#include <cstdint>
#include "deepseek/mhc/multilayer_recompute_pto.hpp"
using namespace supernpu::tile_isa;
int main() {
    [[maybe_unused]] auto fn = &multilayer_recompute<16, 16, 2>;
    return 0;
}
