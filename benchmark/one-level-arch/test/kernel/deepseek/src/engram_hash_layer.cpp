#include <common/pto_tileop.hpp>
#include <cstdint>
#include "deepseek/engram/engram_hash_pto.hpp"
using namespace supernpu::tile_isa;
int main() {
    [[maybe_unused]] auto fn = &engram_hash_layer<16, 8, 8, 8>;
    return 0;
}
