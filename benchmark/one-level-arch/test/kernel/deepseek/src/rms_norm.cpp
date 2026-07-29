#include <common/pto_tileop.hpp>
#include <cstdint>
#include "deepseek/mhc/norm_fn_pto.hpp"
using namespace supernpu::tile_isa;
int main() {
    [[maybe_unused]] auto fn = &rms_norm<16, 8>;
    return 0;
}
